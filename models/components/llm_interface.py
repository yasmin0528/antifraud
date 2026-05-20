"""
LLM 规则生成接口 —— 对应大模型.md Step 1

功能：
- 调用 LLaMA3-8B（本地 4bit 或 API）生成 IF-THEN 规则
- 输出结构化 JSON
- 内置缓存（避免每轮重复调用）
- 内置 fallback（LLM 不可用时使用默认规则模板）
"""

import json
import hashlib
from typing import List, Dict, Optional


DEFAULT_RULES: List[Dict] = [
    {"rule": "IF amount > 10000 AND freq > 5 THEN suspicious", "confidence": 0.8},
    {"rule": "IF circular_transfer_detected THEN suspicious", "confidence": 0.9},
    {"rule": "IF amount > 50000 AND single_transaction THEN suspicious", "confidence": 0.7},
    {"rule": "IF freq > 10 AND same_counterparty THEN suspicious", "confidence": 0.6},
    {"rule": "IF amount > 100000 THEN suspicious", "confidence": 0.85},
    {"rule": "IF amount > 20000 AND tx_type == rapid THEN suspicious", "confidence": 0.75},
    {"rule": "IF layered_transfer_detected THEN suspicious", "confidence": 0.85},
    {"rule": "IF amount_near_threshold AND freq_high THEN suspicious", "confidence": 0.65},
]


class RuleCache:
    """规则缓存，避免重复调用 LLM。"""

    def __init__(self, max_size: int = 50):
        self.cache: Dict[str, List[Dict]] = {}
        self.max_size = max_size

    def _make_key(self, transaction_summary: str) -> str:
        return hashlib.md5(transaction_summary.encode("utf-8")).hexdigest()

    def get(self, transaction_summary: str) -> Optional[List[Dict]]:
        return self.cache.get(self._make_key(transaction_summary))

    def set(self, transaction_summary: str, rules: List[Dict]):
        key = self._make_key(transaction_summary)
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = rules

    def clear(self):
        self.cache.clear()


class LLMInterface:
    """LLaMA3-8B 规则生成接口。"""

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3.2-8B-Instruct",
        use_api: bool = True,
        api_url: Optional[str] = None,
        api_key: Optional[str] = None,
        device: str = "cpu",
        max_retries: int = 3,
    ):
        self.model_name = model_name
        self.use_api = use_api
        self.api_url = api_url or "http://localhost:8000/v1/completions"
        self.api_key = api_key
        self.device = device
        self.max_retries = max_retries
        self._model = None
        self._tokenizer = None
        self.cache = RuleCache()
        self._fallback_count = 0

        if not use_api:
            self._try_load_local()

    def _try_load_local(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                load_in_4bit=True,
                device_map="auto",
                trust_remote_code=True,
            )
        except Exception as e:
            print(f"[LLMInterface] Failed to load local LLM ({e}), will use fallback.")

    def generate_rules(
        self, transaction_summary: str, force_refresh: bool = False
    ) -> List[Dict]:
        if not force_refresh:
            cached = self.cache.get(transaction_summary)
            if cached is not None:
                return cached

        rules = self._call_llm(transaction_summary)

        if not rules:
            rules = self._fallback_rules(transaction_summary)
            self._fallback_count += 1
            if self._fallback_count == 1:
                print(
                    f"[LLMInterface] LLM call failed (api_url={self.api_url}), "
                    f"using {len(rules)} fallback rules. "
                    f"To use LLM, provide a valid --llm_api_url."
                )

        self.cache.set(transaction_summary, rules)
        return rules

    def _call_llm(self, transaction_summary: str) -> List[Dict]:
        prompt = self._build_prompt(transaction_summary)
        if self.use_api:
            return self._call_api(prompt)
        elif self._model is not None:
            return self._call_local(prompt)
        return []

    @staticmethod
    def _build_prompt(transaction_summary: str) -> str:
        return f"""You are an anti-money laundering expert. Given the following transaction patterns:

{transaction_summary}

Generate IF-THEN rules to detect suspicious money laundering behavior.
Each rule should include a condition and a confidence score (0.0 to 1.0).

Rules must cover:
- High-value transactions
- High-frequency transfers
- Circular / layered transfers
- Structuring behavior (amounts near thresholds)
- Rapid succession transactions

Output ONLY a valid JSON array (no extra text):
[
  {{"rule": "IF ... THEN suspicious", "confidence": 0.8}},
  ...
]"""

    def _call_api(self, prompt: str) -> List[Dict]:
        import requests
        is_chat_endpoint = "/v1/chat/completions" in self.api_url
        is_ollama_native = "/api/generate" in self.api_url

        for attempt in range(self.max_retries):
            try:
                headers = {"Content-Type": "application/json"}
                if self.api_key:
                    headers["Authorization"] = f"Bearer {self.api_key}"

                if is_chat_endpoint:
                    # vLLM 的 chat/completions 接口通常接受任意 model 值
                    payload = {
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1024, "temperature": 0.3, "top_p": 0.9,
                        "stream": False,
                    }
                elif is_ollama_native:
                    payload = {
                        "model": self.model_name,
                        "prompt": prompt, "stream": False,
                        "options": {"temperature": 0.3, "top_p": 0.9, "num_predict": 1024},
                    }
                else:
                    payload = {
                        "model": self.model_name, "prompt": prompt,
                        "max_tokens": 1024, "temperature": 0.3, "top_p": 0.9,
                    }

                response = requests.post(
                    self.api_url, json=payload, headers=headers, timeout=120
                )

                if response.status_code == 200:
                    result = response.json()
                    text = ""

                    # 尝试多种响应格式
                    try:
                        text = result["choices"][0]["message"]["content"]
                    except (KeyError, IndexError, TypeError):
                        pass
                    if not text:
                        try:
                            text = result["choices"][0]["text"]
                        except (KeyError, IndexError, TypeError):
                            pass
                    if not text:
                        try:
                            text = result["response"]
                        except (KeyError, TypeError):
                            pass
                    if not text:
                        text = result.get("text", "")

                    if text:
                        parsed = self._parse_rules(text)
                        if parsed:
                            return parsed
                    else:
                        print(f"[LLM API] Attempt {attempt + 1}: empty response, "
                              f"status={response.status_code}, "
                              f"body_keys={list(result.keys()) if isinstance(result, dict) else 'N/A'}")

            except Exception as e:
                print(f"[LLM API] Attempt {attempt + 1} error: {e}")

        return []

    def _call_local(self, prompt: str) -> List[Dict]:
        for attempt in range(self.max_retries):
            try:
                inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self._model.generate(
                    **inputs, max_new_tokens=512,
                    temperature=0.3, do_sample=True, top_p=0.9,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
                response = self._tokenizer.decode(
                    outputs[0][inputs["input_ids"].size(1):], skip_special_tokens=True,
                )
                parsed = self._parse_rules(response)
                if parsed:
                    return parsed
            except Exception as e:
                print(f"[LLM Local] Attempt {attempt + 1} error: {e}")
        return []

    @staticmethod
    def _parse_rules(text: str) -> List[Dict]:
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end > start:
                json_str = text[start: end + 1]
                rules = json.loads(json_str)
                if isinstance(rules, list) and all(
                    isinstance(r, dict) and "rule" in r and "confidence" in r
                    for r in rules
                ):
                    return rules
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        return []

    @staticmethod
    def _fallback_rules(transaction_summary: str) -> List[Dict]:
        return list(DEFAULT_RULES)

    def get_fallback_count(self) -> int:
        return self._fallback_count

"""
规则编码器 —— 对应大模型.md Step 2。

将规则转为数值信号（rule_bias）：
方法1（默认）：MLP 从 edge_attr 计算基础 rule_bias
方法2（可选）：LLM 规则调制

规则匹配修复（V2）：
- edge_attr 中 amount_norm 是 z-score 归一化的，不能直接和绝对阈值比较
- 改用 edge_attr 中的 log_amount (=log1p(raw_amount)) 还原近似原始金额
- LLM 规则的阈值条件使用还原后的原始金额进行比较
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class RuleEncoder(nn.Module):
    """规则编码器。"""

    def __init__(self, edge_attr_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.edge_attr_dim = edge_attr_dim

        self.rule_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self._last_trace: Optional[Dict[str, torch.Tensor | List[Dict]]] = None

    def forward(
        self,
        edge_attr: torch.Tensor,
        llm_rules: Optional[List[Dict]] = None,
    ) -> Dict[str, torch.Tensor | List[Dict] | List[str] | int]:
        """
        Args:
            edge_attr: [num_edges, edge_attr_dim]
                dim 0: amount_norm (z-score)
                dim 1: log_amount (= log1p(raw_amount))
                dim 2+: other features
            llm_rules: LLM 生成的规则列表（可选）

        Returns:
            rule_bias: [num_edges, 1] 规则偏置
        """
        base_bias = self.rule_mlp(edge_attr).view(-1)
        rule_bias = base_bias.clone()
        rule_match_mask = torch.zeros(edge_attr.size(0), dtype=torch.bool, device=edge_attr.device)
        rule_match_score = torch.zeros(edge_attr.size(0), dtype=edge_attr.dtype, device=edge_attr.device)
        rule_trace_idx = torch.full((edge_attr.size(0),), -1, dtype=torch.long, device=edge_attr.device)

        if llm_rules is not None and len(llm_rules) > 0:
            rule_bias, rule_match_mask, rule_match_score, rule_trace_idx = self._apply_llm_rules(
                rule_bias, edge_attr, llm_rules
            )

        result = {
            "rule_bias": rule_bias,
            "rule_match_mask": rule_match_mask,
            "rule_match_score": rule_match_score,
            "rule_trace_idx": rule_trace_idx,
            "matched_rule_confidence": torch.zeros(edge_attr.size(0), dtype=edge_attr.dtype, device=edge_attr.device),
            "matched_rule_text": [""] * edge_attr.size(0),
            "matched_rule_type": ["none"] * edge_attr.size(0),
            "rules": llm_rules or [],
            "available_rule_count": len(llm_rules or []),
        }
        self._last_trace = result
        if llm_rules is not None and len(llm_rules) > 0:
            matched_rule_confidence = result["matched_rule_confidence"]
            matched_rule_text = result["matched_rule_text"]
            matched_rule_type = result["matched_rule_type"]
            for edge_idx, rule_idx in enumerate(rule_trace_idx.tolist()):
                if rule_idx < 0 or rule_idx >= len(llm_rules):
                    continue
                rule = llm_rules[rule_idx]
                matched_rule_confidence[edge_idx] = float(rule.get("confidence", 0.0))
                matched_rule_text[edge_idx] = str(rule.get("rule", ""))
                matched_rule_type[edge_idx] = str(rule.get("rule_type", "generic"))
        return result

    @staticmethod
    def _apply_llm_rules(
        rule_bias: torch.Tensor,
        edge_attr: torch.Tensor,
        llm_rules: List[Dict],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        用 LLM 规则对 rule_bias 进行调制。

        edge_attr[..., 0] = amount_norm (z-score)
        edge_attr[..., 1] = log_amount  (= log1p(raw_amount))

        使用 log_amount 还原原始金额用于阈值比较。
        """
        # 从 log_amount 还原原始金额
        log_amount = edge_attr[..., 1]
        raw_amounts = torch.expm1(log_amount)  # expm1(x) = exp(x) - 1

        rule_match_mask = torch.zeros(edge_attr.size(0), dtype=torch.bool, device=edge_attr.device)
        rule_match_score = torch.zeros(edge_attr.size(0), dtype=edge_attr.dtype, device=edge_attr.device)
        rule_trace_idx = torch.full((edge_attr.size(0),), -1, dtype=torch.long, device=edge_attr.device)

        for rule_idx, rule in enumerate(llm_rules):
            confidence = rule.get("confidence", 0.5)
            condition = str(rule.get("rule", ""))
            thresholds = rule.get("thresholds", {}) or {}
            rule_type = str(rule.get("rule_type", "generic"))
            current_mask = torch.zeros(edge_attr.size(0), dtype=torch.bool, device=edge_attr.device)
            current_score = torch.zeros(edge_attr.size(0), dtype=edge_attr.dtype, device=edge_attr.device)

            amount_threshold = float(thresholds.get("amount_gt", 0.0) or 0.0)
            if amount_threshold > 0:
                current_mask |= raw_amounts > amount_threshold
                current_score = torch.maximum(
                    current_score,
                    torch.clamp(raw_amounts / max(amount_threshold, 1.0), min=0.0, max=2.0),
                )

            if thresholds.get("near_threshold"):
                threshold = amount_threshold if amount_threshold > 0 else 10000.0
                near_mask = (raw_amounts > threshold * 0.8) & (raw_amounts < threshold)
                current_mask |= near_mask
                current_score = torch.where(near_mask, torch.full_like(current_score, 0.5), current_score)

            if rule_type in {"high_frequency", "rapid_transfer"}:
                current_mask |= raw_amounts > 0
                current_score = torch.where(raw_amounts > 0, torch.full_like(current_score, 0.3), current_score)

            if rule_type in {"cycle", "layering"}:
                current_mask |= raw_amounts > 0
                current_score = torch.where(raw_amounts > 0, torch.full_like(current_score, 0.4), current_score)

            if not current_mask.any() and "amount >" in condition:
                for threshold in (10000.0, 20000.0, 50000.0, 100000.0):
                    if f"{int(threshold)}" in condition:
                        current_mask |= raw_amounts > threshold
                        current_score = torch.maximum(
                            current_score,
                            torch.clamp(raw_amounts / threshold, min=0.0, max=2.0),
                        )

            if current_mask.any():
                boost = 1.0 + confidence * torch.clamp(current_score[current_mask], min=0.0, max=2.0)
                rule_bias[current_mask] = rule_bias[current_mask] * boost
                better_mask = current_mask & (current_score >= rule_match_score)
                rule_match_mask |= current_mask
                rule_match_score[better_mask] = current_score[better_mask]
                rule_trace_idx[better_mask] = rule_idx

        return rule_bias, rule_match_mask, rule_match_score, rule_trace_idx

    def get_last_trace(self) -> Optional[Dict[str, torch.Tensor | List[Dict]]]:
        return self._last_trace

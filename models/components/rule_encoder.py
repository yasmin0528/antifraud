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

    def forward(
        self,
        edge_attr: torch.Tensor,
        llm_rules: Optional[List[Dict]] = None,
    ) -> torch.Tensor:
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
        rule_bias = self.rule_mlp(edge_attr)

        if llm_rules is not None and len(llm_rules) > 0:
            rule_bias = self._apply_llm_rules(rule_bias, edge_attr, llm_rules)

        return rule_bias

    @staticmethod
    def _apply_llm_rules(
        rule_bias: torch.Tensor,
        edge_attr: torch.Tensor,
        llm_rules: List[Dict],
    ) -> torch.Tensor:
        """
        用 LLM 规则对 rule_bias 进行调制。

        edge_attr[..., 0] = amount_norm (z-score)
        edge_attr[..., 1] = log_amount  (= log1p(raw_amount))

        使用 log_amount 还原原始金额用于阈值比较。
        """
        # 从 log_amount 还原原始金额
        log_amount = edge_attr[..., 1]
        raw_amounts = torch.expm1(log_amount)  # expm1(x) = exp(x) - 1

        for rule in llm_rules:
            confidence = rule.get("confidence", 0.5)
            condition = rule.get("rule", "")

            # 金额类规则 —— 使用还原后的原始金额
            amount_thresholds = [
                ("> 100000", 100000),
                ("> 50000", 50000),
                ("> 20000", 20000),
                ("> 10000", 10000),
            ]
            for keyword, threshold in amount_thresholds:
                if keyword in condition:
                    mask = raw_amounts > threshold
                    if mask.any():
                        boost = 1.0 + confidence * min(
                            raw_amounts[mask].mean().item() / threshold, 2.0
                        )
                        rule_bias[mask] = rule_bias[mask] * max(boost, 1.0)

            # 频率类规则（简化：检测到高金额频次模式）
            if "freq >" in condition and "AND" in condition:
                for keyword, _ in amount_thresholds:
                    if keyword in condition:
                        mask = raw_amounts > 0
                        rule_bias[mask] = rule_bias[mask] * (1.0 + confidence * 0.3)

            # 大额单笔交易规则
            if "single_transaction" in condition or "amount >" in condition:
                for keyword, threshold in amount_thresholds:
                    if keyword in condition:
                        mask = raw_amounts > threshold
                        if mask.any():
                            boost = 1.0 + confidence * min(
                                raw_amounts[mask].mean().item() / threshold, 2.0
                            )
                            rule_bias[mask] = rule_bias[mask] * max(boost, 1.0)

            # 近阈值交易模式（structuring detection）
            if "near_threshold" in condition:
                for keyword, threshold in amount_thresholds:
                    if keyword in condition:
                        # 接近但不超过阈值（阈值的 80%-100%）
                        mask = (raw_amounts > threshold * 0.8) & (raw_amounts < threshold)
                        if mask.any():
                            rule_bias[mask] = rule_bias[mask] * (1.0 + confidence * 0.5)

        return rule_bias

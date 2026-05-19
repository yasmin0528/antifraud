"""
规则编码器 —— 对应大模型.md Step 2。

将规则转为数值信号（rule_bias）：
方法1（默认）：MLP 从 edge_attr 计算基础 rule_bias
方法2（可选）：LLM 规则调制
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
            edge_attr: [num_edges, edge_attr_dim] 边特征
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
        """用 LLM 规则对 rule_bias 进行调制。"""
        amounts = edge_attr[..., 0]

        for rule in llm_rules:
            confidence = rule.get("confidence", 0.5)
            condition = rule.get("rule", "")

            # 金额类规则
            amount_thresholds = [
                ("> 100000", 100000),
                ("> 50000", 50000),
                ("> 20000", 20000),
                ("> 10000", 10000),
            ]
            for keyword, threshold in amount_thresholds:
                if keyword in condition:
                    mask = amounts > threshold
                    if mask.any():
                        boost = 1.0 + confidence * min(
                            amounts[mask].mean().item() / threshold, 2.0
                        )
                        rule_bias[mask] = rule_bias[mask] * max(boost, 1.0)

            # 频率类规则
            if "freq >" in condition and "AND" in condition:
                for keyword, _ in amount_thresholds:
                    if keyword in condition:
                        mask = amounts > 0
                        rule_bias[mask] = rule_bias[mask] * (1.0 + confidence * 0.3)

        return rule_bias

"""
Rule-guided GAT 层 —— 对应大模型.md Step 3 & 4。

注意力计算：
    attention = softmax(Q @ K^T + rule_bias)

rule_bias 作为自上而下的控制信号，调控信息传播路径。
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class RuleGuidedGATLayer(nn.Module):
    """
    Rule-guided GAT 层。

    将 rule_bias 注入注意力计算，实现 top-down 控制。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int = 4,
        dropout: float = 0.2,
        negative_slope: float = 0.2,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads if out_dim % num_heads == 0 else out_dim
        self.out_dim = self.head_dim * num_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(in_dim, self.head_dim * num_heads)
        self.k_proj = nn.Linear(in_dim, self.head_dim * num_heads)
        self.v_proj = nn.Linear(in_dim, self.head_dim * num_heads)

        # 输出投影
        self.out_proj = nn.Linear(self.out_dim, out_dim)

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self._last_edge_trace: Optional[dict[str, torch.Tensor]] = None
        self._init_weights()

    def _init_weights(self):
        for proj in [self.q_proj, self.k_proj, self.v_proj]:
            nn.init.xavier_uniform_(proj.weight, gain=1.0)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        rule_bias: Optional[torch.Tensor] = None,
    ) -> (torch.Tensor, Optional[torch.Tensor]):
        """
        Args:
            x: [num_nodes, in_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            rule_bias: [num_edges, 1] 规则偏置

        Returns:
            out: [num_nodes, out_dim] 更新后的节点特征
            attn_weights: [num_edges, num_heads] 注意力权重（用于可视化）
        """
        num_nodes = x.size(0)
        H, D = self.num_heads, self.head_dim

        Q = self.q_proj(x).view(num_nodes, H, D)
        K = self.k_proj(x).view(num_nodes, H, D)
        V = self.v_proj(x).view(num_nodes, H, D)

        src, dst = edge_index

        Q_dst = Q[dst]
        K_src = K[src]
        V_src = V[src]

        # 注意力分数 = Q*K^T / sqrt(D)
        attn = (Q_dst * K_src).sum(dim=-1) / math.sqrt(D)  # [E, H]

        # Rule bias 注入（核心）
        if rule_bias is not None:
            attn = attn + rule_bias.view(-1, 1)

        # Per-destination softmax
        attn = self._edge_softmax(attn, dst, num_nodes)
        attn_weights = attn.detach().clone()
        self._last_attn = attn_weights
        attn = self.dropout(attn)

        # 消息聚合
        messages = attn.unsqueeze(-1) * V_src  # [E, H, D]
        out = torch.zeros(num_nodes, H * D, device=x.device)
        out.index_add_(0, dst, messages.reshape(-1, H * D))

        out = self.out_proj(out)
        self._last_edge_trace = {
            "attention": attn_weights.detach().clone(),
            "rule_bias": rule_bias.detach().clone() if rule_bias is not None else torch.zeros(attn_weights.size(0), device=attn_weights.device),
        }
        return out, attn_weights

    def set_last_edge_trace(
        self,
        attention: torch.Tensor,
        rule_bias: Optional[torch.Tensor] = None,
        rule_match_score: Optional[torch.Tensor] = None,
        rule_trace_idx: Optional[torch.Tensor] = None,
    ):
        self._last_edge_trace = {
            "attention": attention.detach().clone(),
            "rule_bias": rule_bias.detach().clone() if rule_bias is not None else torch.zeros(attention.size(0), device=attention.device),
            "rule_match_score": rule_match_score.detach().clone() if rule_match_score is not None else torch.zeros(attention.size(0), device=attention.device),
            "rule_trace_idx": rule_trace_idx.detach().clone() if rule_trace_idx is not None else torch.full((attention.size(0),), -1, dtype=torch.long, device=attention.device),
        }

    def get_last_edge_trace(self) -> Optional[dict[str, torch.Tensor]]:
        return self._last_edge_trace

    @staticmethod
    def _edge_softmax(
        attn: torch.Tensor, dst: torch.Tensor, num_nodes: int
    ) -> torch.Tensor:
        """按目标节点进行 softmax 归一化。"""
        attn_max = torch.zeros(num_nodes, attn.size(1), device=attn.device)
        attn_max.index_reduce_(0, dst, attn, "amax", include_self=False)
        attn_max = attn_max[dst]

        attn_exp = torch.exp(attn - attn_max)
        attn_sum = torch.zeros(num_nodes, attn.size(1), device=attn.device)
        attn_sum.index_add_(0, dst, attn_exp)
        attn_sum = attn_sum[dst] + 1e-8

        return attn_exp / attn_sum

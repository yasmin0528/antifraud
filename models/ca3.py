"""
CA3_AGM: 团伙记忆与模式补全模块（模拟CA3联想记忆）。

核心改进（v2）：
1. 多记忆槽（num_groups ≥ 16）：每个槽位代表一种交易模式原型
2. 可学习软分配：通过 MLP 将节点嵌入软分配到不同记忆槽
3. 门控融合：可学习的 sigmoid 门控替代简单残差加法
4. 可学习记忆：记忆槽通过梯度下降优化（不再手动动量更新）
5. 可选 RPE 调制：多巴胺 RPE 信号调节记忆写入强度
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3_AGM(nn.Module):
    """团伙记忆与模式补全：模拟CA3联想记忆（v2 重设计版）。"""

    def __init__(
        self,
        emb_dim: int = 128,
        num_groups: int = 16,
        rpe_dim: int = 1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_groups = num_groups

        # ---- 可学习的记忆槽 ----
        self.memory_bank = nn.Parameter(torch.randn(num_groups, emb_dim) * 0.1)

        # ---- 软分配网络：h → 各记忆槽的权重 ----
        self.group_assigner = nn.Sequential(
            nn.Linear(emb_dim + rpe_dim, num_groups),
            nn.Tanh(),
        )

        # ---- 门控网络：决定从记忆中读取多少信息 ----
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim + rpe_dim, emb_dim),
            nn.Sigmoid(),
        )

        # ---- 记忆读取后的变换 ----
        self.memory_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.xavier_uniform_(self.memory_proj.weight)

    def forward(
        self,
        h: torch.Tensor,
        group_ids: torch.Tensor = None,
        da_signal: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [num_nodes, emb_dim] 节点嵌入
            group_ids: [num_nodes] 团伙ID（未使用，保留兼容接口）
            da_signal: float 或标量tensor, 全局多巴胺RPE信号（可选, 调节记忆强度）

        Returns:
            h_enhanced: [num_nodes, emb_dim] 增强后的节点嵌入
        """
        # ---- 构造调制信号 ----
        # da_signal: 高 RPE → 更关注记忆读取（加大记忆影响）
        # 使用全局标量（生物VTA多巴胺是全局调质，非per-neuron）
        if da_signal is None:
            da_signal = 0.0
        if isinstance(da_signal, (int, float)):
            da_signal = torch.full((h.size(0), 1), da_signal, device=h.device, dtype=h.dtype)
        else:
            da_signal = da_signal.view(-1, 1)

        # 将 RPE 信号拼接到嵌入上用于分配和门控
        h_rpe = torch.cat([h, da_signal], dim=-1)  # [N, emb_dim + 1]

        # ---- 软分配：每个节点在各记忆槽上的分布 ----
        assign_logits = self.group_assigner(h_rpe)   # [N, num_groups]
        assign_weights = F.softmax(assign_logits, dim=-1)  # [N, num_groups]

        # ---- 加权读取记忆 ----
        # memory_out_i = sum_j assign_ij * memory_j
        memory_out = assign_weights @ self.memory_bank  # [N, emb_dim]
        memory_out = self.memory_proj(memory_out)        # [N, emb_dim]

        # ---- 可学习门控融合 ----
        gate = self.gate_net(h_rpe)  # [N, emb_dim], 值在 (0, 1)
        h_enhanced = h + gate * memory_out

        return h_enhanced

    def get_memory_patterns(self) -> torch.Tensor:
        """返回当前的记忆槽（用于可视化和分析）。"""
        return self.memory_bank.data.clone()

    def get_assignment_weights(self, h: torch.Tensor) -> torch.Tensor:
        """返回节点到记忆槽的软分配权重（用于可视化）。"""
        h_rpe = torch.cat([h, torch.zeros(h.size(0), 1, device=h.device)], dim=-1)
        assign_logits = self.group_assigner(h_rpe)
        return F.softmax(assign_logits, dim=-1)

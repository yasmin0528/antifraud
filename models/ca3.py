"""
CA3_AGM: 团伙记忆与模式补全模块（模拟CA3联想记忆）。

维护一个团伙记忆库（memory_bank），
通过注意力机制将当前节点嵌入与团伙记忆融合。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3_AGM(nn.Module):
    """团伙记忆与模式补全：模拟CA3联想记忆。"""

    def __init__(
        self,
        emb_dim: int = 128,
        num_groups: int = 1,
        memory_momentum: float = 0.9,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_groups = num_groups
        self.memory_momentum = memory_momentum

        # 团伙记忆库
        self.register_buffer("memory_bank", torch.randn(num_groups, emb_dim) * 0.1)

    def forward(
        self,
        h: torch.Tensor,
        group_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            h: [num_nodes, emb_dim] 节点嵌入
            group_ids: [num_nodes] 团伙ID（可选，-1表示无团伙）

        Returns:
            h_enhanced: [num_nodes, emb_dim] 增强后的节点嵌入
        """
        if group_ids is not None:
            valid_mask = group_ids >= 0
            if valid_mask.any():
                self._update_memory(h[valid_mask], group_ids[valid_mask])

        # 注意力读取记忆
        attn = F.softmax(h @ self.memory_bank.T, dim=-1)
        memory_out = attn @ self.memory_bank
        h_enhanced = h + memory_out
        return h_enhanced

    def _update_memory(self, h: torch.Tensor, group_ids: torch.Tensor):
        """动量更新记忆库。"""
        group_sums = torch.zeros_like(self.memory_bank)
        group_counts = torch.zeros(self.num_groups, device=h.device)
        group_sums.index_add_(0, group_ids, h)
        group_counts.index_add_(
            0, group_ids,
            torch.ones(group_ids.size(0), device=h.device),
        )
        group_has_data = group_counts > 0
        group_mean = group_sums / group_counts.unsqueeze(-1).clamp(min=1.0)

        with torch.no_grad():
            updated = self.memory_bank.clone()
            updated[group_has_data] = (
                self.memory_bank[group_has_data] * self.memory_momentum
                + group_mean[group_has_data] * (1 - self.memory_momentum)
            )
            self.memory_bank.copy_(updated)

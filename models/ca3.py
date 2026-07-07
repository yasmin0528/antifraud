"""
CA3_AGM: explicit group-memory module for suspicious pattern retrieval.

The module maintains one prototype per explicit group id and uses supervised
prototype retrieval instead of free-form memory slots.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CA3_AGM(nn.Module):
    """Explicit group memory with learnable retrieval and gated fusion."""

    def __init__(
        self,
        emb_dim: int = 128,
        num_groups: int = 16,
        rpe_dim: int = 1,
        memory_momentum: float = 0.9,
        temperature: float = 0.2,
        memory_mode: str = "explicit_group",
        update_mode: str = "ema_group_proto",
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.num_groups = max(int(num_groups), 1)
        self.rpe_dim = rpe_dim
        self.memory_momentum = memory_momentum
        self.temperature = temperature
        self.memory_mode = memory_mode
        self.update_mode = update_mode

        self.memory_bank = nn.Parameter(torch.randn(self.num_groups, emb_dim) * 0.02)
        self.background_memory = nn.Parameter(torch.zeros(1, emb_dim))

        self.query_proj = nn.Linear(emb_dim + rpe_dim, emb_dim)
        self.gate_net = nn.Sequential(
            nn.Linear(emb_dim + rpe_dim, emb_dim),
            nn.Sigmoid(),
        )
        self.memory_proj = nn.Linear(emb_dim, emb_dim)
        nn.init.xavier_uniform_(self.query_proj.weight)
        nn.init.xavier_uniform_(self.memory_proj.weight)

        self._last_scores: Optional[torch.Tensor] = None
        self._last_group_ids: Optional[torch.Tensor] = None
        self._group_meta: Dict[int, Dict[str, object]] = {}

    def _prepare_da(self, h: torch.Tensor, da_signal: Optional[torch.Tensor | float]) -> torch.Tensor:
        if da_signal is None:
            return torch.zeros((h.size(0), 1), device=h.device, dtype=h.dtype)
        if isinstance(da_signal, (int, float)):
            return torch.full((h.size(0), 1), float(da_signal), device=h.device, dtype=h.dtype)
        da_signal = da_signal.to(device=h.device, dtype=h.dtype)
        if da_signal.dim() == 0:
            return da_signal.view(1, 1).expand(h.size(0), 1)
        if da_signal.dim() == 1:
            if da_signal.numel() == 1:
                return da_signal.view(1, 1).expand(h.size(0), 1)
            if da_signal.size(0) == h.size(0):
                return da_signal.view(-1, 1)
            return da_signal.mean().view(1, 1).expand(h.size(0), 1)
        if da_signal.numel() == 1:
            return da_signal.view(1, 1).expand(h.size(0), 1)
        if da_signal.size(0) == h.size(0):
            return da_signal.reshape(h.size(0), 1)
        return da_signal.mean().view(1, 1).expand(h.size(0), 1)

    def forward(
        self,
        h: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        da_signal: Optional[torch.Tensor | float] = None,
        update_memory: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            h: [N, D]
            group_ids: [N], explicit group ids, -1 means background/unknown.
            da_signal: global or per-sample neuromodulation signal.
            update_memory: whether to update group prototypes with current batch.
        """
        da = self._prepare_da(h, da_signal)
        h_rpe = torch.cat([h, da], dim=-1)
        query = F.normalize(self.query_proj(h_rpe), dim=-1)

        memory_norm = F.normalize(self.memory_bank, dim=-1)
        scores = query @ memory_norm.t()
        assign_weights = F.softmax(scores / self.temperature, dim=-1)
        retrieved = assign_weights @ self.memory_bank

        if group_ids is not None:
            group_ids = group_ids.to(h.device).long()
            valid_mask = (group_ids >= 0) & (group_ids < self.num_groups)
            if valid_mask.any():
                explicit_memory = self.memory_bank[group_ids[valid_mask]]
                retrieved = retrieved.clone()
                retrieved[valid_mask] = explicit_memory
            if (~valid_mask).any():
                retrieved = retrieved.clone()
                retrieved[~valid_mask] = self.background_memory.expand((~valid_mask).sum(), -1)
            self._last_group_ids = group_ids.detach().cpu()
        else:
            self._last_group_ids = None

        gate = self.gate_net(h_rpe)
        memory_out = self.memory_proj(retrieved)
        h_enhanced = h + gate * memory_out

        self._last_scores = scores.detach().cpu()

        if update_memory and group_ids is not None:
            self.update_group_memory(h.detach(), group_ids)

        return h_enhanced

    @torch.no_grad()
    def update_group_memory(self, h: torch.Tensor, group_ids: torch.Tensor):
        """Momentum-update explicit group prototypes from current batch."""
        if group_ids is None:
            return
        group_ids = group_ids.to(h.device).long()
        valid_mask = (group_ids >= 0) & (group_ids < self.num_groups)
        if not valid_mask.any():
            return

        valid_h = h[valid_mask]
        valid_groups = group_ids[valid_mask]
        unique_groups = valid_groups.unique()

        for gid in unique_groups.tolist():
            member_mask = valid_groups == gid
            proto = valid_h[member_mask].mean(dim=0)
            if self.update_mode == "batch_mean":
                self.memory_bank.data[gid] = proto
            else:
                self.memory_bank.data[gid] = (
                    self.memory_momentum * self.memory_bank.data[gid]
                    + (1.0 - self.memory_momentum) * proto
                )
            self._group_meta[int(gid)] = {
                "group_id": int(gid),
                "update_mode": self.update_mode,
                "member_count": int(member_mask.sum().item()),
                "memory_mode": self.memory_mode,
            }

    @torch.no_grad()
    def update_group_memory_from_aggregates(self, group_repr: torch.Tensor, group_ids: torch.Tensor):
        if group_repr is None or group_ids is None or group_repr.numel() == 0:
            return
        self.update_group_memory(group_repr, group_ids)

    def memory_loss(self, h: torch.Tensor, group_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """Supervised prototype classification loss over valid explicit groups."""
        if group_ids is None:
            return torch.zeros((), device=h.device, dtype=h.dtype)

        group_ids = group_ids.to(h.device).long()
        valid_mask = (group_ids >= 0) & (group_ids < self.num_groups)
        if not valid_mask.any():
            return torch.zeros((), device=h.device, dtype=h.dtype)

        query = F.normalize(h[valid_mask], dim=-1)
        memory_norm = F.normalize(self.memory_bank, dim=-1)
        logits = query @ memory_norm.t()
        return F.cross_entropy(logits / self.temperature, group_ids[valid_mask])

    def export_memory_state(self) -> Dict[str, torch.Tensor | int]:
        return {
            "num_groups": self.num_groups,
            "memory_mode": self.memory_mode,
            "update_mode": self.update_mode,
            "memory_bank": self.memory_bank.detach().cpu(),
            "background_memory": self.background_memory.detach().cpu(),
            "last_scores": self._last_scores,
            "last_group_ids": self._last_group_ids,
        }

    def export_group_meta(self) -> Dict[int, Dict[str, object]]:
        return dict(self._group_meta)

    def get_memory_patterns(self) -> torch.Tensor:
        return self.memory_bank.detach().cpu().clone()

    def get_assignment_weights(self, h: torch.Tensor) -> torch.Tensor:
        query = F.normalize(h, dim=-1)
        memory_norm = F.normalize(self.memory_bank, dim=-1)
        logits = query @ memory_norm.t()
        return F.softmax(logits / self.temperature, dim=-1)

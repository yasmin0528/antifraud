"""
模型基类 —— 所有 AML 模型的统一接口。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class BaseAMLModel(nn.Module):
    """
    所有 AML 检测模型的抽象基类。

    Subclasses must implement:
        forward(self, batch) -> Dict[str, torch.Tensor]
    """

    def __init__(self):
        super().__init__()

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        前向传播。

        Args:
            batch: 包含以下 key 的字典：
                - sequences: [batch, seq_len, feat_dim]
                - sender_idx: [batch]
                - receiver_idx: [batch]
                - edge_attr: [batch, edge_dim]
                - alert_idx: [batch]
                - labels: [batch]
                - (optional) transaction_summary: str

        Returns:
            dict with keys:
                - logit: [batch, 1]
                - prob: [batch, 1]
                - node_emb: [num_nodes, hidden_dim]  (optional)
                - attention_weights:  (optional)
        """
        raise NotImplementedError

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重（用于可视化）。"""
        return None

    def get_node_embeddings(self) -> Optional[torch.Tensor]:
        """获取节点嵌入（用于 t-SNE）。"""
        return None

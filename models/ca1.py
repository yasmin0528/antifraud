"""
CA1_TTPM: 时间序列建模模块（模拟海马体CA1）。

对应方案中的时间编码 + 微观异常评分。
输入：交易序列 [batch_size, seq_len, feature_dim]
输出：embedding h + 异常分数 score_micro
"""

import torch
import torch.nn as nn


class CA1_TTPM(nn.Module):
    """时间序列建模模块：模拟海马体CA1。"""

    def __init__(
        self,
        feature_dim: int = 3,
        hidden_dim: int = 128,
        n_types: int = 0,
        type_emb_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_type_embedding = n_types > 0
        self.type_emb_dim = type_emb_dim if self.use_type_embedding else 1
        self.n_types = n_types

        if self.use_type_embedding:
            self.type_emb = nn.Embedding(n_types, type_emb_dim)
            lstm_input_dim = 1 + type_emb_dim + 1
        else:
            lstm_input_dim = feature_dim

        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc_score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            x: [batch_size, seq_len, feature_dim] 交易序列

        Returns:
            h: [batch_size, hidden_dim] 序列嵌入
            score_micro: [batch_size, 1] 微观异常分数
        """
        if self.use_type_embedding:
            amount = x[..., 0:1]
            type_id = x[..., 1].long().clamp(min=0)
            time_diff = x[..., 2:3]
            type_emb = self.type_emb(type_id)
            x_in = torch.cat([amount, type_emb, time_diff], dim=-1)
        else:
            x_in = x

        lstm_out, _ = self.lstm(x_in)
        h = lstm_out[:, -1, :]
        h = self.dropout(h)
        score_micro = torch.sigmoid(self.fc_score(h))
        return h, score_micro

"""
MPFC (Medial Prefrontal Cortex) 模块 —— 规则驱动 + 图推理 + 任务控制。

对应 大模型.md Step 1 ~ 7。

LLM 是 mPFC 模块的内置设计，模拟前额叶皮层的"符号推理能力"：
- LLaMA3 → 符号规则生成（前额叶推理）
- RuleEncoder → 自上而下调控（top-down control）
- RuleGuidedGAT → 关系推理 + 规则调控
- Task Gating → 目标驱动注意（goal selection）
- Pooling → 抽象认知（group reasoning）

use_llm=False 模式用于消融实验，测试去除 LLM 符号推理后的效果。
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .components.rule_encoder import RuleEncoder
from .components.rule_guided_gat import RuleGuidedGATLayer


class MPFC(nn.Module):
    """
    完整 MPFC 模块 —— LLM 为 mPFC 天然组成部分。

    LLM 模拟前额叶的符号推理与规则生成能力，通过以下流程完成检测：
    Step 1: LLM 规则生成（前额叶推理）
    Step 2: 规则编码（自上而下调控）
    Step 3-4: 多层 Rule-guided GNN（关系推理 + 规则调控）
    Step 5: Task-guided Gating（目标驱动注意）
    Step 6: 子图聚合
    Step 7: 输出层

    当 use_llm=False 时，跳过 LLM 规则生成步骤，使用边特征的 MLP 规则偏置作为替代。
    """

    def __init__(
        self,
        emb_dim: int = 128,
        edge_attr_dim: int = 2,
        hidden_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        llm_config: Optional[Dict] = None,
        use_llm: bool = True,
    ):
        super().__init__()

        self.input_dim = emb_dim + 1  # CA3 emb_dim + CA1 score_micro
        self.edge_attr_dim = edge_attr_dim
        self.hidden_dim = hidden_dim
        self.num_gnn_layers = num_gnn_layers
        self.use_llm = use_llm

        # ---- Step 1: 规则生成 ----
        # use_llm=True: LLM 生成符号规则（前额叶推理）
        # use_llm=False: 用 MLP 从边特征生成规则偏置（消融实验用）
        self.rule_mlp = nn.Sequential(
            nn.Linear(edge_attr_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- Step 2: 规则编码 ----
        self.rule_encoder = RuleEncoder(edge_attr_dim, hidden_dim)

        # ---- Step 3 & 4: Multi-layer Rule-guided GNN ----
        self.gnn_layers = nn.ModuleList()
        self.residual_proj = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for i in range(num_gnn_layers):
            in_dim = self.input_dim if i == 0 else hidden_dim
            out_dim = hidden_dim

            self.gnn_layers.append(
                RuleGuidedGATLayer(in_dim, out_dim, num_heads, dropout)
            )

            if in_dim != out_dim:
                self.residual_proj.append(nn.Linear(in_dim, out_dim))
            else:
                self.residual_proj.append(nn.Identity())

            self.layer_norms.append(nn.LayerNorm(out_dim))

        # ---- Step 5: Task-guided Gating ----
        self.task_vector = nn.Parameter(torch.randn(hidden_dim))
        self.task_gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        nn.init.normal_(self.task_vector, mean=0.0, std=0.02)

        # ---- Step 7: Output Layer ----
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = nn.Dropout(dropout)

        # ---- LLM 配置（mPFC 内置组件） ----
        llm_config = llm_config or {}
        self.llm_model_name = llm_config.get("model_name", "meta-llama/Llama-3.2-8B-Instruct")
        self.llm_use_api = llm_config.get("use_api", True)
        self.llm_api_url = llm_config.get("api_url", None)
        self.llm_api_key = llm_config.get("api_key", None)
        self.rule_update_frequency = llm_config.get("rule_update_frequency", 100)
        self.max_rules = llm_config.get("max_rules", 20)

        self._rule_update_counter = 0
        self._current_rules: Optional[List[Dict]] = None
        self._llm_interface = None

    # ----------------------------------------------------------
    # LLM 接口懒加载（仅在 use_llm=True 时使用）
    # ----------------------------------------------------------
    @property
    def llm_interface(self):
        if not self.use_llm:
            return None
        if self._llm_interface is None:
            from .components.llm_interface import LLMInterface

            self._llm_interface = LLMInterface(
                model_name=self.llm_model_name,
                use_api=self.llm_use_api,
                api_url=self.llm_api_url,
                api_key=self.llm_api_key,
                device="cpu",
            )
        return self._llm_interface

    # ----------------------------------------------------------
    # Step 1: 规则生成（带频率控制和缓存）
    # ----------------------------------------------------------
    def update_rules(self, transaction_summary: Optional[str] = None):
        """按频率更新 LLM 生成的规则。"""
        if not self.use_llm or transaction_summary is None:
            return

        self._rule_update_counter += 1

        if self._rule_update_counter == 1 or (
            self._rule_update_counter % self.rule_update_frequency == 0
        ):
            rules = self.llm_interface.generate_rules(transaction_summary)
            if len(rules) > self.max_rules:
                rules = sorted(
                    rules, key=lambda r: r.get("confidence", 0), reverse=True
                )[: self.max_rules]
            self._current_rules = rules

            if rules:
                print(
                    f"[MPFC] LLM rules updated (step {self._rule_update_counter}): "
                    f"{len(rules)} rules, "
                    f"fallback_count={self.llm_interface.get_fallback_count()}"
                )
                self._save_rules_to_file()

    def _save_rules_to_file(self, filepath: Optional[str] = None):
        """将当前规则保存到 JSON 文件。"""
        if not self._current_rules:
            return
        if filepath is None:
            filepath = "llm_rules.json"
        # 如果在 base_trainer 中运行，保存到输出目录
        output_dir = getattr(self, '_output_dir', None)
        if output_dir:
            filepath = os.path.join(output_dir, "llm_rules.json")
        else:
            filepath = "llm_rules.json"
        import json
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._current_rules, f, ensure_ascii=False, indent=2)
        print(f"[MPFC] Rules saved to {filepath}")

    def set_output_dir(self, output_dir: str):
        """设置输出目录，用于保存规则文件。"""
        self._output_dir = output_dir

    # ----------------------------------------------------------
    # 辅助：全局平均池化
    # ----------------------------------------------------------
    @staticmethod
    def _global_mean_pool(x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """子图级别池化（对应 Step 6）。"""
        num_graphs = int(batch.max().item()) + 1
        pooled = torch.zeros(num_graphs, x.size(-1), device=x.device)
        counts = torch.zeros(num_graphs, 1, device=x.device)
        pooled.index_add_(0, batch, x)
        counts.index_add_(
            0, batch, torch.ones_like(batch, dtype=torch.float).unsqueeze(-1)
        )
        return pooled / counts.clamp(min=1.0)

    # ----------------------------------------------------------
    # 完整前向传播（Step 1 ~ 7）
    # ----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        transaction_summary: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        完整前向传播（大模型.md Step 1 ~ 7）。

        Args:
            x: [num_nodes, input_dim] 节点特征
            edge_index: [2, num_edges] 边索引
            edge_attr: [num_edges, edge_attr_dim] 边特征
            batch: [num_nodes] 子图批次索引
            transaction_summary: 交易模式描述（用于 Step 1 LLM 规则生成）

        Returns:
            x: [num_nodes, hidden_dim] 更新后的节点 embedding
            logit: [num_nodes/graphs, 1] 未归一化 logit
            prob: [num_nodes/graphs, 1] 风险概率
        """
        if edge_attr is None:
            edge_attr = torch.zeros(
                edge_index.size(1), self.edge_attr_dim, device=x.device
            )

        # ---- Step 1: 规则生成 ----
        if self.use_llm:
            # LLM 符号规则生成（前额叶推理）
            self.update_rules(transaction_summary)
            rule_bias = self.rule_encoder(edge_attr, self._current_rules)
        else:
            # MLP 规则偏置（消融实验：去除 LLM 后的效果）
            rule_bias = self.rule_mlp(edge_attr).squeeze(-1)  # [E]

        # ---- Step 3 & 4: Layer-wise Rule-guided GNN 传播 ----
        # Step 3: Rule-guided attention（规则调控信息传播）
        # Step 4: 关系推理（图传播）
        for i in range(self.num_gnn_layers):
            residual = self.residual_proj[i](x)
            x_new = self.gnn_layers[i](x, edge_index, rule_bias)[0]
            x = F.relu(self.layer_norms[i](x_new + residual))
            x = self.dropout(x)

        # ---- Step 5: Task-guided Gating ----
        # 目标驱动注意：强化与洗钱检测相关的节点，抑制无关节点
        basic_gate = torch.sigmoid((x * self.task_vector).sum(dim=-1, keepdim=True))
        adaptive_gate = torch.sigmoid(self.task_gate_mlp(x))
        gate = 0.7 * basic_gate + 0.3 * adaptive_gate
        x = x * gate

        # ---- Step 6: Subgraph Aggregation ----
        # 从个体 → 团体抽象（high-level reasoning）
        if batch is not None:
            graph_repr = self._global_mean_pool(x, batch)
            logit = self.score_mlp(graph_repr)
            prob = torch.sigmoid(logit)
            return x, logit, prob

        # ---- Step 7: 输出 ----
        logit = self.score_mlp(x)
        prob = torch.sigmoid(logit)
        return x, logit, prob

    def get_attention_weights(self) -> Optional[List[torch.Tensor]]:
        """收集所有 GNN 层的注意力权重用于可视化。"""
        weights = []
        for layer in self.gnn_layers:
            if hasattr(layer, "_last_attn") and layer._last_attn is not None:
                weights.append(layer._last_attn)
        return weights if weights else None

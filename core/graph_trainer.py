"""
CryptopiaHacker 图级别节点分类训练器（增强版 V2）。

优化改进：
1. 增强特征：节点序列 3 维 [amount_norm, log_amount, time_diff] + 节点静态特征 10 维
2. CA1 支持 pack_padded_sequence 处理变长序列
3. VTA/DA 信号正确启用：epoch 级别 RPE → CA3 + MPFC
4. 规则编码器使用 log_amount 还原原始金额进行阈值匹配
5. 合理化的损失函数
"""

from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from core.aml_dataset import build_transaction_summary
from core.aml_dataset import PREPROCESS_SCHEMA_VERSION
from core.aml_dataset_cryptopia_graph import preprocess_cryptopia_graph
from models import CA1_TTPM, CA3_AGM, MPFC
from models.vta import compute_da_signal, compute_global_da_signal
from utils import (
    CheckpointManager,
    ClassificationMetrics,
    Config,
    Logger,
    MetricTracker,
    Visualizer,
    compute_alert_level_metrics,
    compute_hit_at_k,
    compute_subgraph_coverage,
    set_seed,
)


class GraphNodeTrainer:
    """
    图级别节点分类训练器（增强版 V2）。
    """

    def __init__(self, cfg: Config, resume: bool = False):
        self.cfg = cfg
        self.device = self._resolve_device()
        self.resume = resume
        self.resume_ckpt = getattr(cfg.experiment, "resume_ckpt", None)

        # 输出目录
        exp_root = os.path.join(cfg.experiment.output_dir, "cryptopia", cfg.experiment.name)

        if resume:
            if self.resume_ckpt:
                self.output_dir = os.path.dirname(os.path.dirname(self.resume_ckpt))
            else:
                self.output_dir = self._find_latest_run_dir(exp_root)
                if self.output_dir is None:
                    raise FileNotFoundError(f"No existing run directory under {exp_root}")
        else:
            run_suffix = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(exp_root, run_suffix)
            os.makedirs(self.output_dir, exist_ok=True)

        self.log_dir = os.path.join(self.output_dir, "log")
        self.fig_dir = os.path.join(self.output_dir, "figures")
        self.ckpt_dir = os.path.join(self.output_dir, "ckpt")
        self.tb_dir = os.path.join(self.output_dir, "tensorboard")
        self.results_dir = os.path.join(self.output_dir, "results")

        for d in (self.log_dir, self.fig_dir, self.ckpt_dir, self.tb_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        self.logger = Logger(log_dir=self.log_dir, name=cfg.experiment.name, console=True)
        self.tb_logger = Logger(log_dir=self.tb_dir, name=f"{cfg.experiment.name}_tb", console=False)
        self.visualizer = Visualizer(save_dir=self.fig_dir)
        self.ckpt_manager = CheckpointManager(ckpt_dir=self.ckpt_dir)

        # 数据
        self.graph_data: Dict = {}
        self.n_nodes: int = 0
        self.n_edges: int = 0

        # 模型
        self.models: Dict[str, nn.Module] = {}
        self.optimizer: Optional[optim.Optimizer] = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.global_step = 0
        self.epoch_losses: List[float] = []
        self._ca3_update_memory = False
        # VTA: 缓存上一轮 da_signal 用于时序平滑
        self._prev_da: Optional[torch.Tensor] = None
        self._last_edge_trace: List[Dict[str, object]] = []
        # 节点静态特征的归一化参数
        self._static_feat_mean: Optional[torch.Tensor] = None
        self._static_feat_std: Optional[torch.Tensor] = None
        self._save_config_snapshot()

    @staticmethod
    def _find_latest_run_dir(exp_root: str) -> Optional[str]:
        if not os.path.exists(exp_root):
            return None
        run_dirs = [
            d for d in os.listdir(exp_root)
            if d.startswith("run_") and os.path.isdir(os.path.join(exp_root, d))
        ]
        if not run_dirs:
            return None
        run_dirs.sort(reverse=True)
        return os.path.join(exp_root, run_dirs[0])

    def _resolve_device(self) -> torch.device:
        cfg_device = self.cfg.experiment.device
        if cfg_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg_device)

    def _log_config(self):
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.cfg.experiment.name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Mode: Graph Node Classification (V2 Enhanced)")
        use_llm = "wo_llm" not in self.cfg.ablation.remove_modules
        llm_status = "MPFC (with LLM)" if use_llm else "MPFC (without LLM)"
        self.logger.info(f"Model: {llm_status}")
        self.logger.info(f"Seed: {self.cfg.experiment.seed}")
        self.logger.info(f"Ablation: remove={self.cfg.ablation.remove_modules}")
        has_vta = "vta" not in self.cfg.ablation.remove_modules
        self.logger.info(f"VTA/DA: {'enabled' if has_vta else 'disabled'}")
        self.logger.info("-" * 60)
        self.logger.info(
            f"Graph: {self.n_nodes} nodes, {self.n_edges} edges, "
            f"train={int(self.graph_data['train_mask'].sum())}, "
            f"val={int(self.graph_data['val_mask'].sum())}, "
            f"test={int(self.graph_data['test_mask'].sum())}"
        )
        self.logger.info(
            f"Train: epochs={self.cfg.train.epochs}, "
            f"lr={self.cfg.train.lr}, "
            f"batch_size={self.cfg.data.batch_size}, "
            f"pos_weight={self.cfg.train.pos_weight}, "
            f"focal_gamma={self.cfg.train.focal_gamma}, "
            f"rpe_beta={self.cfg.train.rpe_beta}"
        )
        self.logger.info("=" * 60)

    def _save_config_snapshot(self):
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg.to_dict(), f, allow_unicode=True, sort_keys=False)

    def _save_metadata(self):
        metadata = {
            "dataset": "cryptopia",
            "data_path": self.cfg.data.data_path,
            "preprocessed_path": self.cfg.data.preprocessed_path,
            "seed": self.cfg.experiment.seed,
            "schema_version": self.graph_data.get("schema_version", PREPROCESS_SCHEMA_VERSION),
            "split_unit": self.graph_data.get("split_unit", "graph_node"),
            "n_nodes": self.n_nodes,
            "n_edges": self.n_edges,
            "n_groups": int(self.graph_data.get("n_groups", 0)),
            "unknown_group_ratio": float(self.graph_data.get("unknown_group_ratio", 0.0)),
            "train_size": int(self.graph_data["train_mask"].sum()),
            "val_size": int(self.graph_data["val_mask"].sum()),
            "test_size": int(self.graph_data["test_mask"].sum()),
            "smote_applied": False,
            "group_type": self.graph_data.get("group_type", "ml_transit_group"),
            "memory_mode": self.cfg.model.ca3.memory_mode,
            "eval_modes": {
                "alert_metrics": self.cfg.eval.enable_alert_metrics,
                "subgraph_metrics": self.cfg.eval.enable_subgraph_metrics,
                "eval_da_mode": self.cfg.eval.eval_da_mode,
                "vta_mode": self.cfg.vta.mode,
                "global_da_mode": True,
            },
            "node_feature_schema": ["ca3_embedding", "node_static_feat", "score_micro"],
        }
        with open(os.path.join(self.results_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _load_data(self):
        """加载或预处理图数据。"""
        cfg = self.cfg
        data_path = cfg.data.preprocessed_path

        if not os.path.exists(data_path) or cfg.data.regenerate:
            self.logger.info(f"Preprocessing Cryptopia graph from {cfg.data.data_path}...")
            self.graph_data = preprocess_cryptopia_graph(
                data_dir=cfg.data.data_path,
                max_seq_len=cfg.data.window_size,
                min_tx_per_addr=1,
                save_path=data_path,
                val_ratio=cfg.data.val_ratio,
                test_ratio=cfg.data.test_ratio,
                random_state=cfg.experiment.seed,
            )
        else:
            self.logger.info(f"Loading graph data from {data_path}...")
            self.graph_data = torch.load(data_path)
            required = ["schema_version", "node_seq", "node_seq_len", "node_labels",
                        "edge_index", "edge_attr", "edge_raw_amount", "node_static_feat",
                        "train_mask", "val_mask", "test_mask", "group_ids"]
            if self.graph_data.get("schema_version") != PREPROCESS_SCHEMA_VERSION or not all(k in self.graph_data for k in required):
                self.logger.info("Graph data outdated, reprocessing...")
                self.graph_data = preprocess_cryptopia_graph(
                    data_dir=cfg.data.data_path,
                    max_seq_len=cfg.data.window_size,
                    min_tx_per_addr=1,
                    save_path=data_path,
                    val_ratio=cfg.data.val_ratio,
                    test_ratio=cfg.data.test_ratio,
                    random_state=cfg.experiment.seed,
                )

        self.n_nodes = self.graph_data["node_labels"].size(0)
        self.n_edges = self.graph_data["edge_index"].size(1)

        # 归一化节点静态特征（在数据集层面做 z-score）
        if "node_static_feat" in self.graph_data:
            sf = self.graph_data["node_static_feat"]
            sf_mean = sf.mean(dim=0, keepdim=True)
            sf_std = sf.std(dim=0, keepdim=True).clamp(min=1e-6)
            self.graph_data["node_static_feat"] = (sf - sf_mean) / sf_std
            self.logger.info(f"Node static feat normalized: {sf.shape}")

        self.logger.info(
            f"Graph loaded: {self.n_nodes} nodes, {self.n_edges} edges, "
            f"max_seq_len={self.graph_data['node_seq'].size(1)}, "
            f"feat_dim={self.graph_data['node_seq'].size(2)}"
        )
        self._save_metadata()

    def _build_models(self):
        """构建 CA1 + CA3 + MPFC 模型。"""
        cfg = self.cfg
        remove_modules = set(cfg.ablation.remove_modules)

        # CA1: node_seq 现在是 [N, max_seq_len, 3]: [amount_norm, log_amount, time_diff]
        if "ca1" not in remove_modules:
            ca1 = CA1_TTPM(
                feature_dim=3,
                hidden_dim=cfg.model.ca1.hidden_dim or cfg.model.hidden_dim,
                n_types=0,
            ).to(self.device)
            self.models["ca1"] = ca1
            self.logger.info(f"CA1 built (hidden={cfg.model.ca1.hidden_dim})")

        # CA3: 记忆增强
        if "ca3" not in remove_modules:
            ca3 = CA3_AGM(
                emb_dim=cfg.model.ca3.emb_dim or cfg.model.hidden_dim,
                num_groups=max(int(self.graph_data.get("n_groups", 0)), 1),
                rpe_dim=1,
                memory_momentum=getattr(cfg.model.ca3, "memory_momentum", 0.9),
                memory_mode=getattr(cfg.model.ca3, "memory_mode", "explicit_group"),
                update_mode=getattr(cfg.model.ca3, "update_mode", "ema_group_proto"),
            ).to(self.device)
            self.models["ca3"] = ca3
            self.logger.info(f"CA3 built (emb_dim={cfg.model.ca3.emb_dim}, groups={ca3.num_groups})")

        # MPFC: 图神经网络节点分类
        if "mpfc" not in remove_modules:
            use_llm = "wo_llm" not in cfg.ablation.remove_modules
            llm_config = {
                "model_name": cfg.model.llm.model_name,
                "use_api": cfg.model.llm.use_api,
                "api_url": cfg.model.llm.api_url,
                "api_key": cfg.model.llm.api_key,
                "rule_update_frequency": cfg.model.llm.rule_update_frequency,
                "max_rules": cfg.model.llm.max_rules,
            }
            mpfc = MPFC(
                emb_dim=cfg.model.hidden_dim,
                edge_attr_dim=cfg.model.edge_attr_dim,  # 3 dims now
                hidden_dim=cfg.model.hidden_dim,
                num_gnn_layers=cfg.model.mpfc.gnn_layers,
                num_heads=cfg.model.mpfc.gnn_heads,
                dropout=cfg.model.dropout,
                llm_config=llm_config,
                use_llm=use_llm,
                input_dim=cfg.model.mpfc.input_dim or (cfg.model.hidden_dim + int(self.graph_data["node_static_feat"].size(1)) + 1),
            ).to(self.device)
            mpfc.set_output_dir(self.output_dir)
            self.models["mpfc"] = mpfc
            llm_status = "with LLM" if use_llm else "without LLM (ablation)"
            self.logger.info(f"MPFC built ({llm_status}, layers={cfg.model.mpfc.gnn_layers})")
        else:
            static_dim = int(self.graph_data["node_static_feat"].size(1)) if "node_static_feat" in self.graph_data else 0
            self.models["classifier"] = nn.Linear(cfg.model.hidden_dim + static_dim + 1, 1).to(self.device)

        # Optimizer
        params = []
        for m in self.models.values():
            params.extend(m.parameters())
        self.optimizer = optim.Adam(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.logger.info(f"Optimizer: Adam (lr={cfg.train.lr}, weight_decay={cfg.train.weight_decay})")

    def _encode_all_nodes(self, da_signal: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对所有节点的交易序列通过 CA1 → CA3 编码。

        改进点：
        - 传递 seq_len 到 CA1 以使用 pack_padded_sequence
        - 全局 da_signal 调制 CA3
        - 拼接节点静态特征

        Returns:
            node_features: [N, hidden_dim + static_feat_dim] 节点特征（即 CA1/CA3 编码 + 静态特征）
            score_micros: [N, 1] 微观异常分数
        """
        cfg = self.cfg
        has_ca1 = "ca1" in self.models
        has_ca3 = "ca3" in self.models

        node_seq = self.graph_data["node_seq"].to(self.device)           # [N, max_seq_len, 3]
        node_seq_len = self.graph_data["node_seq_len"].to(self.device)   # [N]
        node_group_ids = self.graph_data.get("group_ids")
        if node_group_ids is not None:
            node_group_ids = node_group_ids.to(self.device)

        N = node_seq.size(0)
        batch_size = cfg.data.batch_size or 256
        hidden_dim = cfg.model.hidden_dim

        embeddings = torch.zeros(N, hidden_dim, device=self.device)
        score_micros = torch.zeros(N, 1, device=self.device)

        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_seq = node_seq[start:end]   # [B, max_seq_len, 3]
            batch_lens = node_seq_len[start:end]  # [B]

            B = end - start

            if has_ca1:
                # 传递 seq_len 给 CA1，启用 pack_padded_sequence
                h, score_micro = self.models["ca1"](batch_seq, seq_len=batch_lens)
            else:
                h = torch.zeros(B, hidden_dim, device=self.device)
                score_micro = torch.zeros(B, 1, device=self.device)

            # CA3 接收 da_signal 调制
            if has_ca3 and da_signal is not None:
                # da_signal 是标量（全局调制），扩展为 batch 维度
                if da_signal.dim() == 0:
                    # 标量 da_signal: 扩展为 [B, 1]
                    batch_da = da_signal.unsqueeze(0).unsqueeze(1).expand(B, 1)
                elif da_signal.dim() == 1 and da_signal.size(0) == 1:
                    # 1D [1] da_signal: 扩展为 [B, 1]
                    batch_da = da_signal.unsqueeze(1).expand(B, 1)
                else:
                    batch_da = da_signal[start:end]
                batch_group_ids = node_group_ids[start:end] if node_group_ids is not None else None
                h = self.models["ca3"](
                    h,
                    group_ids=batch_group_ids,
                    da_signal=batch_da,
                    update_memory=self._ca3_update_memory,
                )

            embeddings[start:end] = h
            score_micros[start:end] = score_micro

        # 拼接节点静态特征
        if "node_static_feat" in self.graph_data:
            static_feat = self.graph_data["node_static_feat"].to(self.device)  # [N, 10]
            node_features = torch.cat([embeddings, static_feat], dim=-1)        # [N, hidden_dim + 10]
        else:
            node_features = embeddings

        return node_features, score_micros

    def _compute_da_signal(self, prob: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
        """
        计算 epoch 级别的多巴胺 RPE 信号。

        委托给 vta.compute_global_da_signal，实现模块复用。
        VTA 只对训练集节点计算 RPE，然后取均值作为全局调制信号。
        这相当于一个 "全局注意警觉" 信号：
        - 如果当前模型在训练集上整体预测错误大 → 增强注意锐利度
        - 如果预测接近正确 → 放松调节
        """
        train_prob = prob[mask].detach()
        train_labels = labels[mask].detach()

        da_signal = compute_global_da_signal(
            prob=train_prob,
            y=train_labels,
            rpe_beta=self.cfg.train.rpe_beta,
            momentum=0.9,
            prev_da=float(self._prev_da) if self._prev_da is not None else None,
        )

        return da_signal

    def _train_epoch(self) -> Tuple[float, Dict]:
        """训练一个 epoch。"""
        cfg = self.cfg
        has_ca1 = "ca1" in self.models
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.train()
        self._ca3_update_memory = True

        # ---- 1. 对所有节点编码 ----
        # 第一个 epoch da_signal=1.0（无调制），之后从 VTA 计算
        da_signal_val = getattr(self, '_current_da', 1.0)
        if isinstance(da_signal_val, torch.Tensor):
            da_signal_val = da_signal_val.item()
        node_features, score_micros = self._encode_all_nodes(
            da_signal=torch.tensor(da_signal_val, device=self.device)
        )

        # ---- 2. 构建全图 MPFC 输入 ----
        node_x = torch.cat([node_features, score_micros], dim=-1)  # [N, h+static+1]
        edge_index = self.graph_data["edge_index"].to(self.device)
        edge_attr = self.graph_data["edge_attr"].to(self.device)    # [E, 3]
        train_mask = self.graph_data["train_mask"].to(self.device)
        labels = self.graph_data["node_labels"].to(self.device)
        group_ids = self.graph_data.get("group_ids")
        if group_ids is not None:
            group_ids = group_ids.to(self.device)

        # 交易摘要（用于 LLM）
        transaction_summary = getattr(self, '_transaction_summary', None)
        if transaction_summary is None and has_mpfc:
            try:
                transaction_summary = build_transaction_summary(
                    edge_attr, labels, raw_amounts=self.graph_data.get("edge_raw_amount")
                )
                self._transaction_summary = transaction_summary
            except Exception as e:
                self.logger.warning(f"Failed to build transaction summary: {e}")

        # ---- 3. MPFC 前向（携带 da_signal） ----
        da_signal_tensor = float(da_signal_val)

        if has_mpfc:
            node_out, logit, prob, edge_trace = self.models["mpfc"](
                node_x, edge_index, edge_attr,
                transaction_summary=transaction_summary,
                da_signal=da_signal_tensor,
            )
        else:
            logit = self.models["classifier"](node_x)
            prob = torch.sigmoid(logit)
            edge_trace = []
        self._last_edge_trace = edge_trace

        pred_logit = logit.squeeze(-1)
        pred_prob = prob.squeeze(-1)

        # ---- 4. 计算 DA 信号（VTA） ----
        da_signal_val = self._compute_da_signal(pred_prob, labels, train_mask)
        self._current_da = da_signal_val
        self._prev_da = torch.tensor(da_signal_val)

        # ---- 5. Loss（只计算 train_mask） ----
        # 改进：使用更平衡的 focal loss 设置
        gamma = getattr(cfg.train, 'focal_gamma', 2.0)
        if gamma > 0:
            pos_weight = getattr(cfg.train, 'pos_weight', 5.0)
            # 更平衡的 alpha：pos_weight 控制正类权重，但不过度压制负类
            alpha_pos = pos_weight / (pos_weight + 1.0)
            alpha_neg = 1.0 / (pos_weight + 1.0)

            bce = F.binary_cross_entropy_with_logits(
                pred_logit[train_mask], labels[train_mask], reduction='none'
            )
            pt = torch.where(
                labels[train_mask] == 1.0,
                pred_prob[train_mask],
                1.0 - pred_prob[train_mask]
            )
            alpha = torch.where(
                labels[train_mask] == 1.0, alpha_pos, alpha_neg
            )
            focal_weight = alpha * (1.0 - pt) ** gamma
            loss = (focal_weight * bce).mean()
        else:
            loss_fn = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(cfg.train.pos_weight, device=self.device)
            )
            loss = loss_fn(pred_logit[train_mask], labels[train_mask])

        # ---- 6. 反向传播 ----
        if "ca3" in self.models and group_ids is not None:
            valid_mask = group_ids >= 0
            unique_groups = group_ids[valid_mask].unique(sorted=True) if valid_mask.any() else torch.zeros((0,), device=self.device, dtype=torch.long)
            group_repr = []
            for gid in unique_groups.tolist():
                member_mask = group_ids == gid
                group_repr.append(node_features[member_mask, :cfg.model.hidden_dim].mean(dim=0))
            if group_repr:
                group_repr_t = torch.stack(group_repr, dim=0)
                self.models["ca3"].update_group_memory_from_aggregates(group_repr_t.detach(), unique_groups.detach())
                memory_loss = self.models["ca3"].memory_loss(group_repr_t, unique_groups)
            else:
                memory_loss = torch.zeros((), device=self.device)
            loss = loss + 0.1 * memory_loss
        self.optimizer.zero_grad()
        loss.backward()
        if cfg.train.grad_clip > 0:
            for m in self.models.values():
                torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.train.grad_clip)
        self.optimizer.step()
        self.global_step += 1

        # ---- 7. 训练指标 ----
        tracker = MetricTracker()
        with torch.no_grad():
            tracker.update(pred_prob[train_mask].cpu(), labels[train_mask].cpu())

        train_metrics = tracker.compute()
        avg_loss = loss.item()
        self.epoch_losses.append(avg_loss)

        if cfg.train.log_interval > 0 and self.current_epoch % cfg.train.log_interval == 0:
            self.logger.info(
                f"Epoch {self.current_epoch}/{cfg.train.epochs} "
                f"loss {avg_loss:.4f} "
                f"train_f1={train_metrics['f1']:.4f} "
                f"train_auc={train_metrics.get('auc', 0):.4f}"
            )

        return avg_loss, train_metrics

    @torch.no_grad()
    def _evaluate(self, mask_key: str, threshold: float = 0.5) -> Dict:
        """在指定 mask 的节点上评估。"""
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.eval()
        self._ca3_update_memory = False

        # 评估时不使用 da_signal（用恒等信号 1.0）
        node_features, score_micros = self._encode_all_nodes(
            da_signal=torch.tensor(1.0, device=self.device)
        )
        node_x = torch.cat([node_features, score_micros], dim=-1)

        edge_index = self.graph_data["edge_index"].to(self.device)
        edge_attr = self.graph_data["edge_attr"].to(self.device)
        mask = self.graph_data[mask_key].to(self.device)
        labels = self.graph_data["node_labels"].to(self.device)

        if has_mpfc:
            _, logit, prob, edge_trace = self.models["mpfc"](node_x, edge_index, edge_attr, da_signal=1.0)
        else:
            logit = self.models["classifier"](node_x)
            prob = torch.sigmoid(logit)
            edge_trace = []
        self._last_edge_trace = edge_trace

        pred_prob = prob.squeeze(-1)[mask]
        y_true = labels[mask].cpu().numpy()
        y_prob = pred_prob.cpu().numpy()
        metrics = ClassificationMetrics(y_true, y_prob, threshold).report()
        group_ids_np = self.graph_data.get("group_ids")
        if group_ids_np is not None:
            group_ids_np = group_ids_np.cpu().numpy()[mask.cpu().numpy()]
            metrics.update(
                compute_alert_level_metrics(
                    group_ids=group_ids_np,
                    y_true=y_true,
                    y_prob=y_prob,
                    threshold=threshold,
                    agg=self.cfg.eval.alert_agg,
                )
            )
            metrics["hit_at_k"] = compute_hit_at_k(
                group_ids=group_ids_np,
                y_true=y_true,
                y_prob=y_prob,
                k=self.cfg.eval.hit_k,
                agg=self.cfg.eval.alert_agg,
            )
            metrics.update(
                compute_subgraph_coverage(
                    true_group_ids=group_ids_np,
                    pred_scores=y_prob,
                    top_k=self.cfg.eval.hit_k,
                )
            )
        return metrics

    @torch.no_grad()
    def _collect_predictions(self, mask_key: str) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.eval()
        self._ca3_update_memory = False

        node_features, score_micros = self._encode_all_nodes(
            da_signal=torch.tensor(1.0, device=self.device)
        )
        node_x = torch.cat([node_features, score_micros], dim=-1)

        edge_index = self.graph_data["edge_index"].to(self.device)
        edge_attr = self.graph_data["edge_attr"].to(self.device)
        mask = self.graph_data[mask_key].to(self.device)
        labels = self.graph_data["node_labels"].to(self.device)

        if has_mpfc:
            _, logit, prob, edge_trace = self.models["mpfc"](node_x, edge_index, edge_attr, da_signal=1.0)
        else:
            logit = self.models["classifier"](node_x)
            prob = torch.sigmoid(logit)
            edge_trace = []

        pred_prob = prob.squeeze(-1)[mask]
        y_true = labels[mask].cpu().numpy()
        y_prob = pred_prob.cpu().numpy()
        self._last_edge_trace = edge_trace
        return y_true, y_prob, edge_trace

    @torch.no_grad()
    def _search_best_threshold(self) -> Tuple[float, Dict]:
        """在验证集上搜索最佳阈值。"""
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.eval()
        self._ca3_update_memory = False

        node_features, score_micros = self._encode_all_nodes(
            da_signal=torch.tensor(1.0, device=self.device)
        )
        node_x = torch.cat([node_features, score_micros], dim=-1)

        edge_index = self.graph_data["edge_index"].to(self.device)
        edge_attr = self.graph_data["edge_attr"].to(self.device)
        val_mask = self.graph_data["val_mask"].to(self.device)
        labels = self.graph_data["node_labels"].to(self.device)

        if has_mpfc:
            _, logit, prob, _ = self.models["mpfc"](node_x, edge_index, edge_attr, da_signal=1.0)
        else:
            logit = self.models["classifier"](node_x)
            prob = torch.sigmoid(logit)

        pred_prob = prob.squeeze(-1)[val_mask]
        y_true = labels[val_mask].cpu().numpy()
        y_prob = pred_prob.cpu().numpy()

        thresholds = np.linspace(0.05, 0.95, 91)
        best_f1, best_thr, best_metrics = -1.0, 0.5, {}

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            metrics = ClassificationMetrics(y_true, y_prob, thr).report()
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_thr = thr
                best_metrics = metrics

        return best_thr, best_metrics

    def train(self) -> Dict:
        """执行完整训练流程。"""
        cfg = self.cfg
        set_seed(cfg.experiment.seed)

        self._load_data()
        self._log_config()
        self._build_models()

        # 断点续训
        start_epoch = 1
        resume_ok = False
        if self.resume:
            if self.resume_ckpt and os.path.exists(self.resume_ckpt):
                ckpt = torch.load(self.resume_ckpt, map_location=self.device)
                resume_ok = ckpt is not None and "model_state_dict" in ckpt
            elif self.ckpt_manager.has_checkpoint():
                ckpt = self.ckpt_manager.load_latest(self.device)
                resume_ok = ckpt is not None and "model_state_dict" in ckpt

        if resume_ok:
            state_dicts = ckpt["model_state_dict"]
            for name, state_dict in state_dicts.items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)
            if "optimizer_state_dict" in ckpt and self.optimizer is not None:
                self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            self.global_step = ckpt.get("global_step", 0)
            self.best_val_f1 = ckpt.get("best_val_f1", -1.0)
            self.best_threshold = ckpt.get("best_threshold", 0.5)
            self.patience_counter = ckpt.get("patience_counter", 0)
            self._prev_da = ckpt.get("prev_da", None)
            if self._prev_da is not None:
                self._prev_da = torch.tensor(self._prev_da)
            self.logger.info(f"Resumed from epoch {ckpt.get('epoch', 0)}")

        train_start = time.time()

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            avg_loss, train_metrics = self._train_epoch()
            val_threshold, val_metrics = self._search_best_threshold()

            if val_metrics["f1"] > self.best_val_f1:
                self.best_val_f1 = val_metrics["f1"]
                self.best_threshold = val_threshold
                self.patience_counter = 0

                state_dicts = {n: m.state_dict() for n, m in self.models.items()}
                ckpt_data = {
                    "model_state_dict": state_dicts,
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "best_val_f1": self.best_val_f1,
                    "best_threshold": self.best_threshold,
                    "patience_counter": self.patience_counter,
                    "prev_da": self._prev_da.item() if isinstance(self._prev_da, torch.Tensor) else self._prev_da,
                }
                self.ckpt_manager.save_best(ckpt_data, val_metrics["f1"])
            else:
                self.patience_counter += 1

            state_dicts = {n: m.state_dict() for n, m in self.models.items()}
            self.ckpt_manager.save(
                {
                    "model_state_dict": state_dicts,
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "best_val_f1": self.best_val_f1,
                    "best_threshold": self.best_threshold,
                    "patience_counter": self.patience_counter,
                    "prev_da": self._prev_da.item() if isinstance(self._prev_da, torch.Tensor) else self._prev_da,
                },
                epoch=epoch, step=self.global_step,
            )

            epoch_time = time.time() - epoch_start

            da_str = ""
            if hasattr(self, '_current_da'):
                da_str = f" | da={self._current_da:.3f}"
            log_msg = (
                f"Epoch {epoch}/{cfg.train.epochs} | "
                f"time={epoch_time:.1f}s | "
                f"loss={avg_loss:.4f} | "
                f"train_f1={train_metrics['f1']:.4f} "
                f"train_auc={train_metrics.get('auc', 0):.4f} | "
                f"val_f1={val_metrics['f1']:.4f} "
                f"val_auc={val_metrics.get('auc', 0):.4f} "
                f"val_thr={val_threshold:.2f}{da_str}"
            )
            self.logger.info(log_msg)

            self.tb_logger.log_scalar("train/loss", avg_loss, epoch)
            self.tb_logger.log_scalar("train/f1", train_metrics["f1"], epoch)
            self.tb_logger.log_scalar("train/auc", train_metrics.get("auc", 0), epoch)
            self.tb_logger.log_scalar("val/f1", val_metrics["f1"], epoch)
            self.tb_logger.log_scalar("val/auc", val_metrics.get("auc", 0), epoch)
            self.tb_logger.log_scalar("val/threshold", val_threshold, epoch)
            if hasattr(self, '_current_da'):
                self.tb_logger.log_scalar("train/da_signal", self._current_da, epoch)

            if cfg.train.patience > 0 and self.patience_counter >= cfg.train.patience:
                self.logger.info(f"Early stopping after {epoch} epochs")
                break

        train_time = time.time() - train_start
        self.logger.info(f"Training completed in {train_time:.1f}s")
        self.logger.info(f"Best val F1: {self.best_val_f1:.4f} at threshold {self.best_threshold:.2f}")

        # 测试
        test_metrics = {}
        best_ckpt = self.ckpt_manager.load_best(self.device)
        if best_ckpt:
            state_dicts = best_ckpt["model_state_dict"]
            for name, state_dict in state_dicts.items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)

        test_metrics = self._evaluate("test_mask", threshold=self.best_threshold)
        self.logger.info(
            f"Test: ACC={test_metrics['acc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"PREC={test_metrics['precision']:.4f}, "
            f"REC={test_metrics['recall']:.4f}, "
            f"AUC={test_metrics.get('auc', 0):.4f}, "
            f"AP={test_metrics.get('ap', 0):.4f}"
        )
        for k, v in test_metrics.items():
            self.tb_logger.log_scalar(f"test/{k}", v, 0)

        self.tb_logger.close()
        self._save_node_scores("test_mask")
        self._save_ca3_artifacts()

        return {
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics or {},
            "test_metrics": test_metrics,
            "train_time": train_time,
            "results_dir": self.results_dir,
            "dataset": "cryptopia",
            "smote_applied": False,
            "train_size": int(self.graph_data["train_mask"].sum()),
            "val_size": int(self.graph_data["val_mask"].sum()),
            "test_size": int(self.graph_data["test_mask"].sum()),
        }

    def test(self, checkpoint_path: Optional[str] = None) -> Dict:
        """仅执行测试评估。"""
        self._load_data()
        self._build_models()

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dicts = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            for name, state_dict in state_dicts.items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            best_ckpt = self.ckpt_manager.load_best(self.device)
            if best_ckpt:
                state_dicts = best_ckpt["model_state_dict"]
                for name, state_dict in state_dicts.items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)

        test_metrics = self._evaluate("test_mask")
        self._save_node_scores("test_mask")
        self.logger.info(
            f"Test: ACC={test_metrics['acc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics.get('auc', 0):.4f}"
        )
        return test_metrics

    def _save_ca3_artifacts(self):
        if "ca3" not in self.models:
            return
        state = self.models["ca3"].export_memory_state()
        torch.save(state, os.path.join(self.results_dir, "ca3_memory.pt"))
        stats = {
            "num_groups": int(state["num_groups"]),
            "memory_shape": list(state["memory_bank"].shape),
        }
        with open(os.path.join(self.results_dir, "ca3_memory_stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        with open(os.path.join(self.results_dir, "group_meta.json"), "w", encoding="utf-8") as f:
            json.dump(self.models["ca3"].export_group_meta(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def _aggregate_node_edge_trace(edge_trace: List[Dict[str, object]]) -> Dict[int, Dict[str, object]]:
        node_to_edges: Dict[int, List[Dict[str, object]]] = {}
        for row in edge_trace:
            src = row.get("src")
            dst = row.get("dst")
            if src is None or dst is None:
                continue
            src = int(src)
            dst = int(dst)
            node_to_edges.setdefault(src, []).append(row)
            if dst != src:
                node_to_edges.setdefault(dst, []).append(row)

        aggregated: Dict[int, Dict[str, object]] = {}
        for node_idx, rows in node_to_edges.items():
            if not rows:
                continue
            attention_values = [float(item.get("attention", 0.0) or 0.0) for item in rows]
            top_row = max(rows, key=lambda item: float(item.get("attention", 0.0) or 0.0))
            aggregated[node_idx] = {
                "attention": float(sum(attention_values) / max(len(attention_values), 1)),
                "matched_rule_text": str(top_row.get("matched_rule_text", "") or ""),
                "matched_rule_type": str(top_row.get("matched_rule_type", "") or ""),
                "rule_confidence": float(top_row.get("rule_confidence", 0.0) or 0.0),
            }
        return aggregated

    def _save_node_scores(self, mask_key: str, filename: str = "node_scores.csv"):
        y_true, y_prob, edge_trace = self._collect_predictions(mask_key)
        mask = self.graph_data[mask_key].cpu().numpy().astype(bool)
        node_indices = np.flatnonzero(mask)
        group_ids = self.graph_data.get("group_ids")
        if group_ids is not None:
            group_ids = group_ids.cpu().numpy()
        node_trace = self._aggregate_node_edge_trace(edge_trace)

        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["node_idx", "group_id", "label", "risk_score", "attention", "rule_text", "rule_type", "rule_confidence"])
            for pos, node_idx in enumerate(node_indices):
                group_id = int(group_ids[node_idx]) if group_ids is not None else -1
                trace = node_trace.get(int(node_idx), {})
                writer.writerow([
                    int(node_idx),
                    group_id,
                    int(y_true[pos]),
                    float(y_prob[pos]),
                    float(trace.get("attention", 0.0)),
                    str(trace.get("matched_rule_text", "")),
                    str(trace.get("matched_rule_type", "")),
                    float(trace.get("rule_confidence", 0.0)),
                ])
        self._save_edge_trace(edge_trace)

    def _save_edge_trace(self, edge_trace: List[Dict[str, object]], filename: str = "edge_trace.csv"):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["edge_id", "src", "dst", "attention", "rule_bias", "rule_match_score", "matched_rule_text", "matched_rule_type", "rule_confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in edge_trace:
                writer.writerow({key: row.get(key, "") for key in fieldnames})

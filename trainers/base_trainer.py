"""
基础训练器 —— 统一训练/验证/测试管线。

功能：
- 训练循环（前向/反向/梯度裁剪）
- 验证（含最佳阈值搜索）
- 测试评估
- Checkpoint 自动保存
- TensorBoard 日志记录
- 早期停止
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from datasets.aml_dataset import (
    build_batch_graph,
    build_transaction_summary,
    get_dataloaders,
    preprocess_data,
)
from models import CA1_TTPM, CA3_AGM, MPFC, vta_weighted_loss
from utils import (
    CheckpointManager,
    ClassificationMetrics,
    Config,
    Logger,
    MetricTracker,
    Visualizer,
    set_seed,
)


class BaseTrainer:
    """
    基础训练器。

    职责：
    1. 根据配置构建模型
    2. 训练循环
    3. 验证 + 最佳阈值搜索
    4. 测试评估
    5. 日志 + Checkpoint + 可视化
    """

    def __init__(self, cfg: Config, resume: bool = False):
        self.cfg = cfg
        self.device = self._resolve_device()
        self.resume = resume
        self.resume_ckpt = getattr(cfg.experiment, "resume_ckpt", None)

        # --------------------------------------------------
        # 输出目录结构：
        #   <output_dir>/<exp_name>/run_YYYYMMDD_HHMMSS/
        #     log/          - 日志文件
        #     figures/      - 可视化图表
        #     ckpt/         - 模型 checkpoint
        #     tensorboard/  - TensorBoard 事件
        #     results/      - 结果 CSV
        # --------------------------------------------------
        exp_root = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)

        if resume:
            if self.resume_ckpt:
                # 指定了 checkpoint 路径，从路径推断 run 目录
                # run_xxx/ckpt/latest.pt → run_xxx
                self.output_dir = os.path.dirname(os.path.dirname(self.resume_ckpt))
            else:
                # 自动找最新的 run_* 目录
                self.output_dir = self._find_latest_run_dir(exp_root)
                if self.output_dir is None:
                    raise FileNotFoundError(
                        f"No existing run directory found under {exp_root} for --resume."
                    )
        else:
            # 正常训练：创建新的时间戳子目录
            run_suffix = datetime.now().strftime("run_%Y%m%d_%H%M%S")
            self.output_dir = os.path.join(exp_root, run_suffix)
            os.makedirs(self.output_dir, exist_ok=True)

        # 子目录
        self.log_dir = os.path.join(self.output_dir, "log")
        self.fig_dir = os.path.join(self.output_dir, "figures")
        self.ckpt_dir = os.path.join(self.output_dir, "ckpt")
        self.tb_dir = os.path.join(self.output_dir, "tensorboard")
        self.results_dir = os.path.join(self.output_dir, "results")

        for d in (self.log_dir, self.fig_dir, self.ckpt_dir, self.tb_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        # 日志（主日志放在 log/ 下）
        self.logger = Logger(
            log_dir=self.log_dir,
            name=cfg.experiment.name,
            console=True,
        )
        # TensorBoard 单独子目录
        self.tb_logger = Logger(
            log_dir=self.tb_dir,
            name=f"{cfg.experiment.name}_tb",
            console=False,
        )

        # 可视化 -> figures/
        self.visualizer = Visualizer(save_dir=self.fig_dir)

        # Checkpoint -> ckpt/（直接写，不再双重嵌套）
        self.ckpt_manager = CheckpointManager(
            ckpt_dir=self.ckpt_dir,
        )

        # 数据
        self.train_loader: Optional[torch.utils.data.DataLoader] = None
        self.val_loader: Optional[torch.utils.data.DataLoader] = None
        self.test_loader: Optional[torch.utils.data.DataLoader] = None
        self.metadata: Dict = {}

        # 模型
        self.models: Dict[str, nn.Module] = {}
        self.optimizer: Optional[optim.Optimizer] = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.global_step = 0

    @staticmethod
    def _find_latest_run_dir(exp_root: str) -> Optional[str]:
        """在 <output_dir>/<exp_name>/ 下查找最新的 run_* 目录。"""
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
        """记录配置到日志。"""
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.cfg.experiment.name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output dir: {self.output_dir}")
        # 判断 MPFC 是否使用 LLM（LLM 是 mPFC 的内置组件）
        use_llm = "wo_llm" not in self.cfg.ablation.remove_modules
        llm_status = "MPFC (with LLM)" if use_llm else "MPFC (without LLM)"
        self.logger.info(f"Model: {llm_status}")
        self.logger.info(f"Seed: {self.cfg.experiment.seed}")
        self.logger.info(f"Ablation: remove={self.cfg.ablation.remove_modules}")
        self.logger.info("-" * 60)

        # 打印关键训练参数
        self.logger.info(
            f"Train: epochs={self.cfg.train.epochs}, "
            f"lr={self.cfg.train.lr}, "
            f"batch_size={self.cfg.data.batch_size}, "
            f"pos_weight={self.cfg.train.pos_weight}, "
            f"focal_gamma={self.cfg.train.focal_gamma}, "
            f"rpe_beta={self.cfg.train.rpe_beta}"
        )
        self.logger.info("=" * 60)

    def _build_models(self):
        """根据配置构建模型。"""
        cfg = self.cfg
        remove_modules = set(cfg.ablation.remove_modules)

        # CA1
        if "ca1" not in remove_modules:
            ca1 = CA1_TTPM(
                feature_dim=cfg.model.ca1.feature_dim or 3,
                hidden_dim=cfg.model.ca1.hidden_dim or cfg.model.hidden_dim,
                n_types=self.metadata.get("n_types", 0),
                type_emb_dim=cfg.model.dim_type,
            ).to(self.device)
            self.models["ca1"] = ca1
            self.logger.info(f"CA1 built (feature_dim={cfg.model.ca1.feature_dim}, hidden={cfg.model.ca1.hidden_dim})")
        else:
            self.logger.info("CA1 module removed by ablation.")

        # CA3
        if "ca3" not in remove_modules:
            ca3 = CA3_AGM(
                emb_dim=cfg.model.ca3.emb_dim or cfg.model.hidden_dim,
                num_groups=max(1, self.metadata.get("n_groups", 1)),
                memory_momentum=cfg.model.ca3.memory_momentum,
            ).to(self.device)
            self.models["ca3"] = ca3
            self.logger.info(f"CA3 built (emb_dim={cfg.model.ca3.emb_dim}, n_groups={self.metadata.get('n_groups', 1)})")
        else:
            self.logger.info("CA3 module removed by ablation.")

        # MPFC（LLM 为 mPFC 内置组件）
        if "mpfc" not in remove_modules:
            # 默认 use_llm=True，由 ablation 中的 wo_llm 控制去除 LLM
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
                edge_attr_dim=cfg.model.edge_attr_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_gnn_layers=cfg.model.mpfc.gnn_layers,
                num_heads=cfg.model.mpfc.gnn_heads,
                dropout=cfg.model.dropout,
                llm_config=llm_config,
                use_llm=use_llm,
            ).to(self.device)
            mpfc.set_output_dir(self.output_dir)  # LLM 规则文件放在 run_xxx/ 根目录
            llm_status = "with LLM" if use_llm else "without LLM (ablation)"
            self.models["mpfc"] = mpfc
            self.logger.info(
                f"MPFC built ({llm_status}, layers={cfg.model.mpfc.gnn_layers}, "
                f"heads={cfg.model.mpfc.gnn_heads})"
            )

        # Optimizer
        params = []
        for m in self.models.values():
            params.extend(m.parameters())
        self.optimizer = optim.Adam(
            params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
        )
        self.logger.info(f"Optimizer: Adam (lr={cfg.train.lr}, weight_decay={cfg.train.weight_decay})")

    def _build_dataloaders(self):
        """构建 DataLoader。"""
        cfg = self.cfg
        data_path = cfg.data.preprocessed_path

        if not os.path.exists(data_path) or cfg.data.regenerate:
            self.logger.info(f"Preprocessing data from {cfg.data.data_path}...")
            data = preprocess_data(
                csv_path=cfg.data.data_path,
                window_size=self.cfg.model.ca1.feature_dim,  # seq_len = feature_dim
                save_path=data_path,
                use_smote=cfg.data.use_smote,
                smote_ratio=cfg.data.smote_ratio,
            )
        else:
            self.logger.info(f"Loading preprocessed data from {data_path}...")
            data = torch.load(data_path)
            required = ["sequences", "labels", "sender_idx", "receiver_idx", "alert_idx", "edge_attr"]
            if not all(k in data for k in required):
                self.logger.info("Preprocessed data outdated, reprocessing...")
                data = preprocess_data(
                    csv_path=cfg.data.data_path,
                    window_size=self.cfg.model.ca1.feature_dim,
                    save_path=data_path,
                    use_smote=cfg.data.use_smote,
                    smote_ratio=cfg.data.smote_ratio,
                )

        self.train_loader, self.val_loader, self.test_loader, self.metadata = get_dataloaders(
            data=data,
            val_ratio=cfg.data.val_ratio,
            test_ratio=cfg.data.test_ratio,
            batch_size=cfg.data.batch_size,
            use_smote=cfg.data.use_smote,
            smote_ratio=cfg.data.smote_ratio,
            random_state=cfg.experiment.seed,
            num_workers=cfg.data.num_workers,
        )

        self.logger.info(
            f"Data: train={self.metadata['train_size']}, "
            f"val={self.metadata['val_size']}, "
            f"test={self.metadata['test_size']}"
        )

    def _train_epoch(self) -> Tuple[float, Dict]:
        """
        训练一个 epoch。

        Returns:
            avg_loss: 平均损失
            train_metrics: 训练集指标字典
        """
        cfg = self.cfg
        has_ca1 = "ca1" in self.models
        has_ca3 = "ca3" in self.models
        has_mpfc = "mpfc" in self.models
        remove_vta = "vta" in cfg.ablation.remove_modules

        for m in self.models.values():
            m.train()

        epoch_loss = 0.0
        tracker = MetricTracker()

        # 交易摘要（用于 MPFC 内置 LLM 规则生成）—— 只在第一轮构建
        transaction_summary = getattr(self, '_transaction_summary', None)
        if transaction_summary is None and has_mpfc:
            try:
                first_batch = next(iter(self.train_loader))
                _, _, _, edge_e_first, _, label_first = first_batch
                transaction_summary = build_transaction_summary(edge_e_first, label_first)
                self._transaction_summary = transaction_summary
                self.logger.info(f"Transaction summary built ({len(transaction_summary)} chars)")
            except (StopIteration, Exception) as e:
                self.logger.warning(f"Failed to build transaction summary: {e}")

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]

            self.optimizer.zero_grad()

            # CA1
            if has_ca1:
                h, score_micro = self.models["ca1"](seq)
            else:
                h = torch.zeros(seq.size(0), cfg.model.hidden_dim, device=self.device)
                score_micro = torch.zeros(seq.size(0), 1, device=self.device)

            # CA3
            if has_ca3:
                h_enhanced = self.models["ca3"](h, group_ids=alert)
            else:
                h_enhanced = h

            # Build graph
            node_x, edge_index, edge_attr_batch, sender_local = build_batch_graph(
                sender, receiver, edge_e, h_enhanced, score_micro
            )

            # MPFC（LLM 规则生成是 mPFC 模块的内置功能）
            if has_mpfc:
                _, logit, prob = self.models["mpfc"](
                    node_x, edge_index, edge_attr_batch,
                    transaction_summary=transaction_summary,
                )
            else:
                # 无 MPFC：直接线性输出
                logit = nn.Linear(node_x.size(-1), 1, device=self.device)(node_x)
                prob = torch.sigmoid(logit)

            # 取发送方对应的预测
            pred_logit = logit[sender_local]
            pred_prob = prob[sender_local].squeeze(-1)

            # 损失
            if remove_vta:
                loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.train.pos_weight, device=self.device))(
                    pred_logit, label.float().view(-1, 1)
                )
            else:
                loss = vta_weighted_loss(
                    logit=pred_logit,
                    y=label,
                    prob=pred_prob.unsqueeze(-1),
                    pos_weight=cfg.train.pos_weight,
                    focal_gamma=cfg.train.focal_gamma,
                    rpe_beta=cfg.train.rpe_beta,
                )

            loss.backward()

            # 梯度裁剪
            if cfg.train.grad_clip > 0:
                for m in self.models.values():
                    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.train.grad_clip)

            self.optimizer.step()
            self.global_step += 1

            epoch_loss += loss.item()
            tracker.update(pred_prob.detach().cpu(), label.cpu())

            # 日志（仅在每个 epoch 结束打印最终 loss）
            if cfg.train.log_interval > 0 and batch_idx % cfg.train.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}/{cfg.train.epochs} "
                    f"batch {batch_idx}/{len(self.train_loader)} "
                    f"loss {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(self.train_loader)
        train_metrics = tracker.compute()

        return avg_loss, train_metrics

    @torch.no_grad()
    def _evaluate(
        self, loader: torch.utils.data.DataLoader, threshold: float = 0.5
    ) -> Dict:
        """
        评估模型。

        Args:
            loader: DataLoader
            threshold: 分类阈值

        Returns:
            metrics: 指标字典
        """
        has_ca1 = "ca1" in self.models
        has_ca3 = "ca3" in self.models
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.eval()

        all_probs = []
        all_labels = []

        for batch in loader:
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]

            if has_ca1:
                h, score_micro = self.models["ca1"](seq)
            else:
                h = torch.zeros(seq.size(0), self.cfg.model.hidden_dim, device=self.device)
                score_micro = torch.zeros(seq.size(0), 1, device=self.device)

            if has_ca3:
                h_enhanced = self.models["ca3"](h, group_ids=alert)
            else:
                h_enhanced = h

            node_x, edge_index, edge_attr_batch, sender_local = build_batch_graph(
                sender, receiver, edge_e, h_enhanced, score_micro
            )

            if has_mpfc:
                _, _, prob = self.models["mpfc"](node_x, edge_index, edge_attr_batch)
            else:
                logit = nn.Linear(node_x.size(-1), 1, device=self.device)(node_x)
                prob = torch.sigmoid(logit)

            all_probs.append(prob[sender_local].squeeze(-1).cpu())
            all_labels.append(label.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()
        y_pred = (y_prob >= threshold).astype(int)

        return ClassificationMetrics(y_true, y_prob, threshold).report()

    @torch.no_grad()
    def _search_best_threshold(
        self, loader: torch.utils.data.DataLoader
    ) -> Tuple[float, Dict]:
        """
        在验证集上搜索最佳阈值。

        Returns:
            best_threshold, best_metrics
        """
        thresholds = np.linspace(0.05, 0.95, 91)

        has_ca1 = "ca1" in self.models
        has_ca3 = "ca3" in self.models
        has_mpfc = "mpfc" in self.models

        for m in self.models.values():
            m.eval()

        all_probs = []
        all_labels = []

        for batch in loader:
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]

            if has_ca1:
                h, score_micro = self.models["ca1"](seq)
            else:
                h = torch.zeros(seq.size(0), self.cfg.model.hidden_dim, device=self.device)
                score_micro = torch.zeros(seq.size(0), 1, device=self.device)

            if has_ca3:
                h_enhanced = self.models["ca3"](h, group_ids=alert)
            else:
                h_enhanced = h

            node_x, edge_index, edge_attr_batch, sender_local = build_batch_graph(
                sender, receiver, edge_e, h_enhanced, score_micro
            )

            if has_mpfc:
                _, _, prob = self.models["mpfc"](node_x, edge_index, edge_attr_batch)
            else:
                logit = nn.Linear(node_x.size(-1), 1, device=self.device)(node_x)
                prob = torch.sigmoid(logit)

            all_probs.append(prob[sender_local].squeeze(-1).cpu())
            all_labels.append(label.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()

        best_f1 = -1.0
        best_threshold = 0.5
        best_metrics = {}

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            metrics = ClassificationMetrics(y_true, y_prob, thr).report()
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = thr
                best_metrics = metrics

        return best_threshold, best_metrics

    def train(self) -> Dict:
        """执行完整训练流程。"""
        cfg = self.cfg
        set_seed(cfg.experiment.seed)
        self._log_config()

        # 构建数据
        self._build_dataloaders()

        # 构建模型
        self._build_models()

        # 断点续训：从 checkpoint 恢复模型和优化器状态
        start_epoch = 1
        resume_ok = False
        if self.resume:
            if self.resume_ckpt:
                # 从指定路径加载
                if os.path.exists(self.resume_ckpt):
                    ckpt = torch.load(self.resume_ckpt, map_location=self.device)
                    resume_ok = ckpt is not None and "model_state_dict" in ckpt
                    if not resume_ok:
                        self.logger.warning(f"Invalid checkpoint: {self.resume_ckpt}")
                else:
                    self.logger.warning(f"Checkpoint not found: {self.resume_ckpt}")
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
            self.logger.info(
                f"Resumed from checkpoint: epoch {ckpt.get('epoch', 0)}, "
                f"global_step {self.global_step}, "
                f"best_val_f1 {self.best_val_f1:.4f}"
            )
        else:
            if self.resume:
                self.logger.warning("Checkpoint not found or invalid, starting from scratch.")
            self.logger.info(f"Starting training for {cfg.train.epochs} epochs...")

        train_start = time.time()

        for epoch in range(start_epoch, cfg.train.epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()

            # 训练
            avg_loss, train_metrics = self._train_epoch()

            # 验证
            if self.val_loader is not None:
                val_threshold, val_metrics = self._search_best_threshold(self.val_loader)

                # 更新最佳模型
                if val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_threshold = val_threshold
                    self.patience_counter = 0

                    # 保存最佳 checkpoint
                    state_dicts = {name: m.state_dict() for name, m in self.models.items()}
                    ckpt_data = {
                        "model_state_dict": state_dicts,
                        "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                        "epoch": epoch,
                        "global_step": self.global_step,
                        "best_val_f1": self.best_val_f1,
                        "best_threshold": self.best_threshold,
                        "patience_counter": self.patience_counter,
                    }
                    self.ckpt_manager.save_best(ckpt_data, val_metrics["f1"])
                else:
                    self.patience_counter += 1
            else:
                val_metrics = None
                val_threshold = 0.5

            # 保存最新 checkpoint
            state_dicts = {name: m.state_dict() for name, m in self.models.items()}
            ckpt_data = {
                "model_state_dict": state_dicts,
                "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                "epoch": epoch,
                "global_step": self.global_step,
                "best_val_f1": self.best_val_f1,
                "best_threshold": self.best_threshold,
                "patience_counter": self.patience_counter,
            }
            self.ckpt_manager.save(
                ckpt_data, epoch=epoch, step=self.global_step,
            )

            epoch_time = time.time() - epoch_start

            # 日志
            log_msg = (
                f"Epoch {epoch}/{cfg.train.epochs} | "
                f"time={epoch_time:.1f}s | "
                f"train_f1={train_metrics['f1']:.4f} | "
                f"train_auc={train_metrics.get('auc', float('nan')):.4f} | "
                f"train_ap={train_metrics.get('ap', float('nan')):.4f}"
            )
            if val_metrics:
                log_msg += (
                    f" | val_f1={val_metrics['f1']:.4f} | "
                    f"val_auc={val_metrics.get('auc', float('nan')):.4f} | "
                    f"val_ap={val_metrics.get('ap', float('nan')):.4f} | "
                    f"val_thr={val_threshold:.2f}"
                )
            self.logger.info(log_msg)

            # TensorBoard
            self.tb_logger.log_scalar("train/loss", avg_loss, epoch)
            self.tb_logger.log_scalar("train/f1", train_metrics["f1"], epoch)
            self.tb_logger.log_scalar("train/auc", train_metrics.get("auc", 0), epoch)
            if val_metrics:
                self.tb_logger.log_scalar("val/f1", val_metrics["f1"], epoch)
                self.tb_logger.log_scalar("val/auc", val_metrics.get("auc", 0), epoch)
                self.tb_logger.log_scalar("val/threshold", val_threshold, epoch)

            # 早期停止
            if (
                cfg.train.patience > 0
                and self.patience_counter >= cfg.train.patience
            ):
                self.logger.info(
                    f"Early stopping after {epoch} epochs "
                    f"(no improvement for {cfg.train.patience} epochs)"
                )
                break

        train_time = time.time() - train_start
        self.logger.info(f"Training completed in {train_time:.1f}s")
        self.logger.info(
            f"Best val F1: {self.best_val_f1:.4f} at threshold {self.best_threshold:.2f}"
        )

        # 测试
        test_metrics = {}
        if self.test_loader is not None:
            # 加载最佳模型
            best_ckpt = self.ckpt_manager.load_best(self.device)
            if best_ckpt:
                state_dicts = (
                    best_ckpt["model_state_dict"]
                    if "model_state_dict" in best_ckpt
                    else best_ckpt
                )
                for name, state_dict in state_dicts.items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)

            test_metrics = self._evaluate(self.test_loader, threshold=self.best_threshold)
            self.logger.info(
                f"Test: ACC={test_metrics['acc']:.4f}, "
                f"F1={test_metrics['f1']:.4f}, "
                f"PREC={test_metrics['precision']:.4f}, "
                f"REC={test_metrics['recall']:.4f}, "
                f"AUC={test_metrics.get('auc', float('nan')):.4f}, "
                f"AP={test_metrics.get('ap', float('nan')):.4f}"
            )

            # TensorBoard
            for k, v in test_metrics.items():
                self.tb_logger.log_scalar(f"test/{k}", v, 0)

        # 可视化
        if cfg.visualization.enabled and self.val_loader is not None:
            self._visualize()

        # 关闭 TensorBoard
        self.tb_logger.close()

        return {
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics or {},
            "test_metrics": test_metrics,
            "train_time": train_time,
            "results_dir": self.results_dir,
        }

    def _visualize(self):
        """生成可视化图表。"""
        cfg = self.cfg.visualization
        if not cfg.save_figures:
            return

        self.logger.info("Generating visualizations...")

        # Loss curve
        if cfg.plot_loss:
            log_path = os.path.join(self.log_dir, f"{self.cfg.experiment.name}.log")
            train_losses = self._extract_losses_from_log(log_path)
            if train_losses:
                self.visualizer.plot_loss_curve(
                    train_losses,
                    name="loss_curve",
                )
            else:
                self.logger.warning("No loss values found for plotting.")

        # ROC & PR curves
        if cfg.plot_roc and self.val_loader is not None:
            try:
                for m in self.models.values():
                    m.eval()
                all_probs = []
                all_labels = []
                with torch.no_grad():
                    has_ca1 = "ca1" in self.models
                    has_ca3 = "ca3" in self.models
                    has_mpfc = "mpfc" in self.models

                    for batch in self.val_loader:
                        seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]
                        if has_ca1:
                            h, score_micro = self.models["ca1"](seq)
                        else:
                            h = torch.zeros(seq.size(0), self.cfg.model.hidden_dim, device=self.device)
                            score_micro = torch.zeros(seq.size(0), 1, device=self.device)
                        if has_ca3:
                            h_enhanced = self.models["ca3"](h, group_ids=alert)
                        else:
                            h_enhanced = h
                        node_x, edge_index, edge_attr_batch, sender_local = build_batch_graph(
                            sender, receiver, edge_e, h_enhanced, score_micro
                        )
                        if has_mpfc:
                            _, _, prob = self.models["mpfc"](node_x, edge_index, edge_attr_batch)
                        else:
                            logit = nn.Linear(node_x.size(-1), 1, device=self.device)(node_x)
                            prob = torch.sigmoid(logit)
                        all_probs.append(prob[sender_local].squeeze(-1).cpu())
                        all_labels.append(label.cpu())

                y_true = torch.cat(all_labels).numpy()
                y_prob = torch.cat(all_probs).numpy()

                self.visualizer.plot_roc_curve(
                    y_true, y_prob,
                    name="roc_curve",
                )
                self.visualizer.plot_pr_curve(
                    y_true, y_prob,
                    name="pr_curve",
                )
                self.visualizer.plot_confusion_matrix(
                    y_true, (y_prob >= self.best_threshold).astype(int),
                    name="confusion_matrix",
                )

                # Log to tensorboard
                self.tb_logger.log_figure("roc_curve", os.path.join(self.fig_dir, "roc_curve.png"))
            except Exception as e:
                self.logger.warning(f"ROC/PR visualization failed: {e}")

    def test(self, checkpoint_path: Optional[str] = None) -> Dict:
        """
        仅执行测试评估。

        Args:
            checkpoint_path: 模型 checkpoint 路径（None 则用最佳 checkpoint）

        Returns:
            test_metrics
        """
        self._build_dataloaders()
        self._build_models()

        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dicts = (
                ckpt["model_state_dict"]
                if "model_state_dict" in ckpt
                else ckpt
            )
            for name, state_dict in state_dicts.items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)
            self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        else:
            best_ckpt = self.ckpt_manager.load_best(self.device)
            if best_ckpt:
                state_dicts = (
                    best_ckpt["model_state_dict"]
                    if "model_state_dict" in best_ckpt
                    else best_ckpt
                )
                for name, state_dict in state_dicts.items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)
                self.logger.info("Loaded best checkpoint")

        test_metrics = self._evaluate(self.test_loader)
        self.logger.info(
            f"Test: ACC={test_metrics['acc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"AUC={test_metrics.get('auc', float('nan')):.4f}, "
            f"AP={test_metrics.get('ap', float('nan')):.4f}"
        )
        return test_metrics

    @staticmethod
    def _extract_losses_from_log(log_path: str) -> List[float]:
        """从日志文件中提取 loss 值（仅用于可视化）。"""
        losses = []
        try:
            with open(log_path, "r") as f:
                for line in f:
                    if "loss=" in line:
                        parts = line.split("loss=")
                        if len(parts) > 1:
                            try:
                                loss_val = float(parts[1].split()[0])
                                losses.append(loss_val)
                            except (ValueError, IndexError):
                                pass
        except (FileNotFoundError, IOError):
            pass
        return losses

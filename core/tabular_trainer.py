"""
TabularTrainer: 基于表格特征的节点分类训练器。

适用于 CryptopiaHacker 数据特征化后的 MLP 训练。
不使用图结构，仅用预计算特征（39 维）进行分类。
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from core.cryptopia_feature_engine import build_cryptopia_features
from utils import (
    CheckpointManager,
    ClassificationMetrics,
    Config,
    Logger,
    MetricTracker,
    set_seed,
)


class TabularMLP(nn.Module):
    """简单 MLP 分类器，替代 CA1+CA3+MPFC 全链路。"""

    def __init__(self, input_dim: int = 39, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logit = self.net(x)
        prob = torch.sigmoid(logit)
        return logit, prob


class TabularTrainer:
    """
    基于表格特征的训练器。

    不涉及图结构、LSTM、GAT。直接使用预计算特征做节点分类。
    但保持与 BaseTrainer 相似的接口以便集成。
    """

    def __init__(self, cfg: Config, resume: bool = False):
        self.cfg = cfg
        self.device = self._resolve_device()
        self.resume = resume

        # 输出目录
        exp_root = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
        run_suffix = datetime.now().strftime("run_%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(exp_root, run_suffix)
        os.makedirs(self.output_dir, exist_ok=True)

        self.log_dir = os.path.join(self.output_dir, "log")
        self.ckpt_dir = os.path.join(self.output_dir, "ckpt")
        self.results_dir = os.path.join(self.output_dir, "results")

        for d in (self.log_dir, self.ckpt_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)

        self.logger = Logger(log_dir=self.log_dir, name=cfg.experiment.name, console=True)
        self.ckpt_manager = CheckpointManager(ckpt_dir=self.ckpt_dir)

        # 数据
        self.data: Dict = {}
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[optim.Optimizer] = None

        # 训练状态
        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.global_step = 0

    def _resolve_device(self) -> torch.device:
        cfg_device = self.cfg.experiment.device
        if cfg_device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(cfg_device)

    def _load_data(self):
        """加载预计算特征或重新计算。"""
        cfg = self.cfg
        feat_path = cfg.data.preprocessed_path or "cryptopia_features.npz"

        if not os.path.exists(feat_path) or cfg.data.regenerate:
            self.logger.info("Computing features...")
            feat_data = build_cryptopia_features(
                data_dir=cfg.data.data_path,
                save_path=feat_path,
            )
        else:
            self.logger.info(f"Loading features from {feat_path}...")
            feat_data = dict(np.load(feat_path, allow_pickle=True))

        self.data = feat_data
        self.n_samples = len(feat_data["y_train"]) + len(feat_data["y_val"]) + len(feat_data["y_test"])
        self.input_dim = feat_data["X_train"].shape[1]
        self.feature_cols = feat_data.get("feature_cols", [])

        heist_rate = feat_data["y_train"].mean()
        self.logger.info(
            f"Data loaded: {self.n_samples} samples, "
            f"{self.input_dim} features, "
            f"heist_rate={heist_rate:.4f}"
        )
        self.logger.info(
            f"  Train: {len(feat_data['y_train'])}, "
            f"Val: {len(feat_data['y_val'])}, "
            f"Test: {len(feat_data['y_test'])}"
        )

    def _build_model(self):
        """构建 MLP 模型。"""
        cfg = self.cfg
        hidden_dim = getattr(cfg.model, "hidden_dim", 128)
        dropout = getattr(cfg.model, "dropout", 0.3)

        self.model = TabularMLP(
            input_dim=self.input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        ).to(self.device)

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=cfg.train.lr,
            weight_decay=getattr(cfg.train, "weight_decay", 1e-4),
        )

        self.logger.info(
            f"TabularMLP built: {self.input_dim} -> {hidden_dim} -> 1"
        )

    def train(self) -> Dict:
        """完整训练流程。"""
        cfg = self.cfg
        set_seed(cfg.experiment.seed)

        self._load_data()
        self._build_model()

        # 数据转 tensor
        X_train = torch.tensor(self.data["X_train"], dtype=torch.float32).to(self.device)
        y_train = torch.tensor(self.data["y_train"], dtype=torch.float32).to(self.device)
        X_val = torch.tensor(self.data["X_val"], dtype=torch.float32).to(self.device)
        y_val = torch.tensor(self.data["y_val"], dtype=torch.float32).to(self.device)
        X_test = torch.tensor(self.data["X_test"], dtype=torch.float32).to(self.device)
        y_test = torch.tensor(self.data["y_test"], dtype=torch.float32).to(self.device)

        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {cfg.experiment.name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info("-" * 60)
        self.logger.info(
            f"Train: epochs={cfg.train.epochs}, lr={cfg.train.lr}, "
            f"batch_size={cfg.data.batch_size}"
        )
        self.logger.info("=" * 60)

        train_start = time.time()

        for epoch in range(1, cfg.train.epochs + 1):
            self.current_epoch = epoch
            self.model.train()

            # Mini-batch training
            perm = torch.randperm(len(X_train))
            batch_size = cfg.data.batch_size or 256
            total_loss = 0.0
            n_batches = 0

            for start in range(0, len(X_train), batch_size):
                end = min(start + batch_size, len(X_train))
                idx = perm[start:end]

                batch_x = X_train[idx]
                batch_y = y_train[idx].unsqueeze(1)

                logit, prob = self.model(batch_x)

                # Focal loss
                pos_weight = getattr(cfg.train, "pos_weight", 5.0)
                gamma = getattr(cfg.train, "focal_gamma", 2.0)

                alpha_pos = pos_weight / (pos_weight + 1.0)
                alpha_neg = 1.0 / (pos_weight + 1.0)

                bce = F.binary_cross_entropy_with_logits(logit, batch_y, reduction="none")
                pt = torch.where(batch_y == 1.0, prob, 1.0 - prob)
                alpha = torch.where(batch_y == 1.0, alpha_pos, alpha_neg)
                focal_weight = alpha * (1.0 - pt) ** gamma
                loss = (focal_weight * bce).mean()

                self.optimizer.zero_grad()
                loss.backward()
                if getattr(cfg.train, "grad_clip", 0) > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                self.optimizer.step()

                total_loss += loss.item()
                n_batches += 1
                self.global_step += 1

            avg_loss = total_loss / n_batches

            # 验证
            self.model.eval()
            with torch.no_grad():
                _, val_prob = self.model(X_val)
                val_prob_np = val_prob.cpu().numpy().flatten()
                val_y_np = y_val.cpu().numpy()

                _, train_prob = self.model(X_train)
                train_prob_np = train_prob.cpu().numpy().flatten()
                train_y_np = y_train.cpu().numpy()

            # Find best threshold on val
            from sklearn.metrics import precision_recall_curve
            precisions, recalls, thresholds = precision_recall_curve(val_y_np, val_prob_np)
            f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-10)
            best_idx = f1_scores.argmax()
            best_thr = thresholds[best_idx]
            val_f1 = f1_scores[best_idx]

            from sklearn.metrics import roc_auc_score, average_precision_score
            val_auc = roc_auc_score(val_y_np, val_prob_np)
            val_ap = average_precision_score(val_y_np, val_prob_np)
            train_auc = roc_auc_score(train_y_np, train_prob_np)

            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self.best_threshold = float(best_thr)
                self.patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_f1": self.best_val_f1,
                    "best_threshold": self.best_threshold,
                }, os.path.join(self.ckpt_dir, "best.pt"))
            else:
                self.patience_counter += 1

            if epoch % max(1, cfg.train.epochs // 10) == 0 or epoch == 1:
                self.logger.info(
                    f"Epoch {epoch}/{cfg.train.epochs} | "
                    f"loss={avg_loss:.4f} | "
                    f"train_auc={train_auc:.4f} | "
                    f"val_f1={val_f1:.4f} val_auc={val_auc:.4f} val_ap={val_ap:.4f}"
                )

            if cfg.train.patience > 0 and self.patience_counter >= cfg.train.patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

        train_time = time.time() - train_start
        self.logger.info(f"Training completed in {train_time:.1f}s")
        self.logger.info(f"Best val F1: {self.best_val_f1:.4f} at threshold {self.best_threshold:.3f}")

        # 测试
        self.model.eval()
        with torch.no_grad():
            _, test_prob = self.model(X_test)
            test_prob_np = test_prob.cpu().numpy().flatten()
            test_y_np = y_test.cpu().numpy()

        test_metrics = ClassificationMetrics(
            test_y_np, test_prob_np, self.best_threshold
        ).report()
        test_metrics["auc"] = roc_auc_score(test_y_np, test_prob_np)
        test_metrics["ap"] = average_precision_score(test_y_np, test_prob_np)

        self.logger.info(
            f"Test: ACC={test_metrics['acc']:.4f}, "
            f"F1={test_metrics['f1']:.4f}, "
            f"PREC={test_metrics['precision']:.4f}, "
            f"REC={test_metrics['recall']:.4f}, "
            f"AUC={test_metrics.get('auc', 0):.4f}, "
            f"AP={test_metrics.get('ap', 0):.4f}"
        )

        return {
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "val_metrics": {"f1": val_f1, "auc": val_auc, "ap": val_ap},
            "test_metrics": test_metrics,
            "train_time": train_time,
            "results_dir": self.results_dir,
        }

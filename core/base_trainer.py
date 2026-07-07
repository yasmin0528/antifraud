from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

from core.aml_dataset import (
    PREPROCESS_SCHEMA_VERSION,
    build_batch_graph,
    build_transaction_summary,
    get_dataloaders,
    preprocess_data,
)
from core.aml_dataset_amlsim import preprocess_amlsim_data
from models import CA1_TTPM, CA3_AGM, MPFC
from models.vta import compute_da_signal
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


class BaseTrainer:
    def __init__(self, cfg: Config, resume: bool = False):
        self.cfg = cfg
        self.device = self._resolve_device()
        self.resume = resume
        self.resume_ckpt = getattr(cfg.experiment, "resume_ckpt", None)

        exp_root = os.path.join(cfg.experiment.output_dir, cfg.data.dataset, cfg.experiment.name)
        if resume:
            if self.resume_ckpt:
                self.output_dir = os.path.dirname(os.path.dirname(self.resume_ckpt))
            else:
                self.output_dir = self._find_latest_run_dir(exp_root)
                if self.output_dir is None:
                    raise FileNotFoundError(f"No existing run directory found under {exp_root}.")
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

        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        self.metadata: Dict = {}
        self.account_sequences: Optional[torch.Tensor] = None
        self.account_seq_len: Optional[torch.Tensor] = None
        self.account_alert_idx: Optional[torch.Tensor] = None
        self.group_ids: Optional[torch.Tensor] = None
        self.memory_group_ids: Optional[torch.Tensor] = None
        self.group_membership: Dict[int, List[int]] = {}
        self._last_edge_trace: List[Dict[str, object]] = []

        self.models: Dict[str, nn.Module] = {}
        self.optimizer: Optional[optim.Optimizer] = None

        self.current_epoch = 0
        self.best_val_f1 = -1.0
        self.best_threshold = 0.5
        self.patience_counter = 0
        self.global_step = 0
        self.epoch_losses: List[float] = []
        self._ca3_update_memory = False
        self._save_config_snapshot()

    @staticmethod
    def _find_latest_run_dir(exp_root: str) -> Optional[str]:
        if not os.path.exists(exp_root):
            return None
        run_dirs = [d for d in os.listdir(exp_root) if d.startswith("run_") and os.path.isdir(os.path.join(exp_root, d))]
        if not run_dirs:
            return None
        run_dirs.sort(reverse=True)
        return os.path.join(exp_root, run_dirs[0])

    def _resolve_device(self) -> torch.device:
        return torch.device("cuda" if self.cfg.experiment.device == "auto" and torch.cuda.is_available() else self.cfg.experiment.device if self.cfg.experiment.device != "auto" else "cpu")

    def _log_config(self):
        self.logger.info("=" * 60)
        self.logger.info(f"Experiment: {self.cfg.experiment.name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Output dir: {self.output_dir}")
        self.logger.info(f"Dataset: {self.cfg.data.dataset}")
        self.logger.info(f"Split unit: {self.metadata.get('split_unit', 'unknown')}")
        self.logger.info(f"Edge features: {self.metadata.get('edge_feature_names', [])}")
        self.logger.info(f"Ablation: remove={self.cfg.ablation.remove_modules}")
        self.logger.info("=" * 60)

    def _save_config_snapshot(self):
        config_path = os.path.join(self.output_dir, "config.yaml")
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.cfg.to_dict(), f, allow_unicode=True, sort_keys=False)

    def _save_metadata(self):
        metadata = dict(self.metadata)
        metadata.update(
            {
                "dataset": self.cfg.data.dataset,
                "data_path": self.cfg.data.data_path,
                "preprocessed_path": self.cfg.data.preprocessed_path,
                "seed": self.cfg.experiment.seed,
                "schema_version": self.metadata.get("schema_version", PREPROCESS_SCHEMA_VERSION),
                "smote_requested": bool(self.cfg.data.use_smote),
                "group_type": self.metadata.get("group_type", "unknown"),
                "memory_mode": self.cfg.model.ca3.memory_mode,
                "eval_modes": {
                    "alert_metrics": self.cfg.eval.enable_alert_metrics,
                    "subgraph_metrics": self.cfg.eval.enable_subgraph_metrics,
                    "eval_da_mode": self.cfg.eval.eval_da_mode,
                    "vta_mode": self.cfg.vta.mode,
                },
            }
        )
        metadata_path = os.path.join(self.results_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _build_models(self):
        cfg = self.cfg
        remove_modules = set(cfg.ablation.remove_modules)

        if "ca1" not in remove_modules:
            self.models["ca1"] = CA1_TTPM(
                feature_dim=cfg.model.ca1.feature_dim or 3,
                hidden_dim=cfg.model.ca1.hidden_dim or cfg.model.hidden_dim,
                n_types=self.metadata.get("n_types", 0),
                type_emb_dim=cfg.model.dim_type,
            ).to(self.device)

        if "ca3" not in remove_modules:
            self.models["ca3"] = CA3_AGM(
                emb_dim=cfg.model.ca3.emb_dim or cfg.model.hidden_dim,
                num_groups=max(int(self.metadata.get("n_groups", 0)), 1),
                rpe_dim=1,
                memory_momentum=getattr(cfg.model.ca3, "memory_momentum", 0.9),
                memory_mode=getattr(cfg.model.ca3, "memory_mode", "explicit_group"),
                update_mode=getattr(cfg.model.ca3, "update_mode", "ema_group_proto"),
            ).to(self.device)

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
                edge_attr_dim=cfg.model.edge_attr_dim,
                hidden_dim=cfg.model.hidden_dim,
                num_gnn_layers=cfg.model.mpfc.gnn_layers,
                num_heads=cfg.model.mpfc.gnn_heads,
                dropout=cfg.model.dropout,
                llm_config=llm_config,
                use_llm=use_llm,
                input_dim=cfg.model.mpfc.input_dim or (cfg.model.hidden_dim + 1),
            ).to(self.device)
            mpfc.set_output_dir(self.output_dir)
            self.models["mpfc"] = mpfc
        else:
            self.models["classifier"] = nn.Linear(cfg.model.hidden_dim + 1, 1).to(self.device)

        params = []
        for model in self.models.values():
            params.extend(model.parameters())
        self.optimizer = optim.Adam(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    def _load_preprocessed_or_build(self) -> Dict:
        cfg = self.cfg
        dataset_type = cfg.data.dataset
        data_path = cfg.data.preprocessed_path

        if dataset_type == "amlsim_hi":
            builder = lambda: preprocess_amlsim_data(
                trans_csv_path=os.path.join(cfg.data.data_path, "HI-Small_Trans.csv"),
                patterns_path=os.path.join(cfg.data.data_path, "HI-Small_Patterns.txt"),
                window_size=cfg.data.window_size,
                save_path=data_path,
            )
        elif dataset_type == "amlsim_li":
            builder = lambda: preprocess_amlsim_data(
                trans_csv_path=os.path.join(cfg.data.data_path, "LI-Small_Trans.csv"),
                patterns_path=os.path.join(cfg.data.data_path, "LI-Small_Patterns.txt"),
                window_size=cfg.data.window_size,
                save_path=data_path,
            )
        else:
            builder = lambda: preprocess_data(
                csv_path=cfg.data.data_path,
                window_size=cfg.data.window_size,
                save_path=data_path,
            )

        if not os.path.exists(data_path) or cfg.data.regenerate:
            return builder()

        data = torch.load(data_path)
        required = {
            "schema_version",
            "split_unit",
            "group_ids",
            "edge_feature_names",
            "sample_group_ids",
            "split_group_ids",
            "sequences",
            "labels",
            "sender_idx",
            "receiver_idx",
            "alert_idx",
            "edge_attr",
            "edge_raw_amount",
            "account_sequences",
            "account_seq_len",
            "account_alert_idx",
        }
        if data.get("schema_version") != PREPROCESS_SCHEMA_VERSION or not required.issubset(data.keys()):
            self.logger.info("Preprocessed data outdated or incompatible, reprocessing...")
            return builder()
        return data

    def _build_dataloaders(self):
        cfg = self.cfg
        data = self._load_preprocessed_or_build()
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

        self.account_sequences = data["account_sequences"]
        self.account_seq_len = data["account_seq_len"]
        self.account_alert_idx = data["account_alert_idx"]
        self.group_ids = data.get("group_ids")
        self.memory_group_ids = data.get("memory_group_ids", self.group_ids)
        self.group_membership = data.get("group_membership", {})
        self.sender_idx = data["sender_idx"]
        self.receiver_idx = data["receiver_idx"]
        self.labels = data["labels"]
        self._save_metadata()

    def _encode_account_batch(self, account_ids: torch.Tensor, da_signal: Optional[torch.Tensor | float]) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_dim = self.cfg.model.hidden_dim
        if account_ids.numel() == 0:
            return (
                torch.zeros((0, hidden_dim), device=self.device),
                torch.zeros((0, 1), device=self.device),
            )

        seq_bank = self.account_sequences[account_ids.cpu()].to(self.device)
        seq_len_bank = self.account_seq_len[account_ids.cpu()].to(self.device)
        alert_bank = self.account_alert_idx[account_ids.cpu()].to(self.device)

        embeddings = torch.zeros((account_ids.size(0), hidden_dim), device=self.device)
        score_micro = torch.zeros((account_ids.size(0), 1), device=self.device)
        valid_mask = seq_len_bank > 0

        if "ca1" in self.models and valid_mask.any():
            valid_h, valid_score = self.models["ca1"](seq_bank[valid_mask], seq_len=seq_len_bank[valid_mask])
            embeddings[valid_mask] = valid_h
            score_micro[valid_mask] = valid_score

        if "ca3" in self.models:
            embeddings = self.models["ca3"](
                embeddings,
                group_ids=alert_bank,
                da_signal=da_signal,
                update_memory=self._ca3_update_memory,
            )

        return embeddings, score_micro

    def _aggregate_group_embeddings(
        self,
        embeddings: torch.Tensor,
        group_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if group_ids is None or embeddings.numel() == 0:
            return (
                torch.zeros((0, embeddings.size(-1)), device=embeddings.device, dtype=embeddings.dtype),
                torch.zeros((0,), device=embeddings.device, dtype=torch.long),
            )
        group_ids = group_ids.to(embeddings.device).long()
        valid_mask = (group_ids >= 0)
        if not valid_mask.any():
            return (
                torch.zeros((0, embeddings.size(-1)), device=embeddings.device, dtype=embeddings.dtype),
                torch.zeros((0,), device=embeddings.device, dtype=torch.long),
            )
        valid_groups = group_ids[valid_mask]
        valid_embeddings = embeddings[valid_mask]
        unique_groups = valid_groups.unique(sorted=True)
        group_repr = []
        for gid in unique_groups.tolist():
            member_mask = valid_groups == gid
            group_repr.append(valid_embeddings[member_mask].mean(dim=0))
        return torch.stack(group_repr, dim=0), unique_groups

    def _forward_samples(
        self,
        seq: torch.Tensor,
        sender: torch.Tensor,
        receiver: torch.Tensor,
        edge_e: torch.Tensor,
        alert: torch.Tensor,
        transaction_summary: Optional[str],
        da_signal: Optional[torch.Tensor | float],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, object]]]:
        has_ca1 = "ca1" in self.models
        has_mpfc = "mpfc" in self.models

        if has_ca1:
            fallback_h, fallback_score = self.models["ca1"](seq)
        else:
            fallback_h = torch.zeros(seq.size(0), self.cfg.model.hidden_dim, device=self.device)
            fallback_score = torch.zeros(seq.size(0), 1, device=self.device)

        valid_accounts = torch.unique(torch.cat([sender[sender >= 0], receiver[receiver >= 0]]))
        account_h, account_score = self._encode_account_batch(valid_accounts, da_signal=da_signal)

        node_x, edge_index, edge_attr_batch, sender_local = build_batch_graph(
            sender,
            receiver,
            edge_e,
            valid_accounts,
            account_h,
            account_score,
            fallback_features=fallback_h,
            fallback_scores=fallback_score,
        )

        if has_mpfc:
            _, logit, prob, edge_trace = self.models["mpfc"](
                node_x,
                edge_index,
                edge_attr_batch,
                transaction_summary=transaction_summary,
                da_signal=da_signal,
            )
        else:
            logit = self.models["classifier"](node_x)
            prob = torch.sigmoid(logit)
            edge_trace = []

        return logit[sender_local], prob[sender_local].squeeze(-1), edge_trace

    def _train_epoch(self) -> Tuple[float, Dict]:
        cfg = self.cfg
        for m in self.models.values():
            m.train()
        self._ca3_update_memory = True

        epoch_loss = 0.0
        tracker = MetricTracker()
        prev_da_signal = None
        transaction_summary = getattr(self, "_transaction_summary", None)
        if transaction_summary is None and "mpfc" in self.models:
            try:
                first_batch = next(iter(self.train_loader))
                _, _, _, edge_e_first, _, label_first = first_batch
                transaction_summary = build_transaction_summary(edge_e_first, label_first)
                self._transaction_summary = transaction_summary
            except (StopIteration, Exception):
                transaction_summary = None

        for batch_idx, batch in enumerate(self.train_loader, start=1):
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]
            self.optimizer.zero_grad()
            pred_logit, pred_prob, edge_trace = self._forward_samples(
                seq, sender, receiver, edge_e, alert, transaction_summary, prev_da_signal
            )
            self._last_edge_trace = edge_trace

            gamma = getattr(cfg.train, "focal_gamma", 0.0)
            if gamma > 0:
                pos_weight = getattr(cfg.train, "pos_weight", 1.0)
                alpha_pos = pos_weight / (pos_weight + 1.0)
                alpha_neg = 1.0 / (pos_weight + 1.0)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    pred_logit, label.float().view(-1, 1), reduction="none"
                )
                pred_prob_2d = torch.sigmoid(pred_logit)
                pt = torch.where(label.float().view(-1, 1) == 1.0, pred_prob_2d, 1.0 - pred_prob_2d)
                alpha = torch.where(label.float().view(-1, 1) == 1.0, alpha_pos, alpha_neg)
                loss = (alpha * (1.0 - pt) ** gamma * bce).mean()
            else:
                loss = nn.BCEWithLogitsLoss(
                    pos_weight=torch.tensor(cfg.train.pos_weight, device=self.device)
                )(pred_logit, label.float().view(-1, 1))

            if "ca3" in self.models:
                valid_accounts = torch.unique(torch.cat([sender[sender >= 0], receiver[receiver >= 0]]))
                if valid_accounts.numel() > 0:
                    account_h, _ = self._encode_account_batch(valid_accounts, da_signal=prev_da_signal)
                    account_group_ids = self.account_alert_idx[valid_accounts.cpu()].to(self.device)
                    group_repr, group_repr_ids = self._aggregate_group_embeddings(account_h, account_group_ids)
                    self.models["ca3"].update_group_memory_from_aggregates(group_repr.detach(), group_repr_ids.detach())
                    memory_loss = self.models["ca3"].memory_loss(group_repr, group_repr_ids)
                    loss = loss + 0.1 * memory_loss

            loss.backward()
            if cfg.train.grad_clip > 0:
                for m in self.models.values():
                    torch.nn.utils.clip_grad_norm_(m.parameters(), cfg.train.grad_clip)
            self.optimizer.step()
            self.global_step += 1

            epoch_loss += loss.item()
            tracker.update(pred_prob.detach().cpu(), label.cpu())

            if "vta" not in cfg.ablation.remove_modules:
                with torch.no_grad():
                    da_current = compute_da_signal(
                        prob=pred_prob.unsqueeze(-1),
                        y=label,
                        rpe_beta=cfg.train.rpe_beta,
                        momentum=0.0,
                        prev_da=None,
                    )
                    prev_da_signal = da_current.detach()
            else:
                prev_da_signal = None

            if cfg.train.log_interval > 0 and batch_idx % cfg.train.log_interval == 0:
                self.logger.info(
                    f"Epoch {self.current_epoch}/{cfg.train.epochs} "
                    f"batch {batch_idx}/{len(self.train_loader)} "
                    f"loss {loss.item():.4f} avg_da={(float(prev_da_signal.mean().item()) if prev_da_signal is not None else 1.0):.3f}"
                )

        avg_loss = epoch_loss / len(self.train_loader)
        train_metrics = tracker.compute()
        self.epoch_losses.append(avg_loss)
        return avg_loss, train_metrics

    @torch.no_grad()
    def _evaluate(self, loader: torch.utils.data.DataLoader, threshold: float = 0.5) -> Dict:
        y_true, y_prob, _ = self._collect_predictions(loader)
        metrics = ClassificationMetrics(y_true, y_prob, threshold).report()
        if self.cfg.eval.enable_alert_metrics and self.group_ids is not None:
            dataset = getattr(loader, "dataset", None)
            sample_indices = getattr(dataset, "indices", None)
            if sample_indices is None:
                sample_indices = list(range(len(y_true)))
            group_ids = self.group_ids[np.array(sample_indices, dtype=np.int64)].cpu().numpy()
            metrics.update(
                compute_alert_level_metrics(
                    group_ids=group_ids,
                    y_true=y_true,
                    y_prob=y_prob,
                    threshold=threshold,
                    agg=self.cfg.eval.alert_agg,
                )
            )
            if self.cfg.eval.enable_subgraph_metrics:
                metrics["hit_at_k"] = compute_hit_at_k(
                    group_ids=group_ids,
                    y_true=y_true,
                    y_prob=y_prob,
                    k=self.cfg.eval.hit_k,
                    agg=self.cfg.eval.alert_agg,
                )
                metrics.update(
                    compute_subgraph_coverage(
                        true_group_ids=group_ids,
                        pred_scores=y_prob,
                        top_k=self.cfg.eval.hit_k,
                    )
                )
        return metrics

    @torch.no_grad()
    def _collect_predictions(self, loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, object]]]:
        for m in self.models.values():
            m.eval()
        self._ca3_update_memory = False
        transaction_summary = getattr(self, "_transaction_summary", None)
        all_probs = []
        all_labels = []
        all_edge_traces: List[Dict[str, object]] = []
        for batch in loader:
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]
            _, pred_prob, edge_trace = self._forward_samples(
                seq, sender, receiver, edge_e, alert, transaction_summary, 1.0
            )
            all_probs.append(pred_prob.cpu())
            all_labels.append(label.cpu())
            all_edge_traces.extend(edge_trace)
        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()
        self._last_edge_trace = all_edge_traces
        return y_true, y_prob, all_edge_traces

    @torch.no_grad()
    def _search_best_threshold(self, loader: torch.utils.data.DataLoader) -> Tuple[float, Dict]:
        for m in self.models.values():
            m.eval()
        self._ca3_update_memory = False
        transaction_summary = getattr(self, "_transaction_summary", None)
        all_probs = []
        all_labels = []
        for batch in loader:
            seq, sender, receiver, edge_e, alert, label = [b.to(self.device) for b in batch]
            _, pred_prob, _ = self._forward_samples(
                seq, sender, receiver, edge_e, alert, transaction_summary, 1.0
            )
            all_probs.append(pred_prob.cpu())
            all_labels.append(label.cpu())
        y_true = torch.cat(all_labels).numpy()
        y_prob = torch.cat(all_probs).numpy()

        thresholds = np.linspace(0.05, 0.95, 91)
        best_threshold = 0.5
        best_metrics = {}
        best_f1 = -1.0
        for thr in thresholds:
            metrics = ClassificationMetrics(y_true, y_prob, thr).report()
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_threshold = thr
                best_metrics = metrics
        return best_threshold, best_metrics

    def train(self) -> Dict:
        cfg = self.cfg
        set_seed(cfg.experiment.seed)
        self._build_dataloaders()
        self._log_config()
        self._build_models()

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
                for name, state_dict in ckpt["model_state_dict"].items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)
                if "optimizer_state_dict" in ckpt and self.optimizer is not None:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_epoch = ckpt.get("epoch", 0) + 1
                self.global_step = ckpt.get("global_step", 0)
                self.best_val_f1 = ckpt.get("best_val_f1", -1.0)
                self.best_threshold = ckpt.get("best_threshold", 0.5)
                self.patience_counter = ckpt.get("patience_counter", 0)

        train_start = time.time()
        val_metrics = {}
        train_metrics = {}
        for epoch in range(start_epoch, cfg.train.epochs + 1):
            self.current_epoch = epoch
            avg_loss, train_metrics = self._train_epoch()
            if self.val_loader is not None:
                val_threshold, val_metrics = self._search_best_threshold(self.val_loader)
                if val_metrics["f1"] > self.best_val_f1:
                    self.best_val_f1 = val_metrics["f1"]
                    self.best_threshold = val_threshold
                    self.patience_counter = 0
                    self.ckpt_manager.save_best(
                        {
                            "model_state_dict": {name: model.state_dict() for name, model in self.models.items()},
                            "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                            "epoch": epoch,
                            "global_step": self.global_step,
                            "best_val_f1": self.best_val_f1,
                            "best_threshold": self.best_threshold,
                            "patience_counter": self.patience_counter,
                        },
                        val_metrics["f1"],
                    )
                else:
                    self.patience_counter += 1
            self.ckpt_manager.save(
                {
                    "model_state_dict": {name: model.state_dict() for name, model in self.models.items()},
                    "optimizer_state_dict": self.optimizer.state_dict() if self.optimizer else None,
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "best_val_f1": self.best_val_f1,
                    "best_threshold": self.best_threshold,
                    "patience_counter": self.patience_counter,
                },
                epoch=epoch,
                step=self.global_step,
            )
            self.logger.info(
                f"Epoch {epoch}/{cfg.train.epochs} | train_f1={train_metrics['f1']:.4f} "
                f"| val_f1={val_metrics.get('f1', 0.0):.4f} | val_thr={self.best_threshold:.2f}"
            )
            self.tb_logger.log_scalar("train/loss", avg_loss, epoch)
            self.tb_logger.log_scalar("train/f1", train_metrics["f1"], epoch)
            if val_metrics:
                self.tb_logger.log_scalar("val/f1", val_metrics["f1"], epoch)
            if cfg.train.patience > 0 and self.patience_counter >= cfg.train.patience:
                break

        train_time = time.time() - train_start
        test_metrics = {}
        if self.test_loader is not None:
            best_ckpt = self.ckpt_manager.load_best(self.device)
            if best_ckpt:
                for name, state_dict in best_ckpt["model_state_dict"].items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)
            test_metrics = self._evaluate(self.test_loader, threshold=self.best_threshold)

        if cfg.visualization.enabled and cfg.visualization.save_figures and self.epoch_losses:
            try:
                self.visualizer.plot_loss_curve(self.epoch_losses, name="loss_curve")
            except Exception as e:
                self.logger.warning(f"Loss visualization failed: {e}")

        self._save_sample_scores(self.test_loader)
        self.tb_logger.close()
        self._save_ca3_artifacts()
        return {
            "best_val_f1": self.best_val_f1,
            "best_threshold": self.best_threshold,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "test_metrics": test_metrics,
            "train_time": train_time,
            "results_dir": self.results_dir,
            "dataset": self.cfg.data.dataset,
            "smote_applied": self.metadata.get("smote_applied", False),
            "train_size": self.metadata.get("train_size", 0),
            "val_size": self.metadata.get("val_size", 0),
            "test_size": self.metadata.get("test_size", 0),
        }

    def test(self, checkpoint_path: Optional[str] = None) -> Dict:
        self._build_dataloaders()
        self._build_models()
        if checkpoint_path:
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dicts = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            for name, state_dict in state_dicts.items():
                if name in self.models:
                    self.models[name].load_state_dict(state_dict)
        else:
            best_ckpt = self.ckpt_manager.load_best(self.device)
            if best_ckpt:
                for name, state_dict in best_ckpt["model_state_dict"].items():
                    if name in self.models:
                        self.models[name].load_state_dict(state_dict)
        metrics = self._evaluate(self.test_loader, threshold=self.best_threshold)
        self._save_sample_scores(self.test_loader)
        self.tb_logger.close()
        return metrics

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

    def _save_sample_scores(self, loader: Optional[torch.utils.data.DataLoader], filename: str = "sample_scores.csv"):
        if loader is None:
            return
        dataset = getattr(loader, "dataset", None)
        sample_indices = getattr(dataset, "indices", None)
        if sample_indices is None:
            sample_indices = list(range(len(dataset)))
        y_true, y_prob, edge_traces = self._collect_predictions(loader)

        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sample_idx", "sender_idx", "receiver_idx", "group_id", "label", "risk_score", "attention", "rule_text", "rule_type", "rule_confidence"])
            for pos, sample_idx in enumerate(sample_indices):
                group_id = -1
                if self.group_ids is not None and int(sample_idx) < len(self.group_ids):
                    group_id = int(self.group_ids[int(sample_idx)])
                trace = edge_traces[pos] if pos < len(edge_traces) else {}
                writer.writerow([
                    int(sample_idx),
                    int(self.sender_idx[int(sample_idx)]),
                    int(self.receiver_idx[int(sample_idx)]),
                    group_id,
                    int(y_true[pos]),
                    float(y_prob[pos]),
                    float(trace.get("attention", 0.0)),
                    str(trace.get("matched_rule_text", "")),
                    str(trace.get("matched_rule_type", "")),
                    float(trace.get("rule_confidence", 0.0)),
                ])
        self._save_edge_trace(edge_traces)

    def _save_edge_trace(self, edge_traces: List[Dict[str, object]], filename: str = "edge_trace.csv"):
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["edge_id", "src", "dst", "attention", "rule_bias", "rule_match_score", "matched_rule_text", "matched_rule_type", "rule_confidence"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for trace in edge_traces:
                writer.writerow({key: trace.get(key, "") for key in fieldnames})

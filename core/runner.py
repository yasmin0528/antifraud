"""
Experiment orchestration for baseline, ablation, sweep, VTA decomposition,
and multi-seed runs.
"""

from __future__ import annotations

import copy
import csv
import itertools
import os
from typing import Any, Dict, List, Optional

import numpy as np

from core.base_trainer import BaseTrainer
from core.graph_trainer import GraphNodeTrainer
from utils import Config, Logger, merge_config, set_seed


class ExperimentRunner:
    def __init__(self, cfg: Config, config_path: Optional[str] = None):
        self.cfg = cfg
        self.config_path = config_path
        self.logger = Logger(
            name="ExperimentRunner",
            log_dir=os.path.join(cfg.experiment.output_dir, cfg.data.dataset, cfg.experiment.name),
            console=True,
            log_file=False,
        )
        self.results: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        cfg = self.cfg

        if cfg.multi_seed.enabled:
            self._run_multi_seed()
        elif cfg.sweep.enabled:
            self._run_sweep()
        elif cfg.vta_decomp.enabled:
            self._run_vta_decomp()
        elif cfg.ablation.enabled:
            self._run_ablation()
        else:
            self.results.append(self._run_single(cfg))

        self._save_results()
        return self.results

    def _normalize_dataset(self, cfg: Config) -> str:
        dataset_name = "cryptopia" if cfg.data.dataset == "cryptopia_graph" else cfg.data.dataset
        cfg.data.dataset = dataset_name
        return dataset_name

    def _make_trainer(self, cfg: Config):
        dataset_name = self._normalize_dataset(cfg)
        if dataset_name == "cryptopia":
            return GraphNodeTrainer(cfg), dataset_name
        return BaseTrainer(cfg), dataset_name

    def _run_single(self, cfg: Config) -> Dict[str, Any]:
        set_seed(cfg.experiment.seed)
        trainer, dataset_name = self._make_trainer(cfg)

        try:
            result = trainer.train()
        except Exception as e:
            trainer.logger.error(f"Training failed: {e}")
            trainer.logger.exception("Full traceback:")
            raise

        result["config_name"] = cfg.experiment.name
        result["seed"] = cfg.experiment.seed
        result["model"] = cfg.model.name
        result["dataset"] = dataset_name
        return result

    def _run_ablation(self):
        cfg = self.cfg
        remove_modules = cfg.ablation.remove_modules

        if remove_modules:
            result = self._run_single(cfg)
            result["variant"] = f"wo_{'_'.join(remove_modules)}"
            self.results.append(result)
            return

        variants = cfg.ablation.variants or ["wo_ca1", "wo_ca3", "wo_mpfc", "wo_vta", "wo_llm"]
        variant_map = {
            "wo_ca1": {"ablation": {"remove_modules": ["ca1"]}},
            "wo_ca3": {"ablation": {"remove_modules": ["ca3"]}},
            "wo_mpfc": {"ablation": {"remove_modules": ["mpfc"]}},
            "wo_vta": {"ablation": {"remove_modules": ["vta"]}},
            "wo_llm": {"ablation": {"remove_modules": ["wo_llm"]}},
        }

        for variant in variants:
            if variant not in variant_map:
                self.logger.warning(f"Unknown ablation variant: {variant}")
                continue
            sub_cfg = copy.deepcopy(cfg)
            sub_cfg = merge_config(sub_cfg, variant_map[variant])
            sub_cfg.experiment.name = f"{cfg.experiment.name}_{variant}"
            result = self._run_single(sub_cfg)
            result["variant"] = variant
            self.results.append(result)

    def _run_sweep(self):
        param_grid = {
            key: values
            for key, values in self.cfg.sweep.params.items()
            if values is not None and len(values) > 0
        }
        if not param_grid:
            self.logger.warning("No sweep parameters specified.")
            return

        keys = list(param_grid.keys())
        combinations = list(itertools.product(*param_grid.values()))
        self.logger.info(f"Grid search: {len(combinations)} combinations")

        for combo in combinations:
            params = dict(zip(keys, combo))
            override: Dict[str, Dict[str, Any]] = {}
            for key, value in params.items():
                if key == "batch_size":
                    override.setdefault("data", {})["batch_size"] = value
                elif key in ("lr", "pos_weight", "focal_gamma", "rpe_beta"):
                    override.setdefault("train", {})[key] = value
                elif key == "hidden_dim":
                    override.setdefault("model", {})["hidden_dim"] = value
                    override["model"]["ca1"] = {"hidden_dim": value}
                    override["model"]["ca3"] = {"emb_dim": value}
                elif key == "dropout":
                    override.setdefault("model", {})["dropout"] = value
                elif key == "gnn_layers":
                    override.setdefault("model", {}).setdefault("mpfc", {})["gnn_layers"] = value
                elif key == "gnn_heads":
                    override.setdefault("model", {}).setdefault("mpfc", {})["gnn_heads"] = value
                elif key == "memory_momentum":
                    override.setdefault("model", {}).setdefault("ca3", {})["memory_momentum"] = value

            sub_cfg = copy.deepcopy(self.cfg)
            sub_cfg = merge_config(sub_cfg, override)
            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            sub_cfg.experiment.name = f"{self.cfg.experiment.name}_sweep_{param_str}"
            result = self._run_single(sub_cfg)
            result["params"] = params
            self.results.append(result)

    def _run_vta_decomp(self):
        for variant in self.cfg.vta_decomp.variants:
            sub_cfg = copy.deepcopy(self.cfg)
            sub_cfg = merge_config(
                sub_cfg,
                {
                    "train": {
                        "focal_gamma": variant["focal_gamma"],
                        "rpe_beta": variant["rpe_beta"],
                        "pos_weight": variant["pos_weight"],
                    }
                },
            )
            sub_cfg.experiment.name = f"{self.cfg.experiment.name}_{variant['name']}"
            result = self._run_single(sub_cfg)
            result["variant"] = variant["name"]
            self.results.append(result)

    def _run_multi_seed(self):
        for seed in self.cfg.multi_seed.seeds:
            sub_cfg = copy.deepcopy(self.cfg)
            sub_cfg.experiment.seed = seed
            sub_cfg.experiment.name = f"{self.cfg.experiment.name}_seed{seed}"
            result = self._run_single(sub_cfg)
            result["seed"] = seed
            self.results.append(result)

        if len(self.results) > 1:
            f1_values = [r.get("test_metrics", {}).get("f1", r.get("val_metrics", {}).get("f1", 0.0)) for r in self.results]
            auc_values = [r.get("test_metrics", {}).get("auc", r.get("val_metrics", {}).get("auc", 0.0)) for r in self.results]
            self.logger.info(
                f"Multi-seed results: F1 mean={np.mean(f1_values):.4f} std={np.std(f1_values):.4f}, "
                f"AUC mean={np.mean(auc_values):.4f} std={np.std(auc_values):.4f}"
            )

    def _result_row(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dataset": result.get("dataset", ""),
            "config_name": result.get("config_name", ""),
            "model": result.get("model", ""),
            "seed": result.get("seed", ""),
            "variant": result.get("variant", ""),
            "best_threshold": result.get("best_threshold", ""),
            "auc": result.get("test_metrics", {}).get("auc", result.get("val_metrics", {}).get("auc", "")),
            "f1": result.get("test_metrics", {}).get("f1", result.get("val_metrics", {}).get("f1", "")),
            "ap": result.get("test_metrics", {}).get("ap", result.get("val_metrics", {}).get("ap", "")),
            "alert_level_ap": result.get("test_metrics", {}).get("alert_level_ap", result.get("val_metrics", {}).get("alert_level_ap", "")),
            "alert_level_f1": result.get("test_metrics", {}).get("alert_level_f1", result.get("val_metrics", {}).get("alert_level_f1", "")),
            "hit_at_k": result.get("test_metrics", {}).get("hit_at_k", result.get("val_metrics", {}).get("hit_at_k", "")),
            "subgraph_coverage_node": result.get("test_metrics", {}).get("subgraph_coverage_node", result.get("val_metrics", {}).get("subgraph_coverage_node", "")),
            "subgraph_coverage_edge": result.get("test_metrics", {}).get("subgraph_coverage_edge", result.get("val_metrics", {}).get("subgraph_coverage_edge", "")),
            "train_time": result.get("train_time", ""),
            "train_size": result.get("train_size", ""),
            "val_size": result.get("val_size", ""),
            "test_size": result.get("test_size", ""),
            "smote_applied": result.get("smote_applied", False),
        }

    def _save_results(self):
        if not self.results:
            return

        for result in self.results:
            results_dir = result.get("results_dir")
            if not results_dir:
                exp_root = os.path.join(
                    self.cfg.experiment.output_dir,
                    result.get("dataset", self.cfg.data.dataset),
                    result.get("config_name", self.cfg.experiment.name),
                )
                results_dir = os.path.join(exp_root, "results")
            os.makedirs(results_dir, exist_ok=True)
            row = self._result_row(result)
            csv_path = os.path.join(results_dir, "results.csv")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=row.keys())
                writer.writeheader()
                writer.writerow(row)

        exp_root = os.path.join(
            self.cfg.experiment.output_dir,
            self.cfg.data.dataset,
            self.cfg.experiment.name,
        )
        os.makedirs(exp_root, exist_ok=True)
        all_rows = [self._result_row(result) for result in self.results]
        summary_path = os.path.join(exp_root, "results_summary.csv")
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        self.logger.info(f"Results summary saved to {summary_path}")

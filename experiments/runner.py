"""
实验运行器 —— 支持多种实验模式。

实验类型：
1. main     - 基础训练实验
2. ablation - 消融实验（移除模块对比）
3. sweep    - 参数敏感性扫描（grid search）
4. multi_seed - 多随机种子重复实验
5. test     - 推理测试
"""

from __future__ import annotations

import copy
import csv
import itertools
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch

from trainers.base_trainer import BaseTrainer
from utils import Config, Logger, load_config, merge_config, set_seed


class ExperimentRunner:
    """
    实验运行器。

    用法：
        runner = ExperimentRunner(cfg)
        runner.run()
    """

    def __init__(self, cfg: Config, config_path: Optional[str] = None):
        self.cfg = cfg
        self.config_path = config_path
        self.logger = Logger(
            name="ExperimentRunner",
            log_dir=os.path.join(cfg.experiment.output_dir, cfg.experiment.name),
            console=True,
        )
        self.results: List[Dict[str, Any]] = []

    def run(self) -> List[Dict[str, Any]]:
        """
        根据配置执行相应实验类型。

        Returns:
            results: 所有实验结果列表
        """
        cfg = self.cfg

        if cfg.multi_seed.enabled:
            self.logger.info("=" * 60)
            self.logger.info(f"Multi-seed experiment: seeds={cfg.multi_seed.seeds}")
            self.logger.info("=" * 60)
            self._run_multi_seed()

        elif cfg.sweep.enabled:
            self.logger.info("=" * 60)
            self.logger.info(f"Parameter sweep: method={cfg.sweep.method}")
            self.logger.info(f"Params={cfg.sweep.params}")
            self.logger.info("=" * 60)
            self._run_sweep()

        elif cfg.ablation.enabled:
            self.logger.info("=" * 60)
            self.logger.info(f"Ablation study: remove={cfg.ablation.remove_modules}")
            if cfg.ablation.variants:
                self.logger.info(f"Variants={cfg.ablation.variants}")
            self.logger.info("=" * 60)
            self._run_ablation()

        else:
            self.logger.info("=" * 60)
            self.logger.info("Main experiment")
            self.logger.info("=" * 60)
            result = self._run_single(cfg)
            self.results.append(result)

        # 保存结果
        self._save_results()

        return self.results

    def _run_single(self, cfg: Config) -> Dict[str, Any]:
        """运行单个实验并返回结果。"""
        set_seed(cfg.experiment.seed)
        trainer = BaseTrainer(cfg)
        result = trainer.train()
        result["config_name"] = cfg.experiment.name
        result["seed"] = cfg.experiment.seed
        result["model"] = cfg.model.name
        return result

    def _run_ablation(self):
        """运行消融实验。"""
        variants = self.cfg.ablation.variants or ["wo_ca1", "wo_ca3", "wo_mpfc", "wo_vta", "wo_llm"]

        variant_map = {
            "wo_ca1": {"ablation": {"remove_modules": ["ca1"]}},
            "wo_ca3": {"ablation": {"remove_modules": ["ca3"]}},
            "wo_mpfc": {"ablation": {"remove_modules": ["mpfc"]}},
            "wo_vta": {"ablation": {"remove_modules": ["vta"]}},
            "wo_llm": {"ablation": {"remove_modules": ["wo_llm"]}},  # 保留 MPFC 但去除 LLM
        }

        for variant in variants:
            if variant not in variant_map:
                self.logger.warning(f"Unknown ablation variant: {variant}")
                continue

            self.logger.info("-" * 40)
            self.logger.info(f"Ablation: {variant}")
            self.logger.info("-" * 40)

            cfg = copy.deepcopy(self.cfg)
            cfg = merge_config(cfg, variant_map[variant])
            cfg.experiment.name = f"{self.cfg.experiment.name}_{variant}"

            result = self._run_single(cfg)
            result["variant"] = variant
            self.results.append(result)

    def _run_sweep(self):
        """运行参数敏感性扫描（网格搜索）。"""
        sweep_params = self.cfg.sweep.params
        method = self.cfg.sweep.method

        # 收集非空参数
        param_grid: Dict[str, List] = {}
        for key, values in sweep_params.items():
            if values is not None and len(values) > 0:
                param_grid[key] = values

        if not param_grid:
            self.logger.warning("No sweep parameters specified.")
            return

        # 生成参数组合
        if method == "grid":
            keys = list(param_grid.keys())
            values_list = list(param_grid.values())
            combinations = list(itertools.product(*values_list))
            self.logger.info(f"Grid search: {len(combinations)} combinations")
        else:
            # random: 使用前几个参数的组合
            keys = list(param_grid.keys())
            max_combos = max(len(v) for v in param_grid.values())
            combinations = []
            for i in range(max_combos):
                combo = []
                for k in keys:
                    vals = param_grid[k]
                    combo.append(vals[i % len(vals)])
                combinations.append(combo)
            self.logger.info(f"Random search: {len(combinations)} combinations")

        for combo in combinations:
            params = dict(zip(keys, combo))
            self.logger.info(f"  Sweeping: {params}")

            # 构建覆盖配置
            override = {}
            data_override = {}
            model_override = {}
            train_override = {}

            for key, value in params.items():
                if key == "batch_size":
                    data_override["batch_size"] = value
                elif key == "hidden_dim":
                    model_override["hidden_dim"] = value
                    model_override["ca1"] = {"hidden_dim": value}
                    model_override["ca3"] = {"emb_dim": value}
                elif key == "dropout":
                    model_override["dropout"] = value
                elif key in ("pos_weight", "focal_gamma", "gnn_layers", "gnn_heads"):
                    if key in ("pos_weight", "focal_gamma"):
                        train_override[key] = value
                    elif key == "gnn_layers":
                        model_override["mpfc"] = {"gnn_layers": value}
                    elif key == "gnn_heads":
                        model_override["mpfc"] = {"gnn_heads": value}
                else:
                    train_override[key] = value

            if data_override:
                override["data"] = data_override
            if model_override:
                override["model"] = model_override
            if train_override:
                override["train"] = train_override

            cfg = copy.deepcopy(self.cfg)
            cfg = merge_config(cfg, override)

            param_str = "_".join(f"{k}{v}" for k, v in params.items())
            cfg.experiment.name = f"{self.cfg.experiment.name}_sweep_{param_str}"

            result = self._run_single(cfg)
            result["params"] = params
            self.results.append(result)

    def _run_multi_seed(self):
        """运行多随机种子实验。"""
        seeds = self.cfg.multi_seed.seeds

        for i, seed in enumerate(seeds):
            self.logger.info(f"Multi-seed [{i + 1}/{len(seeds)}]: seed={seed}")

            cfg = copy.deepcopy(self.cfg)
            cfg.experiment.seed = seed
            cfg.experiment.name = f"{self.cfg.experiment.name}_seed{seed}"

            result = self._run_single(cfg)
            result["seed"] = seed
            self.results.append(result)

        # 计算统计
        if len(self.results) > 1:
            f1_values = [r.get("best_val_f1", 0) for r in self.results]
            auc_values = [
                r.get("val_metrics", {}).get("auc", 0) for r in self.results
            ]
            self.logger.info(
                f"Multi-seed results: F1 mean={np.mean(f1_values):.4f} "
                f"std={np.std(f1_values):.4f}, "
                f"AUC mean={np.mean(auc_values):.4f} "
                f"std={np.std(auc_values):.4f}"
            )

    def _save_results(self):
        """将实验结果保存为 CSV。"""
        if not self.results:
            return

        output_dir = os.path.join(
            self.cfg.experiment.output_dir, self.cfg.experiment.name
        )
        os.makedirs(output_dir, exist_ok=True)

        # 展平结果
        rows = []
        for result in self.results:
            row = {
                "config_name": result.get("config_name", ""),
                "model": result.get("model", ""),
                "seed": result.get("seed", ""),
                "variant": result.get("variant", ""),
                "params": str(result.get("params", "")),
                "best_val_f1": result.get("best_val_f1", ""),
                "best_threshold": result.get("best_threshold", ""),
                "train_time": result.get("train_time", ""),
            }

            # 展平指标
            for prefix in ("train_metrics", "val_metrics", "test_metrics"):
                metrics = result.get(prefix, {})
                for k, v in metrics.items():
                    row[f"{prefix}_{k}"] = v

            rows.append(row)

        csv_path = os.path.join(output_dir, "results.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        self.logger.info(f"Results saved to {csv_path}")


import numpy as np  # noqa: E402 (needed for multi-seed stats)

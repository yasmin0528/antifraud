#!/usr/bin/env python3
"""
run_experiments.py
AML 反洗钱模型 —— 全面实验调度与结果汇总脚本。

支持五种实验类型：
  1. baseline    - 基础实验（完整 MPFC 模型）
  2. ablation    - 消融实验（逐一移除 CA1/CA3/MPFC/VTA/LLM 模块）
  3. sensitivity  - 参数敏感性实验（learning_rate/focal_gamma/rpe_beta/memory_momentum）
  4. vta_decomp   - VTA 损失解耦实验（Focal/RPE/PW 逐组件清零验证）
  5. multi_seed  - 多随机种子实验（5 个种子评估稳定性）

用法：
    # 运行全部实验
    python scripts/run_experiments.py --llm_api_url http://localhost:11434/v1/chat/completions

    # 仅运行消融实验
    python scripts/run_experiments.py --only ablation --llm_api_url <URL>

    # 仅运行 VTA 损失解耦实验（无需 LLM，跑得最快）
    python scripts/run_experiments.py --only vta_decomp

    # 指定数据路径
    python scripts/run_experiments.py --data_path /path/to/dataset.csv

    # 汇总已有结果（不重新运行）
    python scripts/run_experiments.py --summarize-only

    # 输出汇总表格到 CSV
    python scripts/run_experiments.py --export results_summary.csv
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================
# 配置
# ============================================================
BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_SCRIPT = BASE_DIR / "train.py"
CONFIG_DIR = BASE_DIR / "configs"
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
ABLATION_DIR = CONFIG_DIR / "ablation"
SENSITIVITY_DIR = CONFIG_DIR / "sensitivity"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs"

ABLATION_VARIANTS = [
    "wo_ca1",
    "wo_ca3",
    "wo_mpfc",
    "wo_vta",
    "wo_llm",
]

SENSITIVITY_VARIANTS = [
    "learning_rate",
    "focal_gamma",
    "rpe_beta",
    "memory_momentum",
]

MULTI_SEEDS = [42, 123, 456, 789, 1111]


class ExperimentManager:
    """实验调度管理器。"""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.llm_api_url = args.llm_api_url
        self.output_dir = args.output_dir or str(DEFAULT_OUTPUT_DIR)
        self.data_path = args.data_path
        self.preprocessed_path = args.preprocessed_path
        self.dry_run = args.dry_run
        self.results: Dict[str, List[Dict]] = defaultdict(list)

    def _base_args(self) -> List[str]:
        """构建所有实验共用的基础命令行参数。"""
        args_list = [
            str(TRAIN_SCRIPT),
            "--config", str(DEFAULT_CONFIG),
        ]
        if self.llm_api_url:
            args_list.extend(["--llm_api_url", self.llm_api_url])
        if self.output_dir:
            args_list.extend(["--output_dir", self.output_dir])
        if self.data_path:
            args_list.extend(["--data_path", self.data_path])
        if self.preprocessed_path:
            args_list.extend(["--preprocessed_path", self.preprocessed_path])
        return args_list

    def run_cmd(self, description: str, cmd_args: List[str]):
        """执行一条实验命令。"""
        print()
        print("=" * 70)
        print(f"  [{datetime.now():%Y-%m-%d %H:%M:%S}] {description}")
        print(f"  Command: {' '.join(cmd_args)}")
        print("=" * 70)

        if self.dry_run:
            print("  (DRY RUN - skipped)")
            return

        result = subprocess.run(cmd_args, cwd=str(BASE_DIR))
        if result.returncode != 0:
            print(f"  [WARNING] Command exited with code {result.returncode}, continuing...")

    # -------- 基础实验 --------
    def run_baseline(self):
        print()
        print("█" * 70)
        print("  基础实验 (Baseline) - 完整 MPFC 模型")
        print("█" * 70)

        args = self._base_args() + [
            "--name", "baseline",
            "--epochs", str(self.args.epochs or 10),
        ]
        self.run_cmd("Baseline: 完整 MPFC 模型", args)
        print("  [基础实验完成]")

    # -------- 消融实验 --------
    def run_ablation(self):
        print()
        print("█" * 70)
        print("  消融实验 (Ablation) - 逐一移除模块")
        print("█" * 70)

        if self.args.use_runner_ablation:
            # 方法一：使用 runner.py 内置消融（一次执行所有变体）
            args = self._base_args() + [
                "--ablation",
                "--name", "ablation_all",
            ]
            self.run_cmd("消融实验: runner.py 内置批量消融", args)
        else:
            # 方法二：逐一运行每个消融配置文件
            for variant in ABLATION_VARIANTS:
                config_file = ABLATION_DIR / f"{variant}.yaml"
                if not config_file.exists():
                    print(f"  [SKIP] Config not found: {config_file}")
                    continue
                args = self._base_args() + [
                    "--config", str(config_file),
                    "--name", f"ablation_{variant}",
                ]
                self.run_cmd(f"消融实验: {variant}", args)

        print("  [消融实验完成]")

    # -------- 参数敏感性实验 --------
    def run_sensitivity(self):
        print()
        print("█" * 70)
        print("  参数敏感性实验 (Sensitivity) - 网格搜索")
        print("█" * 70)

        for variant in SENSITIVITY_VARIANTS:
            config_file = SENSITIVITY_DIR / f"{variant}.yaml"
            if not config_file.exists():
                print(f"  [SKIP] Config not found: {config_file}")
                continue
            args = self._base_args() + [
                "--config", str(config_file),
                "--name", f"sensitivity_{variant}",
            ]
            self.run_cmd(f"参数敏感性: {variant}", args)

        print("  [参数敏感性实验完成]")

    # -------- VTA 损失解耦实验 --------
    def run_vta_decomp(self):
        print()
        print("█" * 70)
        print("  VTA 损失解耦实验 - Focal / RPE / PW 逐组件清零验证")
        print("█" * 70)

        config_file = CONFIG_DIR / "vta_decomposition.yaml"
        if not config_file.exists():
            print(f"  [SKIP] Config not found: {config_file}")
            return

        args = self._base_args() + [
            "--config", str(config_file),
            "--name", "vta_decomp",
        ]
        self.run_cmd("VTA 损失解耦: 5 个变体 (full / wo_focal / wo_rpe / wo_pw / bce_only)", args)

        print("  [VTA 损失解耦实验完成]")

    # -------- 多随机种子实验 --------
    def run_multi_seed(self):
        print()
        print("█" * 70)
        print("  多随机种子实验 (Multi-Seed) - 评估稳定性")
        print("█" * 70)

        if self.args.use_runner_multi_seed:
            # 方法一：使用 runner.py 内置多种子模式
            args = self._base_args() + [
                "--multi_seed",
                "--name", "multi_seed_5runs",
            ]
            self.run_cmd("多种子实验: runner.py 内置 5 个种子", args)
        else:
            # 方法二：逐一运行每个种子
            for seed in MULTI_SEEDS:
                args = self._base_args() + [
                    "--seed", str(seed),
                    "--name", f"multi_seed_seed{seed}",
                ]
                self.run_cmd(f"多种子实验: seed={seed}", args)

        print("  [多随机种子实验完成]")

    # -------- 结果汇总 --------
    def summarize(self):
        """汇总所有实验结果并打印统计。"""
        print()
        print("█" * 70)
        print("  实验结果汇总")
        print("█" * 70)

        all_results = self._collect_results()

        if not all_results:
            print("  (无实验结果可汇总)")
            return

        # 按实验类型分组
        grouped = defaultdict(list)
        for r in all_results:
            # 根据实验名称推断类型
            name = r.get("config_name", "")
            if name.startswith("ablation") or name.startswith("baseline"):
                group = "ablation" if "ablation" in name else "baseline"
            elif "sweep" in name or "sensitivity" in name:
                group = "sensitivity"
            elif "multi_seed" in name or "seed" in name:
                group = "multi_seed"
            else:
                group = "other"
            grouped[group].append(r)

        # 输出各组统计
        for group, results in sorted(grouped.items()):
            print(f"\n--- [{group.upper()}] ---")
            print(f"  {'实验名称':<35} {'best_val_f1':<10} {'val_auc':<10} {'val_ap':<10}")
            print(f"  {'-'*35} {'-'*10} {'-'*10} {'-'*10}")
            for r in results:
                name = r.get("config_name", "?")
                f1 = r.get("best_val_f1", "?")
                auc = r.get("val_metrics", {}).get("auc", "?")
                ap = r.get("val_metrics", {}).get("ap", "?")
                print(f"  {name:<35} {str(f1):<10} {str(auc):<10} {str(ap):<10}")

            # 多种子实验计算 mean±std
            if group == "multi_seed":
                f1_vals = [
                    float(r.get("best_val_f1", 0))
                    for r in results
                    if r.get("best_val_f1") not in ("", "?", None)
                ]
                auc_vals = [
                    float(r.get("val_metrics", {}).get("auc", 0))
                    for r in results
                    if r.get("val_metrics", {}).get("auc") not in ("", "?", None)
                ]
                if f1_vals:
                    import statistics
                    print(f"\n  >>> Multi-Seed Statistics (n={len(f1_vals)}):")
                    print(f"      F1:  mean={statistics.mean(f1_vals):.4f}  std={statistics.stdev(f1_vals):.4f}")
                if auc_vals:
                    import statistics
                    print(f"      AUC: mean={statistics.mean(auc_vals):.4f}  std={statistics.stdev(auc_vals):.4f}")

        # 导出到 CSV
        if self.args.export:
            self._export_results(all_results, self.args.export)

    def _collect_results(self) -> List[Dict]:
        """从 outputs 目录收集所有结果。"""
        results = []
        output_dir = Path(self.output_dir)
        if not output_dir.exists():
            return results

        # 查找所有实验目录下的 results.csv
        for exp_dir in sorted(output_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            csv_file = exp_dir / "results.csv"
            if not csv_file.exists():
                continue

            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    results.append(row)

        return results

    def _export_results(self, results: List[Dict], export_path: str):
        """导出汇总结果到 CSV。"""
        if not results:
            print("  [WARNING] No results to export.")
            return

        # 收集所有可能的字段
        fieldnames = set()
        for r in results:
            fieldnames.update(r.keys())
        fieldnames = sorted(fieldnames)

        with open(export_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\n  [结果已导出] {export_path} ({len(results)} 条记录)")

    # -------- 主流程 --------
    def run(self):
        start_time = time.time()

        if self.args.summarize_only:
            self.summarize()
            return

        if self.args.only:
            only = self.args.only.lower()
            if only == "baseline":
                self.run_baseline()
            elif only == "ablation":
                self.run_ablation()
            elif only == "sensitivity":
                self.run_sensitivity()
            elif only == "vta_decomp":
                self.run_vta_decomp()
            elif only == "multi_seed":
                self.run_multi_seed()
            else:
                print(f"Unknown experiment type: {only}")
                sys.exit(1)
        else:
            self.run_baseline()
            self.run_ablation()
            self.run_sensitivity()
            self.run_vta_decomp()
            self.run_multi_seed()

        if not self.args.no_summarize:
            self.summarize()

        elapsed = time.time() - start_time
        print(f"\n总耗时: {elapsed / 60:.1f} 分 ({elapsed:.0f} 秒)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML 反洗钱模型 - 全面实验调度脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 运行全部实验
  python scripts/run_experiments.py --llm_api_url http://localhost:11434/v1/chat/completions

  # 仅运行消融实验
  python scripts/run_experiments.py --only ablation --llm_api_url <URL>

  # 仅汇总已有结果
  python scripts/run_experiments.py --summarize-only

  # 导出结果到 CSV 文件
  python scripts/run_experiments.py --summarize-only --export summary.csv
        """,
    )

    parser.add_argument("--llm_api_url", type=str, default="http://localhost:11434/v1/chat/completions",
                        help="LLM API endpoint URL")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory (default: outputs/)")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Raw data CSV path")
    parser.add_argument("--preprocessed_path", type=str, default=None,
                        help="Preprocessed data path")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs for baseline (default: 10)")

    parser.add_argument("--only", type=str, choices=["baseline", "ablation", "sensitivity", "vta_decomp", "multi_seed"],
                        help="Run only one experiment type")
    parser.add_argument("--no_summarize", action="store_true",
                        help="Skip final summary")
    parser.add_argument("--summarize-only", action="store_true",
                        help="Only summarize existing results, do not run experiments")
    parser.add_argument("--export", type=str, default=None,
                        help="Export summary to CSV file path")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")

    # 实验方式选择
    parser.add_argument("--use-runner-ablation", action="store_true",
                        help="Use runner.py's built-in ablation (single command)")
    parser.add_argument("--use-runner-multi-seed", action="store_true",
                        help="Use runner.py's built-in multi-seed (single command)")

    return parser.parse_args()


def main():
    args = parse_args()
    manager = ExperimentManager(args)

    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║        AML 反洗钱检测 · 全面实验脚本 (Python)                  ║")
    print("╠══════════════════════════════════════════════════════════════════╣")
    print(f"║  LLM API : {args.llm_api_url}")
    print(f"║  输出目录: {args.output_dir or DEFAULT_OUTPUT_DIR}")
    print(f"║  Dry Run : {args.dry_run}")
    print("╚══════════════════════════════════════════════════════════════════╝")

    manager.run()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
命令行入口脚本 —— 支持各种实验模式。

用法：
    # 基础实验（MPFC，LLM 为 mPFC 内置组件）
    python scripts/run.py --config configs/default.yaml \
        --llm_api_url http://localhost:11434/v1/chat/completions

    # 消融实验（包含 wo_llm：保留 MPFC 但去除 LLM 符号推理）
    python scripts/run.py --config configs/default.yaml --ablation

    # 参数敏感性
    python scripts/run.py --config configs/sensitivity/batch_size.yaml

    # 多随机种子
    python scripts/run.py --config configs/default.yaml --multi_seed

    # 测试模式
    python scripts/run.py --config configs/default.yaml --test --checkpoint path/to/model.pt
"""

from __future__ import annotations

import argparse
import os
import sys

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.runner import ExperimentRunner
from train import setup_experiment


def parse_args():
    parser = argparse.ArgumentParser(
        description="AML 反洗钱检测实验脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 基础实验 (MPFC，LLM 为 mPFC 内置组件)
  python scripts/run.py --config configs/default.yaml \\
      --llm_api_url http://localhost:11434/v1/chat/completions

  # 消融实验 (含 wo_llm: 保留 MPFC 但去除 LLM)
  python scripts/run.py --config configs/default.yaml --ablation

  # 参数扫描
  python scripts/run.py --config configs/sensitivity/batch_size.yaml

  # 多随机种子
  python scripts/run.py --config configs/default.yaml --multi_seed

  # 仅测试
  python scripts/run.py --config configs/default.yaml --test --checkpoint outputs/.../model_best.pt
        """,
    )

    # 配置文件
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="输出目录覆盖",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子覆盖",
    )

    # 实验模式
    parser.add_argument(
        "--ablation", action="store_true",
        help="启用消融实验模式",
    )
    parser.add_argument(
        "--multi_seed", action="store_true",
        help="启用多随机种子实验",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="启用参数扫描",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="仅测试模式 (需指定 --checkpoint)",
    )

    # LLM 配置（LLM 是 mPFC 内置组件）
    parser.add_argument(
        "--llm_api_url", type=str, default=None,
        help="LLM API URL（mPFC 内置 LLM 规则生成接口）",
    )
    parser.add_argument(
        "--llm_model_name", type=str, default=None,
        help="LLM 模型名称",
    )

    # 数据
    parser.add_argument(
        "--data_path", type=str, default=None,
        help="数据路径覆盖",
    )
    parser.add_argument(
        "--use_smote", action="store_true",
        help="启用 SMOTE",
    )
    parser.add_argument(
        "--smote_ratio", type=float, default=None,
        help="SMOTE 比例",
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="强制重新预处理数据",
    )

    # 训练参数
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--gnn_layers", type=int, default=None)
    parser.add_argument("--gnn_heads", type=int, default=None)

    # 测试
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


def main():
    args = parse_args()

    # 构建命令行覆盖配置
    cli_overrides = {}

    if args.output_dir:
        cli_overrides["experiment"] = {"output_dir": args.output_dir}
    if args.seed:
        cli_overrides["experiment"] = cli_overrides.get("experiment", {})
        cli_overrides["experiment"]["seed"] = args.seed
    if args.test:
        cli_overrides["experiment"] = cli_overrides.get("experiment", {})
        cli_overrides["experiment"]["mode"] = "test"

    # LLM 是 mPFC 内置组件，通过配置进行设置
    if args.llm_api_url:
        cli_overrides.setdefault("model", {}).setdefault("llm", {})["api_url"] = args.llm_api_url
    if args.llm_model_name:
        cli_overrides.setdefault("model", {}).setdefault("llm", {})["model_name"] = args.llm_model_name

    if args.data_path:
        cli_overrides["data"] = {"data_path": args.data_path}
    if args.use_smote:
        cli_overrides.setdefault("data", {})["use_smote"] = True
    if args.smote_ratio:
        cli_overrides.setdefault("data", {})["smote_ratio"] = args.smote_ratio
    if args.regenerate:
        cli_overrides.setdefault("data", {})["regenerate"] = True

    if args.epochs:
        cli_overrides["train"] = {"epochs": args.epochs}
    if args.batch_size:
        cli_overrides["data"] = cli_overrides.get("data", {})
        cli_overrides["data"]["batch_size"] = args.batch_size
    if args.lr:
        cli_overrides["train"] = cli_overrides.get("train", {})
        cli_overrides["train"]["lr"] = args.lr
    if args.pos_weight:
        cli_overrides["train"] = cli_overrides.get("train", {})
        cli_overrides["train"]["pos_weight"] = args.pos_weight
    if args.focal_gamma:
        cli_overrides["train"] = cli_overrides.get("train", {})
        cli_overrides["train"]["focal_gamma"] = args.focal_gamma
    if args.patience:
        cli_overrides["train"] = cli_overrides.get("train", {})
        cli_overrides["train"]["patience"] = args.patience
    if args.gnn_layers:
        cli_overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_layers"] = args.gnn_layers
    if args.gnn_heads:
        cli_overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_heads"] = args.gnn_heads

    if args.ablation:
        cli_overrides["ablation"] = {"enabled": True}
    if args.multi_seed:
        cli_overrides["multi_seed"] = {"enabled": True}
    if args.sweep:
        cli_overrides["sweep"] = {"enabled": True}

    # 创建并运行实验
    cfg = setup_experiment(args.config, cli_overrides)

    if args.test and args.checkpoint:
        from trainers.base_trainer import BaseTrainer

        trainer = BaseTrainer(cfg)
        trainer.test(checkpoint_path=args.checkpoint)
    else:
        runner = ExperimentRunner(cfg, config_path=args.config)
        runner.run()


if __name__ == "__main__":
    main()

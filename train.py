#!/usr/bin/env python3
"""
统一训练入口 —— AML 反洗钱检测模型。

功能：
- 配置加载与命令行覆盖
- 实验运行（基础/消融/参数扫描/多种子/推理）
- 结果记录与可视化

用法：
    # 基础实验
    python train.py --config configs/default.yaml

    # 默认实验（LLM 为 mPFC 内置组件）
    python train.py --config configs/default.yaml --llm_api_url http://localhost:11434/v1/chat/completions

    # 消融实验
    python train.py --config configs/default.yaml --ablation

    # 参数敏感性
    python train.py --config configs/sensitivity/batch_size.yaml

    # 多随机种子
    python train.py --config configs/default.yaml --multi_seed

    # 推理测试
    python train.py --config configs/default.yaml --test --checkpoint outputs/model_best.pt

    # 快速覆盖
    python train.py --config configs/default.yaml --epochs 20 --lr 0.0005 --batch_size 64
"""

import argparse
import os
import sys
from typing import Dict, Optional

from experiments.runner import ExperimentRunner
from utils import Logger, load_config, merge_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML 反洗钱检测模型训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径 (默认: configs/default.yaml)",
    )

    # 实验模式
    parser.add_argument(
        "--ablation", action="store_true",
        help="运行消融实验",
    )
    parser.add_argument(
        "--multi_seed", action="store_true",
        help="运行多随机种子实验",
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="运行参数敏感性扫描",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="推理测试模式（需配合 --checkpoint）",
    )

    # 实验名称与输出
    parser.add_argument("--name", type=str, default=None, help="实验名称")
    parser.add_argument("--output_dir", type=str, default=None, help="输出目录")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")

    # 模型超参数
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)

    # LLM 参数（LLM 是 mPFC 内置组件）
    parser.add_argument("--llm_api_url", type=str, default=None, help="LLM API URL（mPFC 内置 LLM 规则生成接口）")
    parser.add_argument("--llm_model_name", type=str, default=None, help="LLM 模型名称")
    parser.add_argument("--llm_update_freq", type=int, default=None, help="LLM 规则更新频率")

    # GNN 参数
    parser.add_argument("--gnn_layers", type=int, default=None)
    parser.add_argument("--gnn_heads", type=int, default=None)

    # 数据
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--preprocessed_path", type=str, default=None)
    parser.add_argument("--use_smote", action="store_true")
    parser.add_argument("--smote_ratio", type=float, default=None)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=None)
    parser.add_argument("--test_ratio", type=float, default=None)

    # 训练
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--focal_gamma", type=float, default=None)
    parser.add_argument("--rpe_beta", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)

    # 测试
    parser.add_argument("--checkpoint", type=str, default=None)

    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict:
    """将命令行参数转为配置覆盖字典。"""
    overrides: Dict = {}

    # 实验模式
    if args.ablation:
        overrides["ablation"] = {"enabled": True}
    if args.multi_seed:
        overrides["multi_seed"] = {"enabled": True}
    if args.sweep:
        overrides["sweep"] = {"enabled": True}
    if args.test:
        overrides["experiment"] = {"mode": "test"}

    # 实验元信息
    if args.name:
        overrides.setdefault("experiment", {})["name"] = args.name
    if args.output_dir:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.seed:
        overrides.setdefault("experiment", {})["seed"] = args.seed

    # 模型
    if args.hidden_dim:
        overrides.setdefault("model", {})["hidden_dim"] = args.hidden_dim
    if args.dropout:
        overrides.setdefault("model", {})["dropout"] = args.dropout

    # LLM
    if args.llm_api_url:
        overrides.setdefault("model", {}).setdefault("llm", {})["api_url"] = args.llm_api_url
    if args.llm_model_name:
        overrides.setdefault("model", {}).setdefault("llm", {})["model_name"] = args.llm_model_name
    if args.llm_update_freq:
        overrides.setdefault("model", {}).setdefault("llm", {})["rule_update_frequency"] = args.llm_update_freq

    # GNN
    if args.gnn_layers:
        overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_layers"] = args.gnn_layers
    if args.gnn_heads:
        overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_heads"] = args.gnn_heads

    # 数据
    if args.data_path:
        overrides.setdefault("data", {})["data_path"] = args.data_path
    if args.preprocessed_path:
        overrides.setdefault("data", {})["preprocessed_path"] = args.preprocessed_path
    if args.use_smote:
        overrides.setdefault("data", {})["use_smote"] = True
    if args.smote_ratio:
        overrides.setdefault("data", {})["smote_ratio"] = args.smote_ratio
    if args.regenerate:
        overrides.setdefault("data", {})["regenerate"] = True
    if args.val_ratio:
        overrides.setdefault("data", {})["val_ratio"] = args.val_ratio
    if args.test_ratio:
        overrides.setdefault("data", {})["test_ratio"] = args.test_ratio
    if args.batch_size:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size

    # 训练
    train_overrides = {}
    if args.epochs:
        train_overrides["epochs"] = args.epochs
    if args.lr:
        train_overrides["lr"] = args.lr
    if args.weight_decay:
        train_overrides["weight_decay"] = args.weight_decay
    if args.pos_weight:
        train_overrides["pos_weight"] = args.pos_weight
    if args.focal_gamma:
        train_overrides["focal_gamma"] = args.focal_gamma
    if args.rpe_beta:
        train_overrides["rpe_beta"] = args.rpe_beta
    if args.patience:
        train_overrides["patience"] = args.patience
    if args.grad_clip:
        train_overrides["grad_clip"] = args.grad_clip
    if train_overrides:
        overrides["train"] = train_overrides

    return overrides


def setup_experiment(
    config_path: str, cli_overrides: Optional[Dict] = None
) -> Config:
    """
    加载配置并设置实验环境。

    Args:
        config_path: YAML 配置文件路径
        cli_overrides: 命令行参数覆盖字典

    Returns:
        cfg: 合并后的配置对象
    """
    # 解析配置文件路径
    if not os.path.isabs(config_path):
        # 尝试相对路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(base_dir, config_path)

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    # 加载基础配置
    cfg = load_config(config_path)

    # 合并 CLI 覆盖
    if cli_overrides:
        cfg = merge_config(cfg, cli_overrides)

    # 创建输出目录
    output_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    return cfg


def main():
    args = parse_args()
    overrides = build_overrides(args)
    cfg = setup_experiment(args.config, overrides)

    # 记录配置
    logger = Logger(
        name="train",
        log_dir=os.path.join(cfg.experiment.output_dir, cfg.experiment.name),
    )
    logger.info(f"Config: {args.config}")
    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Model: {cfg.model.name}")

    if args.test and args.checkpoint:
        # 推理测试模式
        from trainers.base_trainer import BaseTrainer

        logger.info(f"Test mode with checkpoint: {args.checkpoint}")
        trainer = BaseTrainer(cfg)
        trainer.test(checkpoint_path=args.checkpoint)
    else:
        # 正常实验模式
        runner = ExperimentRunner(cfg, config_path=args.config)
        runner.run()


if __name__ == "__main__":
    main()

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

    # 消融实验
    python train.py --config configs/default.yaml --ablation

    # 参数敏感性
    python train.py --config configs/sensitivity/batch_size.yaml

    # 多随机种子
    python train.py --config configs/default.yaml --multi_seed

    # 推理测试
    python train.py --config configs/default.yaml --test --checkpoint outputs/model_best.pt

    # 断点续训（自动找最新 checkpoint）
    python train.py --config configs/default.yaml --resume

    # 断点续训（指定 checkpoint 路径）
    python train.py --config configs/default.yaml --resume outputs/exp_name/run_xxx/ckpt/latest.pt

    # 快速覆盖
    python train.py --config configs/default.yaml --epochs 20 --lr 0.0005 --batch_size 64
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

from core.runner import ExperimentRunner
from utils import Logger, load_config, merge_config
from utils.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML 反洗钱检测模型训练入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # 配置（支持多个 --config，后面的覆盖前面的）
    parser.add_argument(
        "--config", type=str, action="append", default=None,
        help='配置文件路径（可指定多个，后面的覆盖前面的；默认: configs/default.yaml）',
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
        "--resume", nargs="?", const="auto", default=None,
        help='断点续训。不传值 = 自动找最新的 latest.pt; 传路径 = 使用指定 checkpoint',
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

    # LLM 参数
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
    if args.resume is not None:
        overrides["experiment"] = {"mode": "resume"}
        if args.resume != "auto":
            # 显式指定了 checkpoint 路径
            overrides["experiment"]["resume_ckpt"] = args.resume
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
    config_paths: List[str], cli_overrides: Optional[Dict] = None
) -> Config:
    """
    加载配置并设置实验环境。

    Args:
        config_paths: YAML 配置文件路径列表（后面的覆盖前面的）
        cli_overrides: 命令行参数覆盖字典

    Returns:
        cfg: 合并后的配置对象
    """
    if not config_paths:
        config_paths = ["configs/default.yaml"]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 解析并加载所有配置文件，依次合并（后面的覆盖前面的）
    cfg = None
    for cp in config_paths:
        if not os.path.isabs(cp):
            cp = os.path.join(base_dir, cp)
        if not os.path.exists(cp):
            raise FileNotFoundError(f"Configuration file not found: {cp}")
        if cfg is None:
            cfg = load_config(cp)
        else:
            # load_config 返回 Config 对象，转 dict 后再 merge
            from dataclasses import asdict
            override_cfg = load_config(cp)
            cfg = merge_config(cfg, asdict(override_cfg))

    if cfg is None:
        raise FileNotFoundError("No configuration file loaded.")

    # 合并 CLI 覆盖
    if cli_overrides:
        cfg = merge_config(cfg, cli_overrides)

    # 创建输出目录
    output_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    # 将数据路径转为绝对路径（相对于主配置文件所在目录）
    main_config = config_paths[0]
    if not os.path.isabs(main_config):
        main_config = os.path.join(base_dir, main_config)
    config_dir = os.path.dirname(os.path.abspath(main_config))
    if cfg.data.data_path and not os.path.isabs(cfg.data.data_path):
        cfg.data.data_path = os.path.join(config_dir, cfg.data.data_path)
    if cfg.data.preprocessed_path and not os.path.isabs(cfg.data.preprocessed_path):
        cfg.data.preprocessed_path = os.path.join(config_dir, cfg.data.preprocessed_path)

    # 保存最终配置到实验输出目录（供 test/resume 模式复用）
    import yaml
    saved_config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(saved_config_path):
        try:
            with open(saved_config_path, "w") as f:
                yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"[WARN] Failed to save config: {e}")

    return cfg


def main():
    args = parse_args()
    overrides = build_overrides(args)
    config_paths = args.config if args.config else ["configs/default.yaml"]
    cfg = setup_experiment(config_paths, overrides)

    # 记录配置（仅控制台，文件日志由 base_trainer 负责）
    logger = Logger(
        name="train",
        log_dir=os.path.join(cfg.experiment.output_dir, cfg.experiment.name),
        log_file=False,
    )
    logger.info(f"Config: {' + '.join(config_paths)}")
    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Model: {cfg.model.name}")

    if args.test and args.checkpoint:
        # 推理测试模式
        from core.base_trainer import BaseTrainer

        # 如果数据路径不合法（用最小配置如 wo_ca1.yaml 加载时可能丢失路径），
        # 尝试从 checkpoint 目录反推：.../exp_name/run_xxx/ckpt/ → exp_name → 找 default.yaml
        if not cfg.data.data_path or not os.path.exists(cfg.data.data_path):
            ckpt_dir_parts = os.path.normpath(args.checkpoint).split(os.sep)
            # 从 checkpoint 路径中找到 exp_name（run_xxx 的上一级）
            try:
                run_idx = next(i for i, p in enumerate(ckpt_dir_parts) if p.startswith("run_"))
                exp_name = ckpt_dir_parts[run_idx - 1]
                # 尝试加载同实验名下保存的完整配置
                exp_config = os.path.join(cfg.experiment.output_dir, exp_name, "config.yaml")
                if os.path.exists(exp_config):
                    cfg = load_config(exp_config)
                    logger.info(f"Reused saved config from {exp_config}")
                else:
                    # 退化回 default.yaml
                    fallback = "configs/default.yaml"
                    if not os.path.isabs(fallback):
                        fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), fallback)
                    if os.path.exists(fallback):
                        cfg = load_config(fallback)
                        cfg.experiment.name = exp_name
                        logger.info(f"Fallback to {fallback} (data path not found)")
                        # 重新 resolve 数据路径
                        config_dir = os.path.dirname(os.path.abspath(fallback))
                        if cfg.data.data_path and not os.path.isabs(cfg.data.data_path):
                            cfg.data.data_path = os.path.join(config_dir, cfg.data.data_path)
                        if cfg.data.preprocessed_path and not os.path.isabs(cfg.data.preprocessed_path):
                            cfg.data.preprocessed_path = os.path.join(config_dir, cfg.data.preprocessed_path)
            except (StopIteration, IndexError):
                logger.warning("Could not infer experiment from checkpoint path, using current config.")

        logger.info(f"Test mode with checkpoint: {args.checkpoint}")
        logger.info(f"Data path: {cfg.data.data_path}")
        logger.info(f"Preprocessed path: {cfg.data.preprocessed_path}")
        trainer = BaseTrainer(cfg)
        trainer.test(checkpoint_path=args.checkpoint)
    elif args.resume:
        # 断点续训模式
        from core.base_trainer import BaseTrainer

        logger.info(f"Resume mode: {cfg.experiment.name}")
        trainer = BaseTrainer(cfg, resume=True)
        trainer.train()
    else:
        # 正常实验模式
        runner = ExperimentRunner(cfg, config_path=config_paths[0])
        runner.run()


if __name__ == "__main__":
    main()

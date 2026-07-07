#!/usr/bin/env python3
"""
Unified training entrypoint for AML experiments.

Examples:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --ablation
    python train.py --config configs/default.yaml --multi_seed
    python train.py --config configs/default.yaml --epochs 20 --lr 0.0005 --batch_size 64
    python train.py --config configs/default.yaml --test --checkpoint outputs/model_best.pt
    python train.py --config configs/default.yaml --resume
    python train.py --config configs/default.yaml --resume outputs/exp_name/run_xxx/ckpt/latest.pt
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Optional

import yaml

from core.runner import ExperimentRunner
from utils import Logger, load_config, merge_config
from utils.config import Config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AML training entrypoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help="Config file path. Can be specified multiple times. Defaults to configs/default.yaml.",
    )

    parser.add_argument("--ablation", action="store_true", help="Run ablation experiments.")
    parser.add_argument("--multi_seed", action="store_true", help="Run multi-seed experiments.")
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep experiments.")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="Resume training. Omit the value to auto-discover the latest checkpoint, or pass a checkpoint path.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test mode. Usually paired with --checkpoint.",
    )

    parser.add_argument("--name", type=str, default=None, help="Experiment name override.")
    parser.add_argument("--output_dir", type=str, default=None, help="Experiment output directory override.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override.")

    parser.add_argument("--hidden_dim", type=int, default=None, help="Model hidden dimension override.")
    parser.add_argument("--dropout", type=float, default=None, help="Model dropout override.")

    parser.add_argument(
        "--llm_api_url",
        type=str,
        default=None,
        help="LLM API URL used by the MPFC rule generator.",
    )
    parser.add_argument("--llm_model_name", type=str, default=None, help="LLM model name override.")
    parser.add_argument("--llm_update_freq", type=int, default=None, help="LLM rule update frequency override.")

    parser.add_argument("--gnn_layers", type=int, default=None, help="Number of MPFC GNN layers.")
    parser.add_argument("--gnn_heads", type=int, default=None, help="Number of MPFC attention heads.")

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help='Dataset name. Common values: "aml", "cryptopia", "amlsim_hi", "amlsim_li".',
    )
    parser.add_argument("--data_path", type=str, default=None, help="Raw dataset path override.")
    parser.add_argument(
        "--preprocessed_path",
        type=str,
        default=None,
        help="Preprocessed cache path override.",
    )
    parser.add_argument("--use_smote", action="store_true", help="Enable SMOTE for the train split.")
    parser.add_argument("--smote_ratio", type=float, default=None, help="SMOTE sampling ratio override.")
    parser.add_argument("--regenerate", action="store_true", help="Force preprocessing regeneration.")
    parser.add_argument("--val_ratio", type=float, default=None, help="Validation ratio override.")
    parser.add_argument("--test_ratio", type=float, default=None, help="Test ratio override.")

    parser.add_argument("--epochs", type=int, default=None, help="Training epoch override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size override.")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate override.")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay override.")
    parser.add_argument("--pos_weight", type=float, default=None, help="Positive-class weight override.")
    parser.add_argument("--focal_gamma", type=float, default=None, help="Focal loss gamma override.")
    parser.add_argument("--rpe_beta", type=float, default=None, help="VTA RPE beta override.")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience override.")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping override.")

    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for --test.")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> Dict:
    """Convert CLI arguments into config overrides."""
    overrides: Dict = {}

    if args.ablation:
        overrides["ablation"] = {"enabled": True}
    if args.multi_seed:
        overrides["multi_seed"] = {"enabled": True}
    if args.sweep:
        overrides["sweep"] = {"enabled": True}
    if args.resume is not None:
        overrides["experiment"] = {"mode": "resume"}
        if args.resume != "auto":
            overrides["experiment"]["resume_ckpt"] = args.resume
    if args.test:
        overrides["experiment"] = {"mode": "test"}

    if args.name:
        overrides.setdefault("experiment", {})["name"] = args.name
    if args.output_dir:
        overrides.setdefault("experiment", {})["output_dir"] = args.output_dir
    if args.seed is not None:
        overrides.setdefault("experiment", {})["seed"] = args.seed

    if args.hidden_dim is not None:
        overrides.setdefault("model", {})["hidden_dim"] = args.hidden_dim
    if args.dropout is not None:
        overrides.setdefault("model", {})["dropout"] = args.dropout

    if args.llm_api_url:
        overrides.setdefault("model", {}).setdefault("llm", {})["api_url"] = args.llm_api_url
    if args.llm_model_name:
        overrides.setdefault("model", {}).setdefault("llm", {})["model_name"] = args.llm_model_name
    if args.llm_update_freq is not None:
        overrides.setdefault("model", {}).setdefault("llm", {})["rule_update_frequency"] = args.llm_update_freq

    if args.gnn_layers is not None:
        overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_layers"] = args.gnn_layers
    if args.gnn_heads is not None:
        overrides.setdefault("model", {}).setdefault("mpfc", {})["gnn_heads"] = args.gnn_heads

    if args.dataset:
        overrides.setdefault("data", {})["dataset"] = args.dataset
    if args.data_path:
        overrides.setdefault("data", {})["data_path"] = args.data_path
    if args.preprocessed_path:
        overrides.setdefault("data", {})["preprocessed_path"] = args.preprocessed_path
    if args.use_smote:
        overrides.setdefault("data", {})["use_smote"] = True
    if args.smote_ratio is not None:
        overrides.setdefault("data", {})["smote_ratio"] = args.smote_ratio
    if args.regenerate:
        overrides.setdefault("data", {})["regenerate"] = True
    if args.val_ratio is not None:
        overrides.setdefault("data", {})["val_ratio"] = args.val_ratio
    if args.test_ratio is not None:
        overrides.setdefault("data", {})["test_ratio"] = args.test_ratio
    if args.batch_size is not None:
        overrides.setdefault("data", {})["batch_size"] = args.batch_size

    train_overrides = {}
    if args.epochs is not None:
        train_overrides["epochs"] = args.epochs
    if args.lr is not None:
        train_overrides["lr"] = args.lr
    if args.weight_decay is not None:
        train_overrides["weight_decay"] = args.weight_decay
    if args.pos_weight is not None:
        train_overrides["pos_weight"] = args.pos_weight
    if args.focal_gamma is not None:
        train_overrides["focal_gamma"] = args.focal_gamma
    if args.rpe_beta is not None:
        train_overrides["rpe_beta"] = args.rpe_beta
    if args.patience is not None:
        train_overrides["patience"] = args.patience
    if args.grad_clip is not None:
        train_overrides["grad_clip"] = args.grad_clip
    if train_overrides:
        overrides["train"] = train_overrides

    return overrides


def setup_experiment(config_paths: List[str], cli_overrides: Optional[Dict] = None) -> Config:
    """Load, merge, and normalize experiment configs."""
    if not config_paths:
        config_paths = ["configs/default.yaml"]

    base_dir = os.path.dirname(os.path.abspath(__file__))

    cfg = None
    for cp in config_paths:
        abs_path = cp if os.path.isabs(cp) else os.path.join(base_dir, cp)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Configuration file not found: {abs_path}")
        if cfg is None:
            cfg = load_config(abs_path)
        else:
            with open(abs_path, "r", encoding="utf-8") as f:
                override_dict = yaml.safe_load(f) or {}
            cfg = merge_config(cfg, override_dict)

    if cfg is None:
        raise FileNotFoundError("No configuration file loaded.")

    if cli_overrides:
        cfg = merge_config(cfg, cli_overrides)

    output_dir = os.path.join(cfg.experiment.output_dir, cfg.experiment.name)
    os.makedirs(output_dir, exist_ok=True)

    main_config = config_paths[0]
    if not os.path.isabs(main_config):
        main_config = os.path.join(base_dir, main_config)
    config_dir = os.path.dirname(os.path.abspath(main_config))

    if cfg.data.data_path and not os.path.isabs(cfg.data.data_path):
        config_rel = os.path.join(config_dir, cfg.data.data_path)
        project_rel = os.path.join(base_dir, cfg.data.data_path)
        if os.path.exists(config_rel):
            cfg.data.data_path = config_rel
        elif os.path.exists(project_rel):
            cfg.data.data_path = project_rel
        else:
            cfg.data.data_path = config_rel

    if cfg.data.preprocessed_path and not os.path.isabs(cfg.data.preprocessed_path):
        cfg.data.preprocessed_path = os.path.join(config_dir, cfg.data.preprocessed_path)

    preprocessed_dir = os.path.dirname(cfg.data.preprocessed_path)
    if preprocessed_dir and not os.path.exists(preprocessed_dir):
        os.makedirs(preprocessed_dir, exist_ok=True)

    saved_config_path = os.path.join(output_dir, "config.yaml")
    if not os.path.exists(saved_config_path):
        try:
            with open(saved_config_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg.to_dict(), f, default_flow_style=False, allow_unicode=True)
        except Exception as exc:
            print(f"[WARN] Failed to save config: {exc}")

    return cfg


def _normalize_dataset_name(dataset_name: str) -> str:
    return "cryptopia" if dataset_name == "cryptopia_graph" else dataset_name


def _make_single_trainer(cfg: Config, resume: bool = False):
    dataset_name = _normalize_dataset_name(cfg.data.dataset)
    if dataset_name == "cryptopia":
        from core.graph_trainer import GraphNodeTrainer

        cfg.data.dataset = dataset_name
        return GraphNodeTrainer(cfg, resume=resume)

    from core.base_trainer import BaseTrainer

    cfg.data.dataset = dataset_name
    return BaseTrainer(cfg, resume=resume)


def main():
    args = parse_args()
    overrides = build_overrides(args)
    config_paths = args.config if args.config else ["configs/default.yaml"]
    cfg = setup_experiment(config_paths, overrides)

    logger = Logger(
        name="train",
        log_dir=os.path.join(cfg.experiment.output_dir, cfg.experiment.name),
        log_file=False,
    )
    logger.info(f"Config: {' + '.join(config_paths)}")
    logger.info(f"Experiment: {cfg.experiment.name}")
    logger.info(f"Model: {cfg.model.name}")

    if args.test and args.checkpoint:
        if not cfg.data.data_path or not os.path.exists(cfg.data.data_path):
            ckpt_dir_parts = os.path.normpath(args.checkpoint).split(os.sep)
            try:
                run_idx = next(i for i, part in enumerate(ckpt_dir_parts) if part.startswith("run_"))
                exp_name = ckpt_dir_parts[run_idx - 1]
                exp_config = os.path.join(cfg.experiment.output_dir, exp_name, "config.yaml")
                if os.path.exists(exp_config):
                    cfg = load_config(exp_config)
                    logger.info(f"Reused saved config from {exp_config}")
                else:
                    fallback = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs/default.yaml")
                    if os.path.exists(fallback):
                        cfg = load_config(fallback)
                        cfg.experiment.name = exp_name
                        logger.info(f"Fallback to {fallback} (data path not found)")
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
        trainer = _make_single_trainer(cfg)
        trainer.test(checkpoint_path=args.checkpoint)
        return

    if args.resume:
        logger.info(f"Resume mode: {cfg.experiment.name}")
        trainer = _make_single_trainer(cfg, resume=True)
        trainer.train()
        return

    runner = ExperimentRunner(cfg, config_path=config_paths[0])
    runner.run()


if __name__ == "__main__":
    main()

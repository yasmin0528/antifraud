"""
Checkpoint 管理 —— 自动保存、恢复、最佳模型追踪。
"""

from __future__ import annotations

import os
import glob
from typing import Dict, Optional, Any

import torch


class CheckpointManager:
    """
    Checkpoint 管理器。

    特性：
    - 自动保存最新模型
    - 保存最佳模型（基于验证指标）
    - 断点续训
    - 自动清理旧 checkpoint
    """

    def __init__(
        self,
        ckpt_dir: str,
        experiment_name: str = "",
        metric_name: str = "val/f1",
        mode: str = "max",          # "max" | "min"
        max_to_keep: int = 3,
    ):
        # ckpt_dir 由调用方（BaseTrainer）决定，不再二次嵌套
        self.ckpt_dir = ckpt_dir
        self.metric_name = metric_name
        self.mode = mode
        self.max_to_keep = max_to_keep

        os.makedirs(self.ckpt_dir, exist_ok=True)

        self.best_metric = -float("inf") if mode == "max" else float("inf")
        self.best_path: Optional[str] = None

    def _is_better(self, value: float) -> bool:
        if self.mode == "max":
            return value > self.best_metric
        return value < self.best_metric

    def save(
        self,
        state: Dict[str, Any],
        epoch: int,
        step: int,
        metric_value: Optional[float] = None,
        is_best: bool = False,
    ):
        """保存 checkpoint（仅保留 latest.pt，不保留 epoch_*.pt 避免磁盘膨胀）。"""
        # 始终保存最新
        latest_path = os.path.join(self.ckpt_dir, "latest.pt")
        torch.save(state, latest_path)

        # 如果指定了指标，更新最佳
        if metric_value is not None and self._is_better(metric_value):
            self.best_metric = metric_value
            self.best_path = os.path.join(self.ckpt_dir, "best.pt")
            torch.save(state, self.best_path)

    def save_best(self, state: Dict[str, Any], metric_value: float):
        """显式保存最佳模型。"""
        self.best_metric = metric_value
        self.best_path = os.path.join(self.ckpt_dir, "best.pt")
        torch.save(state, self.best_path)

    def load_latest(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """加载最新 checkpoint。"""
        path = os.path.join(self.ckpt_dir, "latest.pt")
        if os.path.exists(path):
            return torch.load(path, map_location=device)
        return None

    def load_best(self, device: str = "cpu") -> Optional[Dict[str, Any]]:
        """加载最佳 checkpoint。"""
        if self.best_path and os.path.exists(self.best_path):
            return torch.load(self.best_path, map_location=device)
        # 尝试找 best.pt
        path = os.path.join(self.ckpt_dir, "best.pt")
        if os.path.exists(path):
            return torch.load(path, map_location=device)
        return None

    def _cleanup(self):
        """清理多余的 epoch checkpoint。"""
        pattern = os.path.join(self.ckpt_dir, "epoch_*.pt")
        files = sorted(glob.glob(pattern))
        while len(files) > self.max_to_keep:
            os.remove(files.pop(0))

    def get_best_metric(self) -> float:
        return self.best_metric

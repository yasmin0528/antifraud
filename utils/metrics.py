"""
评估指标计算与追踪。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


class ClassificationMetrics:
    """
    二分类指标计算器。

    用法：
        metrics = ClassificationMetrics(y_true, y_prob, threshold=0.5)
        print(metrics.report())
    """

    def __init__(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ):
        self.y_true = np.asarray(y_true).ravel()
        self.y_prob = np.asarray(y_prob).ravel()
        self.threshold = threshold
        self.y_pred = (self.y_prob >= threshold).astype(int)

    @property
    def acc(self) -> float:
        return accuracy_score(self.y_true, self.y_pred)

    @property
    def precision(self) -> float:
        return precision_score(self.y_true, self.y_pred, zero_division=0)

    @property
    def recall(self) -> float:
        return recall_score(self.y_true, self.y_pred, zero_division=0)

    @property
    def f1(self) -> float:
        return f1_score(self.y_true, self.y_pred, zero_division=0)

    @property
    def auc(self) -> Optional[float]:
        try:
            return roc_auc_score(self.y_true, self.y_prob)
        except ValueError:
            return None

    @property
    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred)

    def report(self) -> Dict[str, float]:
        return {
            "acc": self.acc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc if self.auc is not None else float("nan"),
            "threshold": self.threshold,
        }

    @staticmethod
    def search_best_threshold(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        thresholds: Optional[np.ndarray] = None,
    ) -> (float, Dict[str, float]):
        """搜索最优阈值（基于 F1）。"""
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 91)

        best_f1 = -1.0
        best_thr = 0.5
        best_metrics = {}

        for thr in thresholds:
            y_pred = (y_prob >= thr).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_metrics = ClassificationMetrics(y_true, y_prob, thr).report()

        return best_thr, best_metrics


class MetricTracker:
    """
    指标追踪器 —— 累积所有 batch 的结果，最后统一计算。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._probs: List[torch.Tensor] = []
        self._labels: List[torch.Tensor] = []
        self._losses: List[float] = []

    def update(
        self,
        prob: torch.Tensor,
        label: torch.Tensor,
        loss: Optional[float] = None,
    ):
        self._probs.append(prob.detach().cpu())
        self._labels.append(label.detach().cpu())
        if loss is not None:
            self._losses.append(loss)

    @property
    def avg_loss(self) -> float:
        if not self._losses:
            return float("nan")
        return float(np.mean(self._losses))

    def compute(self, threshold: float = 0.5) -> Dict[str, float]:
        y_true = torch.cat(self._labels).numpy()
        y_prob = torch.cat(self._probs).numpy()
        metrics = ClassificationMetrics(y_true, y_prob, threshold)
        result = metrics.report()
        result["avg_loss"] = self.avg_loss
        return result

    def compute_best(self) -> (float, Dict[str, float]):
        y_true = torch.cat(self._labels).numpy()
        y_prob = torch.cat(self._probs).numpy()
        return ClassificationMetrics.search_best_threshold(y_true, y_prob)

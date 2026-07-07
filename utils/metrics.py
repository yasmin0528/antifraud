"""
评估指标计算与追踪。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
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
    def ap(self) -> Optional[float]:
        try:
            return average_precision_score(self.y_true, self.y_prob)
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
            "ap": self.ap if self.ap is not None else float("nan"),
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


def aggregate_group_scores(
    group_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    agg: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    group_ids = np.asarray(group_ids).ravel()
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()

    valid_mask = group_ids >= 0
    if not valid_mask.any():
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    valid_groups = np.unique(group_ids[valid_mask])
    agg_true: List[int] = []
    agg_prob: List[float] = []
    for gid in valid_groups:
        mask = group_ids == gid
        agg_true.append(int(y_true[mask].max()))
        if agg == "mean":
            agg_prob.append(float(y_prob[mask].mean()))
        else:
            agg_prob.append(float(y_prob[mask].max()))
    return np.asarray(agg_true), np.asarray(agg_prob, dtype=np.float32)


def compute_alert_level_metrics(
    group_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    agg: str = "max",
) -> Dict[str, float]:
    agg_true, agg_prob = aggregate_group_scores(group_ids, y_true, y_prob, agg=agg)
    if agg_true.size == 0:
        return {"alert_level_ap": float("nan"), "alert_level_f1": float("nan")}
    report = ClassificationMetrics(agg_true, agg_prob, threshold).report()
    return {
        "alert_level_ap": report["ap"],
        "alert_level_f1": report["f1"],
    }


def compute_hit_at_k(
    group_ids: np.ndarray,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int = 10,
    agg: str = "max",
) -> float:
    agg_true, agg_prob = aggregate_group_scores(group_ids, y_true, y_prob, agg=agg)
    if agg_true.size == 0:
        return float("nan")
    order = np.argsort(-agg_prob)[: max(int(k), 1)]
    return float(agg_true[order].mean()) if order.size > 0 else float("nan")


def compute_subgraph_coverage(
    true_group_ids: np.ndarray,
    pred_scores: np.ndarray,
    top_k: int = 10,
) -> Dict[str, float]:
    true_group_ids = np.asarray(true_group_ids).ravel()
    pred_scores = np.asarray(pred_scores).ravel()
    valid_mask = true_group_ids >= 0
    if not valid_mask.any():
        return {
            "subgraph_coverage_node": float("nan"),
            "subgraph_coverage_edge": float("nan"),
        }

    top_k = min(max(int(top_k), 1), pred_scores.size)
    top_idx = np.argsort(-pred_scores)[:top_k]
    pred_mask = np.zeros(pred_scores.size, dtype=bool)
    pred_mask[top_idx] = True
    overlap = float((pred_mask & valid_mask).sum())
    denom = float(valid_mask.sum()) if valid_mask.sum() > 0 else float("nan")
    coverage = overlap / denom if denom and not np.isnan(denom) else float("nan")
    return {
        "subgraph_coverage_node": coverage,
        "subgraph_coverage_edge": coverage,
    }

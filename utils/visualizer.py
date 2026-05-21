"""
可视化模块 —— 支持科研论文级别的图表生成。

功能：
- Loss 曲线
- ROC 曲线
- PR 曲线
- t-SNE 节点嵌入
- Attention 热图
- 混淆矩阵
- 网络结构图
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # 非交互后端，支持无显示器环境
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, auc
from sklearn.manifold import TSNE

# 中文字体支持（如果可用）；无中文环境的 Linux 自动回退 DejaVu Sans
import matplotlib.font_manager as _fm
_available_fonts = {f.name for f in _fm.fontManager.ttflist}
_font_list = ["DejaVu Sans"]
for _font in ["SimHei", "Arial Unicode MS"]:
    if _font in _available_fonts:
        _font_list.append(_font)
plt.rcParams["font.family"] = _font_list
plt.rcParams["axes.unicode_minus"] = False


class Visualizer:
    """
    可视化器。

    Args:
        save_dir: 图表保存目录
        dpi: 图片分辨率
        format: 保存格式
    """

    def __init__(
        self,
        save_dir: str = "outputs/figures",
        dpi: int = 150,
        fmt: str = "png",
    ):
        self.save_dir = save_dir
        self.dpi = dpi
        self.fmt = fmt
        os.makedirs(save_dir, exist_ok=True)

    def _save(self, name: str):
        path = os.path.join(self.save_dir, f"{name}.{self.fmt}")
        plt.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close()
        return path

    def plot_loss_curve(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "Training Loss Curve",
        name: str = "loss_curve",
    ) -> str:
        """绘制 loss 曲线。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, "b-", label="Train Loss", linewidth=2)

        if val_losses:
            val_epochs = range(1, len(val_losses) + 1)
            ax.plot(val_epochs, val_losses, "r-", label="Val Loss", linewidth=2)

        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return self._save(name)

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "ROC Curve",
        name: str = "roc_curve",
    ) -> str:
        """绘制 ROC 曲线。"""
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return self._save(name)

    def plot_pr_curve(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        title: str = "Precision-Recall Curve",
        name: str = "pr_curve",
    ) -> str:
        """绘制 PR 曲线。"""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision, "b-", linewidth=2, label=f"PR (AUC = {pr_auc:.4f})")

        ax.set_xlabel("Recall", fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return self._save(name)

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        title: str = "Confusion Matrix",
        name: str = "confusion_matrix",
    ) -> str:
        """绘制混淆矩阵。"""
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Fraud"])
        ax.set_yticklabels(["Normal", "Fraud"])

        # 标注数值
        thresh = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i, j], "d"),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        ax.set_xlabel("Predicted", fontsize=12)
        ax.set_ylabel("True", fontsize=12)
        ax.set_title(title, fontsize=14)
        return self._save(name)

    def plot_tsne(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        title: str = "t-SNE Visualization",
        name: str = "tsne",
        perplexity: int = 30,
    ) -> str:
        """t-SNE 可视化节点嵌入。"""
        # 采样以避免过慢
        max_samples = 5000
        if len(embeddings) > max_samples:
            idx = np.random.choice(len(embeddings), max_samples, replace=False)
            embeddings = embeddings[idx]
            labels = labels[idx]

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        emb_2d = tsne.fit_transform(embeddings)

        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["blue", "red"]
        for label in [0, 1]:
            mask = labels == label
            ax.scatter(
                emb_2d[mask, 0], emb_2d[mask, 1],
                c=colors[label], label=f"{'Normal' if label == 0 else 'Fraud'}",
                alpha=0.6, s=10,
            )

        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return self._save(name)

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        title: str = "Attention Heatmap",
        name: str = "attention_heatmap",
    ) -> str:
        """绘制注意力权重热图。"""
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(attention_weights, cmap="viridis", aspect="auto")
        ax.figure.colorbar(im, ax=ax, label="Attention Weight")
        ax.set_xlabel("Target Node", fontsize=12)
        ax.set_ylabel("Source Node", fontsize=12)
        ax.set_title(title, fontsize=14)
        return self._save(name)

    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "f1",
        title: str = "Experiment Comparison",
        name: str = "metric_comparison",
    ) -> str:
        """多实验指标对比柱状图。"""
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(results.keys())
        values = [results[n].get(metric, 0) for n in names]

        bars = ax.bar(names, values, color="steelblue", edgecolor="black", linewidth=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=9)

        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(metric.upper(), fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.grid(True, alpha=0.3, axis="y")
        fig.tight_layout()
        return self._save(name)

    def plot_sensitivity_curve(
        self,
        param_name: str,
        param_values: List,
        metrics_dict: Dict[str, List[float]],
        title: str = "Parameter Sensitivity",
        name: str = None,
    ) -> str:
        """参数敏感性曲线。"""
        if name is None:
            name = f"sensitivity_{param_name}"

        fig, ax = plt.subplots(figsize=(8, 5))
        for metric_name, values in metrics_dict.items():
            ax.plot(param_values, values, "o-", linewidth=2, label=metric_name.upper())

        ax.set_xlabel(param_name, fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        return self._save(name)

    def plot_model_graph(self, model, input_size: Tuple, name: str = "model_graph"):
        """绘制网络结构图（使用 torchview 或 hiddenlayer）。"""
        try:
            from torchview import draw_graph

            model_graph = draw_graph(model, input_size=input_size)
            path = os.path.join(self.save_dir, f"{name}.{self.fmt}")
            model_graph.render(path, format=self.fmt)
            return path
        except ImportError:
            print("[Visualizer] torchview not installed. Skip model graph plotting.")
            return ""

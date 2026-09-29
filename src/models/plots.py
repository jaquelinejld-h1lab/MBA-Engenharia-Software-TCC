"""Figures for the evidence folder. Pure consumers of already computed numbers."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless; must precede pyplot import
import matplotlib.pyplot as plt

from src.models.evaluate import RocPoints


def plot_roc(curves: dict[str, tuple[RocPoints, float]], path: Path, title: str) -> Path:
    """Draw one ROC curve per entry of ``curves`` (label -> (points, auc)).

    Parameters
    ----------
    curves : dict
        Label to ``(RocPoints, auc)``.
    path : Path
        Output PNG.
    title : str
        Figure title.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, (points, auc) in curves.items():
        ax.plot(points.fpr, points.tpr, lw=1.8, label=f"{label} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", lw=1, color="grey")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Taxa de verdadeiros positivos (sensibilidade)")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_confusion_matrix(tn: int, fp: int, fn: int, tp: int, path: Path, title: str) -> Path:
    """Draw a 2x2 confusion matrix with counts."""
    fig, ax = plt.subplots(figsize=(4.5, 4))
    matrix = [[tn, fp], [fn, tp]]
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i][j]), ha="center", va="center", fontsize=12)
    ax.set_xticks([0, 1], labels=["Predito 0", "Predito 1"])
    ax.set_yticks([0, 1], labels=["Real 0", "Real 1"])
    ax.set_title(title)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

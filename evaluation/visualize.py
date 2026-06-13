"""
Visualisation utilities for training analysis and results reporting.

All functions:
  - Accept a save_path argument and write a PNG there.
  - Return the matplotlib Figure so callers can embed or close it.
  - Use a non-interactive backend (Agg) safe for server-side generation.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

# Colour palette for the three modality modes
_MODE_COLORS = {"image": "#4C72B0", "text": "#DD8452", "fusion": "#55A868"}
_FIGSIZE_WIDE = (14, 5)
_DPI = 150


# ─────────────────────────────────────────────────────────────────────────────
# Training curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_curves(
    history: Dict[str, List],
    save_path: str = "results/training_curves.png",
) -> plt.Figure:
    """
    Three-panel figure: (a) Loss, (b) val mAP, (c) F1 micro/macro.

    Parameters
    ----------
    history : dict produced by Trainer.fit()['history']
    """
    epochs     = history["epoch"]
    fig, axes  = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Training History", fontsize=13, fontweight="bold")

    # Panel 1 – Loss
    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Train Loss", color="#E74C3C")
    ax.plot(epochs, history["val_loss"],   label="Val Loss",   color="#3498DB", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.set_title("Loss"); ax.legend(); ax.grid(alpha=0.3)

    # Panel 2 – mAP
    ax = axes[1]
    ax.plot(epochs, history["val_mAP"], color="#2ECC71", marker="o", markersize=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("mAP")
    ax.set_title("Validation mAP (primary metric)")
    ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    best_idx = int(np.argmax(history["val_mAP"]))
    ax.axvline(epochs[best_idx], color="grey", linestyle=":", alpha=0.8,
               label=f"Best: {history['val_mAP'][best_idx]:.4f}")
    ax.legend()

    # Panel 3 – F1
    ax = axes[2]
    ax.plot(epochs, history["val_f1_micro"], label="F1 Micro", color="#9B59B6")
    ax.plot(epochs, history["val_f1_macro"], label="F1 Macro", color="#E67E22", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("F1")
    ax.set_title("Validation F1"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info(f"Training curves → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Modality comparison (the core deliverable)
# ─────────────────────────────────────────────────────────────────────────────

def plot_modality_comparison(
    test_metrics: Dict[str, Dict],
    save_path: str = "results/modality_comparison.png",
) -> plt.Figure:
    """
    Grouped bar chart comparing image / text / fusion across 7 metrics.

    Parameters
    ----------
    test_metrics : {'image': metrics, 'text': metrics, 'fusion': metrics}
    """
    metric_keys = [
        ("mAP",            "mAP"),
        ("hamming_loss",   "Hamming Loss\n(lower=better)"),
        ("precision_at_1", "P@1"),
        ("precision_at_3", "P@3"),
        ("precision_at_5", "P@5"),
        ("f1_micro",       "F1 Micro"),
        ("f1_macro",       "F1 Macro"),
    ]
    modes   = ["image", "text", "fusion"]
    n_m     = len(metric_keys)
    x       = np.arange(n_m)
    width   = 0.25

    fig, ax = plt.subplots(figsize=(16, 5))
    for i, mode in enumerate(modes):
        vals = [test_metrics[mode].get(k, 0.0) for k, _ in metric_keys]
        bars = ax.bar(x + i * width, vals, width, label=mode.capitalize(),
                      color=_MODE_COLORS[mode], alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{v:.3f}",
                ha="center", va="bottom", fontsize=7, rotation=45,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([lbl for _, lbl in metric_keys], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(
        "Modality Comparison: Image-Only vs Text-Only vs Fusion",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

    plt.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info(f"Modality comparison → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Per-class F1 bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_class_f1(
    metrics: Dict,
    class_names: List[str],
    top_n: int = 20,
    save_path: str = "results/per_class_f1.png",
) -> plt.Figure:
    """
    Horizontal bar chart of per-class F1 for the top_n highest-support classes.

    Parameters
    ----------
    metrics     : metrics dict returned by compute_all_metrics()
    class_names : list of class name strings
    top_n       : how many classes to show (ranked by support)
    """
    pc      = metrics.get("per_class", {})
    records = [(name, pc[name]) for name in class_names if name in pc]

    # Sort by support descending, take top_n
    records.sort(key=lambda x: x[1]["support"], reverse=True)
    records = records[:top_n]
    records.sort(key=lambda x: x[1]["f1"])   # ascending for horizontal bar

    names    = [r[0].split("::", 1)[1] for r in records]
    f1_vals  = [r[1]["f1"] for r in records]
    ap_vals  = [r[1]["ap"] for r in records]
    supports = [r[1]["support"] for r in records]

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.35)))
    y_pos   = np.arange(len(names))
    ax.barh(y_pos, f1_vals, 0.4, label="F1", color="#4C72B0", alpha=0.85)
    ax.barh(y_pos + 0.4, ap_vals, 0.4, label="AP", color="#DD8452", alpha=0.85)

    ax.set_yticks(y_pos + 0.2)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.05)
    ax.set_title(f"Per-class F1 & AP — Top {top_n} by support (fusion mode)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    # Annotate support counts
    for i, (f1v, sup) in enumerate(zip(f1_vals, supports)):
        ax.text(0.01, y_pos[i], f"n={sup}", va="center", fontsize=6, color="white")

    plt.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info(f"Per-class F1 → {save_path}")
    plt.close(fig)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix (top 10 classes by support)
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix_top10(
    metrics: Dict,
    class_names: List[str],
    save_path: str = "results/confusion_matrix_top10.png",
) -> plt.Figure:
    """
    For the 10 highest-support classes, show a co-occurrence heatmap:
    rows = ground-truth positives, cols = what the model predicted.

    Each cell (i, j) contains the count of samples that are truly class i
    and are also predicted as class j.

    Parameters
    ----------
    metrics     : metrics dict from compute_all_metrics() (must contain _y_true, _y_pred)
    class_names : full list of class names
    """
    y_true = metrics.get("_y_true")
    y_pred = metrics.get("_y_pred")
    pc     = metrics.get("per_class", {})

    if y_true is None or y_pred is None:
        logger.warning("_y_true/_y_pred not in metrics; skipping confusion matrix.")
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center")
        fig.savefig(save_path, dpi=_DPI)
        plt.close(fig)
        return fig

    # Pick top 10 by support
    records = [(name, pc[name]["support"]) for name in class_names if name in pc]
    records.sort(key=lambda x: x[1], reverse=True)
    top10   = [r[0] for r in records[:10]]
    top10_idx = [class_names.index(n) for n in top10]

    short_labels = [n.split("::", 1)[1] for n in top10]

    # Co-occurrence: how often true class i is predicted as class j
    sub_true = y_true[:, top10_idx]   # (N, 10)
    sub_pred = y_pred[:, top10_idx]   # (N, 10)

    comat = np.zeros((10, 10), dtype=np.int32)
    for i in range(10):
        true_i_mask = sub_true[:, i] == 1   # rows where class i is truly positive
        for j in range(10):
            comat[i, j] = int(sub_pred[true_i_mask, j].sum())

    # Normalise per row (fraction of true class-i predicted as class-j)
    row_sum = comat.sum(axis=1, keepdims=True).clip(min=1)
    comat_norm = comat.astype(float) / row_sum

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        comat_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=short_labels, yticklabels=short_labels,
        linewidths=0.4, ax=ax, vmin=0, vmax=1,
    )
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("True class", fontsize=10)
    ax.set_title(
        "Co-occurrence heatmap — Top 10 classes (row-normalised, fusion mode)",
        fontsize=11, fontweight="bold",
    )
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=_DPI, bbox_inches="tight")
    logger.info(f"Confusion matrix → {save_path}")
    plt.close(fig)
    return fig

"""
Evaluation metrics for multi-label product tagging.

compute_all_metrics() is the single entry point consumed by the Trainer
and the final test-set evaluation loop.

Metrics computed
----------------
mAP             – mean Average Precision (macro, i.e. per-class AP averaged)
hamming_loss    – fraction of labels that are incorrectly predicted
precision_at_1  – fraction of samples where the top-1 predicted label is positive
precision_at_3  – mean precision in top-3 predictions per sample
precision_at_5  – mean precision in top-5 predictions per sample
f1_micro        – micro-averaged F1 (treats all class-sample pairs equally)
f1_macro        – macro-averaged F1 (unweighted mean over classes)
per-class data  – dict "class_<name>" with keys ap, precision, recall, f1, support
"""
from __future__ import annotations
from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    hamming_loss,
    precision_recall_fscore_support,
)


# ─────────────────────────────────────────────────────────────────────────────
# Precision @ K
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: int,
) -> float:
    """
    Mean Precision@K over all samples.

    For each sample, find the K labels with highest predicted probability,
    then compute what fraction of those are truly positive.

    Parameters
    ----------
    y_true : (N, C) binary ground-truth matrix
    y_prob : (N, C) predicted probability matrix (sigmoid outputs)
    k      : number of top predictions to consider

    Returns
    -------
    scalar float in [0, 1]
    """
    n_samples = y_true.shape[0]
    scores: List[float] = []
    for i in range(n_samples):
        top_k_idx = np.argsort(y_prob[i])[::-1][:k]
        hits = y_true[i, top_k_idx].sum()
        scores.append(hits / k)
    return float(np.mean(scores))


# ─────────────────────────────────────────────────────────────────────────────
# Per-class metrics
# ─────────────────────────────────────────────────────────────────────────────

def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
) -> Dict[str, Dict]:
    """
    Compute AP, precision, recall, F1, and support for every class.

    Returns
    -------
    dict keyed by class name with sub-dict {ap, precision, recall, f1, support}
    """
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    results: Dict[str, Dict] = {}
    for i, name in enumerate(class_names):
        try:
            ap = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            ap = 0.0
        results[name] = {
            "ap":        float(ap),
            "precision": float(precision[i]),
            "recall":    float(recall[i]),
            "f1":        float(f1[i]),
            "support":   int(support[i]),
        }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Compute the full suite of multi-label evaluation metrics.

    Parameters
    ----------
    y_true      : (N, C) binary ground-truth matrix  float32
    y_prob      : (N, C) sigmoid probability matrix  float32
    threshold   : decision threshold for binary predictions
    class_names : list of length C; if None, indices are used

    Returns
    -------
    Flat dict.  Per-class results are nested under key 'per_class'.
    All scalar values are Python floats (safe for JSON / MLflow).
    """
    if class_names is None:
        class_names = [str(i) for i in range(y_true.shape[1])]

    y_pred = (y_prob >= threshold).astype(np.float32)

    # ── Global metrics ────────────────────────────────────────────────────
    # mAP: compute per-class AP, average only over classes with ≥1 positive sample
    per_class_ap: List[float] = []
    for c in range(y_true.shape[1]):
        if y_true[:, c].sum() > 0:
            try:
                ap = average_precision_score(y_true[:, c], y_prob[:, c])
            except ValueError:
                ap = 0.0
            per_class_ap.append(ap)
    mAP = float(np.mean(per_class_ap)) if per_class_ap else 0.0

    hl = float(hamming_loss(y_true, y_pred))

    f1_micro = float(f1_score(y_true, y_pred, average="micro",  zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro",  zero_division=0))

    p1 = precision_at_k(y_true, y_prob, k=1)
    p3 = precision_at_k(y_true, y_prob, k=3)
    p5 = precision_at_k(y_true, y_prob, k=5)

    # Per-class accuracy (fraction correct for each binary sub-problem)
    per_class_acc = (y_pred == y_true).mean(axis=0)  # (C,)

    # ── Per-class detailed breakdown ──────────────────────────────────────
    pc_metrics = per_class_metrics(y_true, y_pred, y_prob, class_names)

    # Flatten per-class accuracy into the per-class dict
    for i, name in enumerate(class_names):
        pc_metrics[name]["accuracy"] = float(per_class_acc[i])

    return {
        # ── Primary metric ─────────────────────────────────────────────
        "mAP":            mAP,
        # ── Secondary metrics ──────────────────────────────────────────
        "hamming_loss":   hl,
        "precision_at_1": p1,
        "precision_at_3": p3,
        "precision_at_5": p5,
        "f1_micro":       f1_micro,
        "f1_macro":       f1_macro,
        "mean_per_class_accuracy": float(per_class_acc.mean()),
        # ── Nested per-class breakdown ─────────────────────────────────
        "per_class":      pc_metrics,
        # ── Convenience arrays for plotting ───────────────────────────
        "_y_true":        y_true,
        "_y_prob":        y_prob,
        "_y_pred":        y_pred,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Formatted summary table (used by predict.py and train.py CLI output)
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_table(metrics_by_mode: Dict[str, Dict]) -> str:
    """
    Build a side-by-side comparison table string for the three inference modes.

    Parameters
    ----------
    metrics_by_mode : {'image': metrics_dict, 'text': metrics_dict, 'fusion': metrics_dict}

    Returns
    -------
    A multi-line ASCII table string.
    """
    keys = [
        ("mAP",            "mAP"),
        ("hamming_loss",   "Hamming Loss"),
        ("precision_at_1", "Precision@1"),
        ("precision_at_3", "Precision@3"),
        ("precision_at_5", "Precision@5"),
        ("f1_micro",       "F1 Micro"),
        ("f1_macro",       "F1 Macro"),
        ("mean_per_class_accuracy", "Mean Class Acc"),
    ]
    modes = list(metrics_by_mode.keys())
    col_w = 13
    header_row = f"{'Metric':<22}" + "".join(f"{m:>{col_w}}" for m in modes)
    sep = "-" * (22 + col_w * len(modes))
    rows = [header_row, sep]
    for key, label in keys:
        row = f"{label:<22}"
        for m in modes:
            val = metrics_by_mode[m].get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        rows.append(row)
    return "\n".join(rows)

"""
Loss functions for multi-label classification.

BCEWithLogitsLoss with pos_weight handles severe class imbalance by
up-weighting rare positive examples:

    pos_weight[c] = (# negative samples for class c) / (# positive samples for class c)

This means rare classes contribute proportionally more to the loss,
preventing the model from predicting everything as the majority negative.
"""
from __future__ import annotations
import logging

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_pos_weight(Y_train: np.ndarray, clip_max: float = 50.0) -> torch.Tensor:
    """
    Compute per-class positive weights for BCEWithLogitsLoss.

    Parameters
    ----------
    Y_train  : binary label matrix (N_train, num_classes) float32
    clip_max : cap the weight to avoid extreme values for ultra-rare classes

    Returns
    -------
    Tensor of shape (num_classes,) on CPU.
    """
    pos = Y_train.sum(axis=0).astype(np.float32)            # [num_classes]
    neg = (Y_train.shape[0] - pos).astype(np.float32)

    # For classes with 0 positive samples set weight=1 (won't affect gradient)
    pos_safe = np.where(pos > 0, pos, 1.0)
    weight   = neg / pos_safe
    weight   = np.clip(weight, 1.0, clip_max)

    logger.info(
        f"pos_weight — min={weight.min():.2f}, "
        f"mean={weight.mean():.2f}, max={weight.max():.2f}"
    )
    return torch.tensor(weight, dtype=torch.float32)


def build_criterion(
    Y_train: np.ndarray,
    device: torch.device,
    clip_max: float = 50.0,
) -> nn.BCEWithLogitsLoss:
    """
    Build BCEWithLogitsLoss with computed class-imbalance weights.

    Parameters
    ----------
    Y_train  : binary label matrix for the training split
    device   : target device for the weight tensor
    clip_max : maximum allowable positive weight

    Returns
    -------
    Configured BCEWithLogitsLoss instance (reduction='mean').
    """
    pos_weight = compute_pos_weight(Y_train, clip_max=clip_max).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="mean")
    logger.info("BCEWithLogitsLoss with pos_weight ready.")
    return criterion

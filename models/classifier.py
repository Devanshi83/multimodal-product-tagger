"""
ClassificationHead
==================
Shared MLP head used by all three inference modes.

  Linear(embed_dim → hidden_dim) → ReLU → Dropout → Linear(hidden_dim → num_classes)

Output is raw logits (sigmoid applied externally for multi-label prediction).
"""
from __future__ import annotations
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Two-layer MLP producing per-class logits for multi-label classification.

    Parameters
    ----------
    input_dim  : Feature dimension coming in (embed_dim, typically 1024).
    hidden_dim : Intermediate projection size (512 per spec).
    num_classes: Total number of binary output labels (derived from dataset).
    dropout    : Dropout probability (0.3 per spec).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [B, input_dim]

        Returns
        -------
        logits : [B, num_classes]  (raw; apply sigmoid for probabilities)
        """
        return self.net(x)

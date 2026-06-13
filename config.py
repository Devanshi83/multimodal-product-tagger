"""
Central configuration for the Multi-Modal Product Tagging system.
All hyperparameters and paths live here so nothing is hardcoded elsewhere.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class Config:
    # ── Paths ──────────────────────────────────────────────────────────────
    data_dir: str = "data/raw"
    image_dir: str = "data/raw/images"
    styles_csv: str = "data/raw/styles.csv"
    checkpoint_dir: str = "checkpoints"
    results_dir: str = "results"

    # ── CLIP backbone ──────────────────────────────────────────────────────
    clip_model_name: str = "openai/clip-vit-large-patch14"
    # Target embedding dimension fed into the classifier head.
    # ViT-L/14 visual encoder → 1024-dim; text encoder → 768-dim (projected here to 1024).
    embed_dim: int = 1024

    # ── Classification head ────────────────────────────────────────────────
    hidden_dim: int = 512
    dropout: float = 0.3

    # ── Training schedule ─────────────────────────────────────────────────
    batch_size: int = 32
    num_workers: int = 4
    max_epochs: int = 20
    # Phase 1: epochs 1-phase1_epochs → CLIP frozen, head only
    phase1_epochs: int = 5
    # Phase 2: epochs (phase1_epochs+1)+ → unfreeze top N vision blocks
    n_unfreeze_blocks: int = 4

    # ── Optimiser ─────────────────────────────────────────────────────────
    lr: float = 1e-4
    weight_decay: float = 1e-4
    max_norm: float = 1.0   # Gradient clipping

    # ── Scheduler ─────────────────────────────────────────────────────────
    eta_min: float = 1e-6   # CosineAnnealingLR minimum lr

    # ── Early stopping ─────────────────────────────────────────────────────
    patience: int = 5       # Epochs without val-mAP improvement

    # ── Inference ─────────────────────────────────────────────────────────
    threshold: float = 0.5  # Sigmoid threshold for positive prediction

    # ── MLflow ────────────────────────────────────────────────────────────
    mlflow_experiment: str = "multimodal-product-tagger"
    mlflow_tracking_uri: str = "mlruns"

    # ── Data splits ────────────────────────────────────────────────────────
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # ── Multi-label target columns ─────────────────────────────────────────
    target_columns: List[str] = field(
        default_factory=lambda: ["masterCategory", "subCategory", "articleType"]
    )

    # ── Mixed precision ────────────────────────────────────────────────────
    use_amp: bool = True

    # ── Reproducibility ───────────────────────────────────────────────────
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        valid = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**valid)


# Module-level singleton — import this everywhere.
CFG = Config()

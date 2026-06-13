#!/usr/bin/env python3
"""
train.py  –  Entry point for training the Multi-Modal Product Tagger
=====================================================================

Usage
-----
    python train.py                          # use defaults from config.py
    python train.py --batch-size 16          # override batch size
    python train.py --max-epochs 10          # quick test run
    python train.py --no-amp                 # disable mixed precision
    python train.py --run-name my_experiment # custom MLflow run name

The script:
  1. Seeds all RNGs for reproducibility.
  2. Loads and validates styles.csv.
  3. Builds binary label matrix (num_classes derived from data — never hardcoded).
  4. Iteratively stratifies into 70 / 15 / 15 splits.
  5. Creates FashionDataset with CLIP preprocessing + augmentation.
  6. Builds MultiModalTagger (CLIPWrapper + FusionLayer + ClassificationHead).
  7. Constructs BCEWithLogitsLoss with per-class pos_weight.
  8. Trains with two-phase CLIP unfreezing and early stopping.
  9. Evaluates on the test set in all three modality modes (image / text / fusion).
  10. Prints a side-by-side comparison table and logs everything to MLflow.
"""
from __future__ import annotations
import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import CLIPProcessor

# ── Project imports ────────────────────────────────────────────────────────
from config import Config, CFG
from data.splits import load_and_prepare_metadata, iterative_train_val_test_split, build_label_matrix
from data.dataset import FashionDataset
from models.fusion import build_model
from training.losses import build_criterion
from training.trainer import Trainer
from evaluation.metrics import format_metrics_table
from evaluation.visualize import plot_confusion_matrix_top10

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Seeding
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing (override config defaults from CLI)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Multi-Modal Product Tagger.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--styles-csv",     type=str,   default=CFG.styles_csv)
    parser.add_argument("--image-dir",      type=str,   default=CFG.image_dir)
    parser.add_argument("--checkpoint-dir", type=str,   default=CFG.checkpoint_dir)
    parser.add_argument("--results-dir",    type=str,   default=CFG.results_dir)
    parser.add_argument("--batch-size",     type=int,   default=CFG.batch_size)
    parser.add_argument("--num-workers",    type=int,   default=CFG.num_workers)
    parser.add_argument("--max-epochs",     type=int,   default=CFG.max_epochs)
    parser.add_argument("--phase1-epochs",  type=int,   default=CFG.phase1_epochs)
    parser.add_argument("--lr",             type=float, default=CFG.lr)
    parser.add_argument("--weight-decay",   type=float, default=CFG.weight_decay)
    parser.add_argument("--patience",       type=int,   default=CFG.patience)
    parser.add_argument("--threshold",      type=float, default=CFG.threshold)
    parser.add_argument("--seed",           type=int,   default=CFG.seed)
    parser.add_argument("--no-amp",         action="store_true",
                        help="Disable automatic mixed precision (AMP).")
    parser.add_argument("--run-name",       type=str,   default=None,
                        help="Custom MLflow run name.")
    parser.add_argument("--device",         type=str,   default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted).")
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # ── Apply CLI overrides to config ──────────────────────────────────────
    cfg = Config(
        styles_csv       = args.styles_csv,
        image_dir        = args.image_dir,
        checkpoint_dir   = args.checkpoint_dir,
        results_dir      = args.results_dir,
        batch_size       = args.batch_size,
        num_workers      = args.num_workers,
        max_epochs       = args.max_epochs,
        phase1_epochs    = args.phase1_epochs,
        lr               = args.lr,
        weight_decay     = args.weight_decay,
        patience         = args.patience,
        threshold        = args.threshold,
        seed             = args.seed,
        use_amp          = not args.no_amp,
    )

    seed_everything(cfg.seed)

    # ── Device ────────────────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.info(f"Device: {device}")
    logger.info(f"AMP enabled: {cfg.use_amp}")

    # ── Validate paths ─────────────────────────────────────────────────────
    if not Path(cfg.styles_csv).exists():
        logger.error(
            f"styles.csv not found at '{cfg.styles_csv}'.\n"
            "Download the dataset first:\n"
            "    python download_data.py\n"
            "or follow the README instructions."
        )
        sys.exit(1)

    if not Path(cfg.image_dir).exists():
        logger.error(
            f"Image directory not found at '{cfg.image_dir}'.\n"
            "Ensure the Kaggle dataset was extracted correctly."
        )
        sys.exit(1)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Load metadata & build label encoder ───────────────────────────────
    logger.info("Loading metadata …")
    df, encoder, num_classes, class_names = load_and_prepare_metadata(
        cfg.styles_csv, target_columns=cfg.target_columns
    )
    logger.info(f"num_classes = {num_classes}  (derived from data, not hardcoded)")

    # ── Build full label matrix then split ────────────────────────────────
    logger.info("Building label matrix …")
    Y = build_label_matrix(df, encoder)   # (N, num_classes)

    logger.info("Running iterative stratified split (70 / 15 / 15) …")
    train_df, val_df, test_df, Y_train, Y_val, Y_test = \
        iterative_train_val_test_split(
            df, Y,
            train_ratio=cfg.train_ratio,
            val_ratio=cfg.val_ratio,
            test_ratio=cfg.test_ratio,
        )

    # ── CLIP processor (shared across splits) ─────────────────────────────
    logger.info(f"Loading CLIPProcessor ({cfg.clip_model_name}) …")
    processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

    # ── DataLoaders ────────────────────────────────────────────────────────
    logger.info("Building DataLoaders …")
    train_ds = FashionDataset(
        train_df, cfg.image_dir, encoder, cfg.target_columns,
        processor, augment=True,  label_matrix=Y_train,
    )
    val_ds = FashionDataset(
        val_df,   cfg.image_dir, encoder, cfg.target_columns,
        processor, augment=False, label_matrix=Y_val,
    )
    test_ds = FashionDataset(
        test_df,  cfg.image_dir, encoder, cfg.target_columns,
        processor, augment=False, label_matrix=Y_test,
    )

    _loader_kwargs = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(cfg.num_workers > 0),
    )
    train_loader = DataLoader(train_ds, shuffle=True,  **_loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **_loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **_loader_kwargs)

    logger.info(
        f"DataLoaders ready — "
        f"train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    logger.info("Building MultiModalTagger …")
    model = build_model(
        clip_model_name=cfg.clip_model_name,
        embed_dim=cfg.embed_dim,
        hidden_dim=cfg.hidden_dim,
        num_classes=num_classes,
        dropout=cfg.dropout,
    )
    params = model.count_parameters()
    logger.info(
        f"Model parameters — "
        f"total={params['total']:,}, "
        f"trainable (initial)={params['trainable']:,}, "
        f"frozen={params['frozen']:,}"
    )

    # ── Loss ──────────────────────────────────────────────────────────────
    logger.info("Building BCEWithLogitsLoss with pos_weight …")
    criterion = build_criterion(Y_train, device=device)

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        cfg=cfg,
        device=device,
        class_names=class_names,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Starting training …")
    logger.info("=" * 60)

    output = trainer.fit(
        test_loader=test_loader,
        encoder=encoder,
        run_name=args.run_name,
    )

    # ── Final results table ───────────────────────────────────────────────
    test_metrics = output["test_metrics"]
    table = format_metrics_table(test_metrics)
    logger.info("\n\nFINAL TEST-SET RESULTS (best checkpoint)\n")
    logger.info("=" * 60)
    print(table)
    logger.info("=" * 60)

    # ── Confusion matrix ──────────────────────────────────────────────────
    try:
        cm_path = os.path.join(cfg.results_dir, "confusion_matrix_top10.png")
        plot_confusion_matrix_top10(
            test_metrics["fusion"],
            class_names=class_names,
            save_path=cm_path,
        )
        logger.info(f"Confusion matrix saved to {cm_path}")
    except Exception as exc:
        logger.warning(f"Confusion matrix plot failed (non-fatal): {exc}")

    logger.info(
        f"\nBest checkpoint: checkpoints/best_model.pt\n"
        f"MLflow UI:        mlflow ui --backend-store-uri {cfg.mlflow_tracking_uri}\n"
        f"API server:       uvicorn api.main:app --host 0.0.0.0 --port 8000\n"
        f"Inference CLI:    python predict.py --image <img> --text '<description>'\n"
    )


if __name__ == "__main__":
    main()

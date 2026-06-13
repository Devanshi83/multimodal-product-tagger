"""
Trainer
=======
Orchestrates two-phase training, validation, early stopping, checkpointing,
and MLflow experiment tracking.

Phase 1 (epochs 1 – phase1_epochs)
    CLIP backbone fully frozen. Only the FusionLayer, ClassificationHead, and
    CLIPWrapper.text_proj are updated. Optimiser built on requires_grad params.

Phase 2 (epoch phase1_epochs + 1 onwards)
    Top n_unfreeze_blocks vision transformer blocks + post_layernorm unfrozen.
    A fresh optimiser is created so Adam moments start clean for newly unlocked
    params. Cosine schedule restarts with T_max = remaining epochs.

MLflow logs:
    Per epoch  — train_loss, val_loss, val_mAP, val_f1_micro, val_f1_macro,
                 val_hamming_loss, lr
    After fit  — test_<mode>_mAP / _f1_micro / _f1_macro / _hamming for each
                 of image / text / fusion modes.
    Artifacts  — training curves PNG, modality comparison PNG,
                 confusion-matrix PNG.
"""
from __future__ import annotations
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from evaluation.metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class Trainer:
    """
    Full training / evaluation orchestrator.

    Parameters
    ----------
    model        : MultiModalTagger
    train_loader : DataLoader for training split
    val_loader   : DataLoader for validation split
    criterion    : BCEWithLogitsLoss (with pos_weight)
    cfg          : Config instance
    device       : torch.device
    class_names  : list of label strings (length == num_classes)
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        cfg: Config,
        device: torch.device,
        class_names: List[str],
    ) -> None:
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.cfg          = cfg
        self.device       = device
        self.class_names  = class_names

        self.best_val_mAP: float  = 0.0
        self.patience_counter: int = 0
        self.current_phase: int    = 0  # 0 = uninitialised

        # AMP scaler (no-op on CPU / MPS)
        self._amp_enabled = cfg.use_amp and device.type == "cuda"
        self.scaler = GradScaler(enabled=self._amp_enabled)

        # Will be set in _enter_phase()
        self.optimizer: Optional[AdamW]               = None
        self.scheduler: Optional[CosineAnnealingLR]   = None

        Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.results_dir).mkdir(parents=True, exist_ok=True)

    # ── Phase management ──────────────────────────────────────────────────

    def _trainable_params(self) -> List[nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _enter_phase1(self) -> None:
        """Freeze CLIP backbone and build Phase-1 optimiser."""
        self.model.clip.freeze_clip()
        self.optimizer = AdamW(
            self._trainable_params(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.cfg.phase1_epochs,
            eta_min=self.cfg.eta_min,
        )
        self.current_phase = 1
        n = sum(p.numel() for p in self._trainable_params())
        logger.info(f"[Phase 1] {n:,} trainable params — CLIP frozen.")

    def _enter_phase2(self) -> None:
        """Unfreeze top vision blocks and build fresh Phase-2 optimiser."""
        self.model.clip.unfreeze_top_vision_blocks(self.cfg.n_unfreeze_blocks)
        remaining = max(1, self.cfg.max_epochs - self.cfg.phase1_epochs)
        self.optimizer = AdamW(
            self._trainable_params(),
            lr=self.cfg.lr * 0.1,   # Lower LR when fine-tuning backbone
            weight_decay=self.cfg.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=remaining,
            eta_min=self.cfg.eta_min,
        )
        self.current_phase = 2
        n = sum(p.numel() for p in self._trainable_params())
        logger.info(
            f"[Phase 2] {n:,} trainable params — "
            f"top {self.cfg.n_unfreeze_blocks} vision blocks unfrozen."
        )

    # ── Single epoch ──────────────────────────────────────────────────────

    def _train_epoch(self, epoch: int) -> float:
        """Run one training epoch (always fusion mode). Returns mean loss."""
        self.model.train()
        running_loss = 0.0
        n_batches    = len(self.train_loader)

        pbar = tqdm(
            enumerate(self.train_loader),
            total=n_batches,
            desc=f"Epoch {epoch:03d} [train]",
            leave=False,
        )

        for step, batch in pbar:
            pv  = batch["pixel_values"].to(self.device, non_blocking=True)
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            msk = batch["attention_mask"].to(self.device, non_blocking=True)
            lbl = batch["labels"].to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self._amp_enabled):
                logits = self.model(pv, ids, msk, mode="fusion")
                loss   = self.criterion(logits, lbl)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.scheduler.step()
        return running_loss / n_batches

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(
        self,
        loader: DataLoader,
        mode: str = "fusion",
    ) -> Dict[str, float]:
        """
        Evaluate the model on `loader` in the given inference mode.

        Returns a dict with: mAP, hamming_loss, f1_micro, f1_macro,
        precision_at_1, precision_at_3, precision_at_5, loss,
        plus per-class scores (keys prefixed with 'class_').
        """
        self.model.eval()
        all_logits: List[torch.Tensor] = []
        all_labels: List[torch.Tensor] = []
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"  eval [{mode}]", leave=False):
            pv  = batch["pixel_values"].to(self.device, non_blocking=True)
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            msk = batch["attention_mask"].to(self.device, non_blocking=True)
            lbl = batch["labels"].to(self.device, non_blocking=True)

            with autocast(enabled=self._amp_enabled):
                logits = self.model(pv, ids, msk, mode=mode)
                loss   = self.criterion(logits, lbl)

            total_loss += loss.item()
            all_logits.append(logits.cpu().float())
            all_labels.append(lbl.cpu().float())

        logits_all = torch.cat(all_logits).numpy()
        labels_all = torch.cat(all_labels).numpy()
        probs_all  = 1.0 / (1.0 + np.exp(-logits_all))  # sigmoid

        metrics = compute_all_metrics(
            y_true=labels_all,
            y_prob=probs_all,
            threshold=self.cfg.threshold,
            class_names=self.class_names,
        )
        metrics["loss"] = total_loss / len(loader)
        return metrics

    # ── Checkpoint I/O ────────────────────────────────────────────────────

    def _save_checkpoint(
        self,
        epoch: int,
        val_mAP: float,
        encoder,
        suffix: str = "best",
    ) -> str:
        path = os.path.join(self.cfg.checkpoint_dir, f"model_{suffix}.pt")
        torch.save(
            {
                "epoch":              epoch,
                "val_mAP":            val_mAP,
                "model_state_dict":   self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config":             self.cfg.to_dict(),
                "class_names":        self.class_names,
                "num_classes":        len(self.class_names),
                "encoder":            encoder,
            },
            path,
        )
        logger.info(f"Checkpoint saved → {path}  (val_mAP={val_mAP:.4f})")
        return path

    # ── Main fit loop ─────────────────────────────────────────────────────

    def fit(
        self,
        test_loader: DataLoader,
        encoder,
        run_name: Optional[str] = None,
    ) -> Dict:
        """
        Full training loop with two-phase CLIP unfreezing, early stopping,
        and MLflow tracking.

        Parameters
        ----------
        test_loader : DataLoader for held-out test split
        encoder     : fitted OneHotEncoder (saved into checkpoint)
        run_name    : optional MLflow run name

        Returns
        -------
        dict with 'history' (per-epoch metrics) and 'test_metrics' (3 modes).
        """
        mlflow.set_tracking_uri(self.cfg.mlflow_tracking_uri)
        mlflow.set_experiment(self.cfg.mlflow_experiment)

        history: Dict[str, List] = {
            k: [] for k in ["epoch", "train_loss", "val_loss", "val_mAP",
                            "val_f1_micro", "val_f1_macro", "val_hamming", "lr"]
        }

        with mlflow.start_run(run_name=run_name or f"run_{int(time.time())}"):
            # ── Log hyperparameters ──────────────────────────────────────
            mlflow.log_params({
                "clip_model":        self.cfg.clip_model_name,
                "embed_dim":         self.cfg.embed_dim,
                "hidden_dim":        self.cfg.hidden_dim,
                "dropout":           self.cfg.dropout,
                "batch_size":        self.cfg.batch_size,
                "max_epochs":        self.cfg.max_epochs,
                "phase1_epochs":     self.cfg.phase1_epochs,
                "n_unfreeze_blocks": self.cfg.n_unfreeze_blocks,
                "lr":                self.cfg.lr,
                "weight_decay":      self.cfg.weight_decay,
                "max_norm":          self.cfg.max_norm,
                "patience":          self.cfg.patience,
                "num_classes":       len(self.class_names),
                "use_amp":           self.cfg.use_amp,
                "seed":              self.cfg.seed,
            })

            # ── Enter Phase 1 ────────────────────────────────────────────
            self._enter_phase1()

            best_ckpt_path: Optional[str] = None

            for epoch in range(1, self.cfg.max_epochs + 1):
                # Phase transition
                if epoch == self.cfg.phase1_epochs + 1:
                    self._enter_phase2()

                t0 = time.time()
                train_loss = self._train_epoch(epoch)
                val_metrics = self.evaluate(self.val_loader, mode="fusion")
                elapsed = time.time() - t0

                val_mAP = val_metrics["mAP"]
                cur_lr  = self.optimizer.param_groups[0]["lr"]

                # ── Per-epoch MLflow metrics ─────────────────────────────
                mlflow.log_metrics(
                    {
                        "train_loss":   train_loss,
                        "val_loss":     val_metrics["loss"],
                        "val_mAP":      val_mAP,
                        "val_f1_micro": val_metrics["f1_micro"],
                        "val_f1_macro": val_metrics["f1_macro"],
                        "val_hamming":  val_metrics["hamming_loss"],
                        "lr":           cur_lr,
                        "epoch_time_s": elapsed,
                    },
                    step=epoch,
                )

                # Persist history
                history["epoch"].append(epoch)
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_metrics["loss"])
                history["val_mAP"].append(val_mAP)
                history["val_f1_micro"].append(val_metrics["f1_micro"])
                history["val_f1_macro"].append(val_metrics["f1_macro"])
                history["val_hamming"].append(val_metrics["hamming_loss"])
                history["lr"].append(cur_lr)

                logger.info(
                    f"Epoch {epoch:03d}/{self.cfg.max_epochs} | "
                    f"phase={self.current_phase} | "
                    f"train_loss={train_loss:.4f} | "
                    f"val_mAP={val_mAP:.4f} | "
                    f"val_f1_micro={val_metrics['f1_micro']:.4f} | "
                    f"lr={cur_lr:.2e} | {elapsed:.1f}s"
                )

                # ── Early stopping ───────────────────────────────────────
                if val_mAP > self.best_val_mAP + 1e-4:
                    self.best_val_mAP  = val_mAP
                    self.patience_counter = 0
                    best_ckpt_path = self._save_checkpoint(epoch, val_mAP, encoder)
                    mlflow.log_metric("best_val_mAP", val_mAP, step=epoch)
                else:
                    self.patience_counter += 1
                    logger.info(
                        f"  No improvement ({self.patience_counter}/{self.cfg.patience})"
                    )
                    if self.patience_counter >= self.cfg.patience:
                        logger.info(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(best val_mAP={self.best_val_mAP:.4f})."
                        )
                        break

            # ── Load best checkpoint for final evaluation ────────────────
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                ckpt = torch.load(best_ckpt_path, map_location=self.device, weights_only=False)
                self.model.load_state_dict(ckpt["model_state_dict"])
                logger.info(
                    f"Loaded best checkpoint (epoch {ckpt['epoch']}, "
                    f"val_mAP={ckpt['val_mAP']:.4f}) for test evaluation."
                )

            # ── Evaluate all three modality modes on test set ────────────
            test_metrics: Dict[str, Dict] = {}
            for mode in ("image", "text", "fusion"):
                logger.info(f"Test evaluation — mode='{mode}' …")
                m = self.evaluate(test_loader, mode=mode)
                test_metrics[mode] = m
                flat = {f"test_{mode}_{k}": v
                        for k, v in m.items()
                        if isinstance(v, (int, float)) and not k.startswith("class_")}
                mlflow.log_metrics(flat)
                logger.info(
                    f"  [{mode}] mAP={m['mAP']:.4f} | "
                    f"f1_micro={m['f1_micro']:.4f} | "
                    f"hamming={m['hamming_loss']:.4f}"
                )

            # ── Generate and log plots ────────────────────────────────────
            try:
                from evaluation.visualize import (
                    plot_training_curves,
                    plot_modality_comparison,
                    plot_per_class_f1,
                )
                curves_path = os.path.join(self.cfg.results_dir, "training_curves.png")
                plot_training_curves(history, save_path=curves_path)
                mlflow.log_artifact(curves_path)

                comp_path = os.path.join(self.cfg.results_dir, "modality_comparison.png")
                plot_modality_comparison(test_metrics, save_path=comp_path)
                mlflow.log_artifact(comp_path)

                perclass_path = os.path.join(self.cfg.results_dir, "per_class_f1.png")
                plot_per_class_f1(
                    test_metrics["fusion"],
                    class_names=self.class_names,
                    top_n=20,
                    save_path=perclass_path,
                )
                mlflow.log_artifact(perclass_path)

            except Exception as exc:
                logger.warning(f"Plot generation failed (non-fatal): {exc}")

            if best_ckpt_path:
                mlflow.log_artifact(best_ckpt_path)

        return {"history": history, "test_metrics": test_metrics}

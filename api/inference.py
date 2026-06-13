"""
InferencePipeline
=================
Loads a trained MultiModalTagger from a checkpoint and provides a clean
`predict()` method used by both the FastAPI server and predict.py CLI.

Usage
-----
    pipeline = InferencePipeline("checkpoints/best_model.pt")
    results  = pipeline.predict(image_pil, text="blue jeans", mode="fusion")
"""
from __future__ import annotations
import base64
import io
import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor

from config import Config
from models.fusion import build_model

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Self-contained inference pipeline loaded from a .pt checkpoint.

    Parameters
    ----------
    checkpoint_path : path to a checkpoint saved by Trainer._save_checkpoint()
    device          : torch.device or str ('cuda', 'cpu'); auto-selects if None
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        logger.info(f"Loading checkpoint from {checkpoint_path} on {self.device}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # ── Rebuild config ────────────────────────────────────────────────
        cfg = Config.from_dict(ckpt["config"])
        self.cfg = cfg

        # ── Rebuild model ─────────────────────────────────────────────────
        num_classes = ckpt["num_classes"]
        self.model  = build_model(
            clip_model_name=cfg.clip_model_name,
            embed_dim=cfg.embed_dim,
            hidden_dim=cfg.hidden_dim,
            num_classes=num_classes,
            dropout=cfg.dropout,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        # ── Class metadata ────────────────────────────────────────────────
        self.class_names: List[str] = ckpt["class_names"]
        self.num_classes            = num_classes
        self.encoder                = ckpt["encoder"]   # sklearn OneHotEncoder

        # Build reverse lookup: class name → (category, label)
        self._class_meta: List[Dict[str, str]] = []
        for name in self.class_names:
            cat, lbl = name.split("::", 1)
            self._class_meta.append({"category": cat, "label": lbl})

        # Group class names by category for /classes endpoint
        self.categories: Dict[str, List[str]] = {}
        for meta in self._class_meta:
            self.categories.setdefault(meta["category"], []).append(meta["label"])

        # ── CLIP processor ────────────────────────────────────────────────
        self.processor = CLIPProcessor.from_pretrained(cfg.clip_model_name)

        logger.info(
            f"InferencePipeline ready | classes={num_classes} | device={self.device}"
        )

    # ── Core prediction ───────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        text: Optional[str] = None,
        mode: str = "fusion",
        threshold: Optional[float] = None,
    ) -> List[Dict]:
        """
        Run inference and return predicted tags above threshold.

        Parameters
        ----------
        image     : PIL Image (any mode; converted to RGB internally)
        text      : product title / description (use '' or None for image-only)
        mode      : 'image' | 'text' | 'fusion'
        threshold : override the checkpoint default if provided

        Returns
        -------
        List of dicts sorted by probability descending:
            [{'label': str, 'category': str, 'probability': float}, ...]
        """
        thr = threshold if threshold is not None else self.cfg.threshold

        # Auto-adjust mode when text is missing
        if mode in ("text", "fusion") and not text:
            logger.warning(
                "mode='%s' requested but no text provided; falling back to 'image'.",
                mode,
            )
            mode = "image"

        image_rgb = image.convert("RGB")
        text_str  = text or "fashion product"

        inputs = self.processor(
            text=[text_str],
            images=[image_rgb],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        )
        pv  = inputs["pixel_values"].to(self.device)
        ids = inputs["input_ids"].to(self.device)
        msk = inputs["attention_mask"].to(self.device)

        logits = self.model(pv, ids, msk, mode=mode)              # [1, C]
        probs  = torch.sigmoid(logits).squeeze(0).cpu().numpy()   # (C,)

        results: List[Dict] = []
        for i, (prob, meta) in enumerate(zip(probs, self._class_meta)):
            if float(prob) >= thr:
                results.append({
                    "label":       meta["label"],
                    "category":    meta["category"],
                    "probability": round(float(prob), 4),
                })

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results

    # ── Batch prediction ──────────────────────────────────────────────────

    @torch.no_grad()
    def predict_batch(
        self,
        items: List[Dict],
    ) -> List[List[Dict]]:
        """
        Batch inference for multiple items.

        Parameters
        ----------
        items : list of {'image': PIL.Image, 'text': str|None, 'mode': str, 'threshold': float}

        Returns
        -------
        List of prediction lists (one per item).
        """
        if not items:
            return []

        # Collect tensors
        all_pv, all_ids, all_msk, modes, thresholds = [], [], [], [], []
        fallback_texts: List[str] = []

        for it in items:
            image     = it["image"].convert("RGB")
            text      = it.get("text") or "fashion product"
            mode      = it.get("mode", "fusion")
            threshold = it.get("threshold", self.cfg.threshold)

            inp = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True,
            )
            all_pv.append(inp["pixel_values"])
            all_ids.append(inp["input_ids"])
            all_msk.append(inp["attention_mask"])
            modes.append(mode)
            thresholds.append(threshold)
            fallback_texts.append(text)

        pv  = torch.cat(all_pv,  dim=0).to(self.device)
        ids = torch.cat(all_ids, dim=0).to(self.device)
        msk = torch.cat(all_msk, dim=0).to(self.device)

        # Group by mode for efficiency (avoid separate forward per mode)
        # Simpler: forward each item individually (batch sizes up to 64 are fine)
        results_list: List[List[Dict]] = []
        for i in range(len(items)):
            logits = self.model(
                pv[i:i+1], ids[i:i+1], msk[i:i+1], mode=modes[i]
            )
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            item_results = []
            for j, (prob, meta) in enumerate(zip(probs, self._class_meta)):
                if float(prob) >= thresholds[i]:
                    item_results.append({
                        "label":       meta["label"],
                        "category":    meta["category"],
                        "probability": round(float(prob), 4),
                    })
            item_results.sort(key=lambda x: x["probability"], reverse=True)
            results_list.append(item_results)

        return results_list

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def decode_base64_image(b64_str: str) -> Image.Image:
        """Decode a base64-encoded image string to a PIL Image."""
        if "," in b64_str:   # strip data-URI prefix if present
            b64_str = b64_str.split(",", 1)[1]
        raw = base64.b64decode(b64_str)
        return Image.open(io.BytesIO(raw))

    def get_all_probabilities(
        self,
        image: Image.Image,
        text: Optional[str] = None,
        mode: str = "fusion",
    ) -> np.ndarray:
        """
        Return the full probability vector (C,) without threshold filtering.
        Useful for analysis and notebooks.
        """
        text_str  = text or "fashion product"
        image_rgb = image.convert("RGB")
        inputs = self.processor(
            text=[text_str], images=[image_rgb],
            return_tensors="pt", padding="max_length",
            max_length=77, truncation=True,
        )
        pv  = inputs["pixel_values"].to(self.device)
        ids = inputs["input_ids"].to(self.device)
        msk = inputs["attention_mask"].to(self.device)
        logits = self.model(pv, ids, msk, mode=mode)
        return torch.sigmoid(logits).squeeze(0).cpu().numpy()

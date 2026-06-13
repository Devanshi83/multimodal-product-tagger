"""
FashionDataset: PyTorch Dataset for the Fashion Product Images dataset.

Each item returns:
    pixel_values   – [3, 224, 224] float32 tensor ready for CLIP
    input_ids      – [max_text_len]  int64 token IDs
    attention_mask – [max_text_len]  int64 mask
    labels         – [num_classes]   float32 binary label vector
    product_id     – str (for tracing)
    text           – str  (the raw text used)
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPProcessor

logger = logging.getLogger(__name__)

MAX_TEXT_LEN = 77  # CLIP's context window


class FashionDataset(Dataset):
    """
    Multi-label fashion product dataset backed by the Kaggle
    imsparsh/fashion-product-images-dataset.

    Parameters
    ----------
    metadata       : DataFrame produced by splits.py
    image_dir      : Folder containing <id>.jpg files
    encoder        : Fitted OneHotEncoder (from splits.load_and_prepare_metadata)
    target_columns : Column names used as label sources
    clip_processor : Instantiated CLIPProcessor
    augment        : Apply training-time augmentation when True
    label_matrix   : Optional pre-computed binary label matrix (N, num_classes).
                     If provided, __getitem__ uses it directly instead of
                     re-encoding every call (faster for large datasets).
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        image_dir: str,
        encoder: OneHotEncoder,
        target_columns: List[str],
        clip_processor: CLIPProcessor,
        augment: bool = False,
        label_matrix: Optional[np.ndarray] = None,
    ) -> None:
        self.metadata       = metadata.reset_index(drop=True)
        self.image_dir      = Path(image_dir)
        self.encoder        = encoder
        self.target_columns = target_columns
        self.clip_processor = clip_processor
        self.augment        = augment

        # Pre-filter to images that actually exist on disk
        self._valid_mask = self._compute_valid_mask()
        self.metadata    = self.metadata[self._valid_mask].reset_index(drop=True)

        # Align pre-computed label matrix with filtered metadata
        if label_matrix is not None:
            self._label_matrix = label_matrix[self._valid_mask]
        else:
            self._label_matrix = None

        # Augmentation pipeline (images already resized by CLIP processor)
        if augment:
            self._aug = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                       saturation=0.2, hue=0.05),
                transforms.RandomRotation(15),
            ])
        else:
            self._aug = None

        logger.info(
            f"FashionDataset ready: {len(self.metadata)} samples "
            f"(augment={augment})"
        )

    # ── Internal helpers ──────────────────────────────────────────────────

    def _compute_valid_mask(self) -> np.ndarray:
        """Return boolean mask for rows that have an image file on disk."""
        mask = []
        for _, row in self.metadata.iterrows():
            path = self.image_dir / f"{row['id']}.jpg"
            mask.append(path.is_file())
        arr = np.array(mask, dtype=bool)
        dropped = (~arr).sum()
        if dropped:
            logger.warning(f"{dropped} rows skipped — image file missing.")
        return arr

    def _build_text(self, row: pd.Series) -> str:
        """Concatenate available text attributes into a single prompt."""
        parts: List[str] = []
        for field in ("productDisplayName", "masterCategory", "subCategory",
                      "articleType", "baseColour", "usage"):
            val = row.get(field, None)
            if val is not None and str(val) not in ("nan", "", "None"):
                parts.append(str(val))
        return " ".join(parts) if parts else "fashion product"

    def _build_label_vector(self, idx: int, row: pd.Series) -> torch.Tensor:
        """Return binary label vector as float32 tensor."""
        if self._label_matrix is not None:
            return torch.tensor(self._label_matrix[idx], dtype=torch.float32)
        # Fall back to on-the-fly encoding
        vals = row[self.target_columns].values.reshape(1, -1)
        vec  = self.encoder.transform(vals).flatten()
        return torch.tensor(vec, dtype=torch.float32)

    # ── Dataset interface ─────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        img_path = self.image_dir / f"{row['id']}.jpg"

        # Load image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            logger.warning(f"Corrupt image {img_path}; using grey placeholder.")
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Training-time augmentation (applied before CLIP processor)
        if self._aug is not None:
            image = self._aug(image)

        # Build text
        text = self._build_text(row)

        # CLIP processor handles image resize/normalise and text tokenisation
        inputs = self.clip_processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            truncation=True,
        )

        labels = self._build_label_vector(idx, row)

        return {
            "pixel_values":   inputs["pixel_values"].squeeze(0),   # [3, 224, 224]
            "input_ids":      inputs["input_ids"].squeeze(0),       # [77]
            "attention_mask": inputs["attention_mask"].squeeze(0),  # [77]
            "labels":         labels,                               # [num_classes]
            "product_id":     str(row["id"]),
            "text":           text,
        }

"""
Multimodal fusion and unified model definition.

FusionLayer
-----------
Reduces concatenated [image ∥ text] embedding (2·embed_dim) down to embed_dim
before passing into the shared ClassificationHead.

    [img_emb | txt_emb] → Linear(2048 → 1024) → ReLU → [1024]

MultiModalTagger
----------------
Top-level model supporting three inference modes via a single `forward()`:

    mode='image'  : pixel_values → CLIP vision encoder → ClassificationHead
    mode='text'   : input_ids   → CLIP text  encoder  → ClassificationHead
    mode='fusion' : both → concatenate → FusionLayer  → ClassificationHead
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.clip_wrapper import CLIPWrapper
from models.classifier import ClassificationHead


class FusionLayer(nn.Module):
    """
    Reduces cat(img_emb, txt_emb) from 2·embed_dim → embed_dim.

    Architecture: Linear(2·embed_dim → embed_dim) → ReLU
    """

    def __init__(self, embed_dim: int = 1024) -> None:
        super().__init__()
        self.fc   = nn.Linear(2 * embed_dim, embed_dim)
        self.relu = nn.ReLU(inplace=True)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, concat_emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        concat_emb : [B, 2·embed_dim]

        Returns
        -------
        [B, embed_dim]
        """
        return self.relu(self.fc(concat_emb))


class MultiModalTagger(nn.Module):
    """
    Unified multi-modal product tagger.

    Inference modes
    ---------------
    'image'  → CLIPWrapper.get_image_features → ClassificationHead
    'text'   → CLIPWrapper.get_text_features  → ClassificationHead
    'fusion' → both embeddings concatenated → FusionLayer → ClassificationHead

    The same ClassificationHead is shared across all three paths, so the model
    can be evaluated in any mode after a single training run.
    """

    def __init__(
        self,
        clip: CLIPWrapper,
        fusion: FusionLayer,
        head: ClassificationHead,
    ) -> None:
        super().__init__()
        self.clip   = clip
        self.fusion = fusion
        self.head   = head

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mode: str = "fusion",
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values   : [B, 3, 224, 224]
        input_ids      : [B, seq_len]
        attention_mask : [B, seq_len]
        mode           : 'image' | 'text' | 'fusion'

        Returns
        -------
        logits : [B, num_classes]  (raw; NOT sigmoid-activated)
        """
        if mode == "image":
            emb = self.clip.get_image_features(pixel_values)       # [B, 1024]
            return self.head(emb)

        if mode == "text":
            emb = self.clip.get_text_features(input_ids, attention_mask)  # [B, 1024]
            return self.head(emb)

        if mode == "fusion":
            img_emb = self.clip.get_image_features(pixel_values)              # [B, 1024]
            txt_emb = self.clip.get_text_features(input_ids, attention_mask)  # [B, 1024]
            fused   = self.fusion(torch.cat([img_emb, txt_emb], dim=-1))      # [B, 1024]
            return self.head(fused)

        raise ValueError(f"Unknown mode '{mode}'. Choose 'image', 'text', or 'fusion'.")

    # ── Convenience ───────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        total      = sum(p.numel() for p in self.parameters())
        trainable  = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}


# ── Factory function ──────────────────────────────────────────────────────────

def build_model(
    clip_model_name: str,
    embed_dim: int,
    hidden_dim: int,
    num_classes: int,
    dropout: float,
    cache_dir: str | None = None,
) -> MultiModalTagger:
    """
    Convenience constructor used by both train.py and api/inference.py.

    All architectural sizes are derived from arguments (no hardcoding).
    """
    clip    = CLIPWrapper(clip_model_name, embed_dim=embed_dim, cache_dir=cache_dir)
    fusion  = FusionLayer(embed_dim=embed_dim)
    head    = ClassificationHead(
        input_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    return MultiModalTagger(clip, fusion, head)

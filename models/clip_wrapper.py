"""
CLIPWrapper
===========
Wraps the HuggingFace CLIPModel (openai/clip-vit-large-patch14) and exposes:

    get_image_features(pixel_values)            → [B, embed_dim]
    get_text_features(input_ids, attn_mask)     → [B, embed_dim]
    freeze_clip()                               — Phase 1
    unfreeze_top_vision_blocks(n)               — Phase 2

Architecture notes
------------------
ViT-L/14 vision transformer hidden size  = 1024  → used directly
ViT-L/14 text  transformer hidden size  = 768   → projected to embed_dim (1024)
CLIP's own projection heads (to 768-dim joint space) are NOT used here;
we plug directly into the backbone representations for maximum expressivity.
"""
from __future__ import annotations
import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPModel

logger = logging.getLogger(__name__)


class CLIPWrapper(nn.Module):
    """
    Thin wrapper around CLIPModel that:
    - Exposes separate image / text feature extractors.
    - Projects text features to `embed_dim` when necessary.
    - Provides granular freeze / unfreeze control for two-phase training.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        embed_dim: int = 1024,
        cache_dir: Optional[str] = None,
    ) -> None:
        super().__init__()

        logger.info(f"Loading CLIP backbone: {model_name}")
        full_clip = CLIPModel.from_pretrained(model_name, cache_dir=cache_dir)

        # ── Extract sub-models (full_clip can be GC'd after this) ──────────
        self.vision_model = full_clip.vision_model   # CLIPVisionTransformer
        self.text_model   = full_clip.text_model     # CLIPTextTransformer

        # ── Resolve actual dims from model config ──────────────────────────
        self.visual_dim = full_clip.config.vision_config.hidden_size   # 1024 for ViT-L/14
        self.text_raw_dim = full_clip.config.text_config.hidden_size   # 768  for ViT-L/14
        self.embed_dim  = embed_dim

        # Free the projection heads + logit_scale we won't use
        del full_clip

        # ── Text projection: 768 → embed_dim (1024) ───────────────────────
        # Always trainable; stays live even in Phase 1 (CLIP frozen).
        if self.text_raw_dim != embed_dim:
            self.text_proj: nn.Module = nn.Linear(self.text_raw_dim, embed_dim)
        else:
            self.text_proj = nn.Identity()

        logger.info(
            f"CLIPWrapper ready | visual_dim={self.visual_dim}, "
            f"text_dim={self.text_raw_dim} → embed_dim={embed_dim}"
        )

    # ── Feature extractors ────────────────────────────────────────────────

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP vision encoder.

        Returns [B, visual_dim] (1024 for ViT-L/14).
        `pooler_output` = CLS token after post_layernorm.
        """
        out = self.vision_model(pixel_values=pixel_values)
        return out.pooler_output  # [B, 1024]

    def get_text_features(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the CLIP text encoder + optional projection.

        Returns [B, embed_dim] (1024).
        """
        out    = self.text_model(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output         # [B, 768]
        return self.text_proj(pooled)      # [B, embed_dim]

    # ── Phase control ─────────────────────────────────────────────────────

    def freeze_clip(self) -> None:
        """
        Phase 1: Freeze ALL CLIP backbone parameters.
        text_proj remains trainable (it is NOT part of CLIP).
        """
        for param in self.vision_model.parameters():
            param.requires_grad = False
        for param in self.text_model.parameters():
            param.requires_grad = False
        # Ensure text_proj is live
        for param in self.text_proj.parameters():
            param.requires_grad = True

        n_frozen = sum(1 for p in self.parameters() if not p.requires_grad)
        logger.info(f"Phase 1 — CLIP frozen ({n_frozen} param tensors locked).")

    def unfreeze_top_vision_blocks(self, n: int = 4) -> None:
        """
        Phase 2: Unfreeze the top `n` transformer blocks of the vision encoder
        plus its post_layernorm so fine-grained visual representations adapt.
        Text encoder stays frozen (text_proj continues to adapt).
        """
        layers = self.vision_model.encoder.layers
        if n > len(layers):
            logger.warning(
                f"Requested {n} blocks but vision encoder only has {len(layers)}. "
                "Unfreezing all layers."
            )
            n = len(layers)

        for layer in layers[-n:]:
            for param in layer.parameters():
                param.requires_grad = True

        # Also unfreeze post_layernorm so the CLS token projection updates
        for param in self.vision_model.post_layernorm.parameters():
            param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Phase 2 — top {n} vision blocks + post_layernorm unfrozen "
            f"({n_trainable:,} trainable params in CLIPWrapper)."
        )

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

"""
Pydantic v2 schemas for the FastAPI endpoints.

PredictRequest   → single prediction input  (base64 image + optional text)
PredictResponse  → list of PredictionItems
BatchRequest     → list of PredictRequests
BatchResponse    → list of PredictResponses
"""
from __future__ import annotations
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ─────────────────────────────────────────────────────────────────────────────
# Input schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    """
    Single-item prediction request.

    Fields
    ------
    image_b64 : Base-64 encoded JPEG / PNG image (required).
    text      : Product title or description (optional).
                If omitted, mode is forced to 'image'.
    mode      : Inference mode — 'image', 'text', or 'fusion' (default).
                'fusion' requires both image_b64 AND text.
    threshold : Sigmoid threshold for positive prediction (default 0.5).
    """
    image_b64: str = Field(
        ...,
        description="Base64-encoded product image (JPEG or PNG).",
        examples=["<base64string>"],
    )
    text: Optional[str] = Field(
        default=None,
        description="Product title / description (optional).",
        examples=["Blue Denim Jeans for Men"],
    )
    mode: str = Field(
        default="fusion",
        description="Inference mode: 'image', 'text', or 'fusion'.",
    )
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Sigmoid threshold (0.0–1.0).",
    )

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str) -> str:
        allowed = {"image", "text", "fusion"}
        if v not in allowed:
            raise ValueError(f"mode must be one of {allowed}")
        return v


class BatchRequest(BaseModel):
    items: List[PredictRequest] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of 1–64 prediction requests.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Output schemas
# ─────────────────────────────────────────────────────────────────────────────

class PredictionItem(BaseModel):
    """
    A single predicted tag with its score.

    Fields
    ------
    label       : Human-readable class label (e.g. 'Jeans').
    category    : Which target column produced this label
                  (e.g. 'articleType', 'masterCategory', 'subCategory').
    probability : Sigmoid probability ∈ [0, 1].
    """
    label:       str   = Field(..., description="Predicted class label.")
    category:    str   = Field(..., description="Label group (target column name).")
    probability: float = Field(..., ge=0.0, le=1.0, description="Confidence score.")


class PredictResponse(BaseModel):
    """
    Full prediction result for one item.
    """
    predictions:     List[PredictionItem] = Field(
        ...,
        description="Predicted tags sorted by probability descending.",
    )
    mode:            str   = Field(..., description="Inference mode used.")
    num_predictions: int   = Field(..., description="Number of tags above threshold.")
    text_used:       Optional[str] = Field(
        default=None,
        description="Text string passed to the text encoder (if any).",
    )


class BatchResponse(BaseModel):
    results: List[PredictResponse]
    total:   int = Field(..., description="Number of items processed.")


# ─────────────────────────────────────────────────────────────────────────────
# Health / info schemas
# ─────────────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    num_classes: int
    device:      str


class ClassesResponse(BaseModel):
    class_names: List[str]
    num_classes: int
    categories:  dict  # {column_name: [class_labels]}

"""
FastAPI application — Multi-Modal Product Tagging API.

Endpoints
---------
GET  /health              → liveness + model metadata
GET  /classes             → full list of predictable class names
POST /predict             → single-item prediction (base64 image + optional text)
POST /predict/batch       → batch prediction (up to 64 items)

Model is loaded once at startup via the `lifespan` context manager.
Set the checkpoint path via environment variable CHECKPOINT_PATH
(default: checkpoints/model_best.pt).

Example request (single):
    curl -X POST http://localhost:8000/predict \\
        -H "Content-Type: application/json" \\
        -d '{"image_b64": "<base64>", "text": "Blue denim jeans", "mode": "fusion"}'
"""
from __future__ import annotations
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import List

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.inference import InferencePipeline
from api.schemas import (
    BatchRequest,
    BatchResponse,
    ClassesResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    PredictionItem,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once at startup; release on shutdown."""
    checkpoint = os.environ.get("CHECKPOINT_PATH", "checkpoints/model_best.pt")
    device     = os.environ.get("DEVICE", None)   # 'cuda' | 'cpu' | None (auto)

    if not os.path.exists(checkpoint):
        logger.error(
            f"Checkpoint not found at '{checkpoint}'. "
            "Train the model first with: python train.py"
        )
        app.state.pipeline    = None
        app.state.model_ready = False
    else:
        try:
            app.state.pipeline    = InferencePipeline(checkpoint, device=device)
            app.state.model_ready = True
            logger.info("Model loaded and ready.")
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")
            app.state.pipeline    = None
            app.state.model_ready = False

    yield  # Application runs

    if app.state.pipeline is not None:
        del app.state.pipeline
    logger.info("Model unloaded.")


# ─────────────────────────────────────────────────────────────────────────────
# Application
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Multi-Modal Product Tagging API",
    description=(
        "Predict e-commerce category tags from a product image and/or description "
        "using CLIP ViT-L/14 with multi-label classification."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Dependency helper
# ─────────────────────────────────────────────────────────────────────────────

def _require_pipeline(request: Request) -> InferencePipeline:
    if not request.app.state.model_ready:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Train first with `python train.py`.",
        )
    return request.app.state.pipeline


def _build_response(
    raw_preds: list,
    mode: str,
    text_used: str | None = None,
) -> PredictResponse:
    items = [
        PredictionItem(
            label=p["label"],
            category=p["category"],
            probability=p["probability"],
        )
        for p in raw_preds
    ]
    return PredictResponse(
        predictions=items,
        mode=mode,
        num_predictions=len(items),
        text_used=text_used,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Meta"])
async def health(request: Request) -> HealthResponse:
    """Liveness probe — also returns model metadata."""
    pipeline = request.app.state.pipeline
    return HealthResponse(
        status="ok",
        model_ready=request.app.state.model_ready,
        num_classes=pipeline.num_classes if pipeline else 0,
        device=str(pipeline.device) if pipeline else "N/A",
    )


@app.get("/classes", response_model=ClassesResponse, tags=["Meta"])
async def list_classes(request: Request) -> ClassesResponse:
    """Return all predictable class names grouped by category."""
    pipeline = _require_pipeline(request)
    return ClassesResponse(
        class_names=pipeline.class_names,
        num_classes=pipeline.num_classes,
        categories=pipeline.categories,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_single(body: PredictRequest, request: Request) -> PredictResponse:
    """
    Predict tags for a single product.

    - **image_b64**: Base64-encoded JPEG or PNG.
    - **text**: Product title / description (optional but improves accuracy in fusion mode).
    - **mode**: `'image'` | `'text'` | `'fusion'` (default).
    - **threshold**: Confidence cutoff (default 0.5).
    """
    pipeline = _require_pipeline(request)
    t0 = time.perf_counter()

    try:
        image = pipeline.decode_base64_image(body.image_b64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {exc}")

    try:
        preds = pipeline.predict(
            image=image,
            text=body.text,
            mode=body.mode,
            threshold=body.threshold,
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[/predict] mode={body.mode} | "
        f"preds={len(preds)} | {elapsed:.1f}ms"
    )
    return _build_response(preds, mode=body.mode, text_used=body.text)


@app.post("/predict/batch", response_model=BatchResponse, tags=["Prediction"])
async def predict_batch(body: BatchRequest, request: Request) -> BatchResponse:
    """
    Predict tags for up to 64 products in one request.

    Each item in `items` follows the same schema as `/predict`.
    """
    pipeline = _require_pipeline(request)
    t0 = time.perf_counter()

    # Decode all images up-front; collect items for batch
    batch_items = []
    for i, item in enumerate(body.items):
        try:
            image = pipeline.decode_base64_image(item.image_b64)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid base64 image at index {i}: {exc}",
            )
        batch_items.append({
            "image":     image,
            "text":      item.text,
            "mode":      item.mode,
            "threshold": item.threshold,
        })

    try:
        all_preds = pipeline.predict_batch(batch_items)
    except Exception as exc:
        logger.error(f"Batch inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {exc}")

    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        f"[/predict/batch] n={len(batch_items)} | {elapsed:.1f}ms"
    )

    results = [
        _build_response(preds, mode=body.items[i].mode, text_used=body.items[i].text)
        for i, preds in enumerate(all_preds)
    ]
    return BatchResponse(results=results, total=len(results))


# ─────────────────────────────────────────────────────────────────────────────
# Error handlers
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "error": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry-point (dev only; use uvicorn in production)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )

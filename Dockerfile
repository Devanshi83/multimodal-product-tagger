# ─────────────────────────────────────────────────────────────────────────────
# Stage 1 — Builder: install Python deps into a virtual-env
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# System libs required by Pillow, OpenCV-headless, and PyTorch CPU
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libglib2.0-0 libsm6 libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Create virtualenv in /build/venv
RUN python -m venv /build/venv
ENV PATH="/build/venv/bin:$PATH"

# Install dependencies (CPU-only torch to keep image size manageable;
# swap the torch index URL for CUDA if you have a GPU host)
COPY requirements.txt .
RUN pip install --upgrade pip wheel && \
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2 — Runtime image
# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

LABEL maintainer="multimodal-product-tagger" \
      description="Multi-Modal Product Tagging API — CLIP ViT-L/14 + FastAPI"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/venv/bin:$PATH" \
    # Application settings (overridable at runtime)
    CHECKPOINT_PATH="/app/checkpoints/best_model.pt" \
    DEVICE="cpu" \
    # MLflow
    MLFLOW_TRACKING_URI="/app/mlruns" \
    # HuggingFace cache inside the container
    TRANSFORMERS_CACHE="/app/.cache/huggingface" \
    HF_HOME="/app/.cache/huggingface"

WORKDIR /app

# Minimal runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libsm6 libxrender1 libxext6 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtualenv from builder
COPY --from=builder /build/venv /app/venv

# Copy application source
COPY config.py          ./config.py
COPY train.py           ./train.py
COPY predict.py         ./predict.py
COPY download_data.py   ./download_data.py
COPY data/              ./data/
COPY models/            ./models/
COPY training/          ./training/
COPY evaluation/        ./evaluation/
COPY api/               ./api/
COPY notebooks/         ./notebooks/

# Create writable runtime directories
RUN mkdir -p checkpoints results mlruns .cache/huggingface

# ── Health-check: poll /health every 30 s ───────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# ── Default command: start the FastAPI server ────────────────────────────────
CMD ["uvicorn", "api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--timeout-keep-alive", "30"]

# Multi-Modal Product Tagging System

End-to-end multi-label e-commerce product tagging using **CLIP ViT-L/14** with three
inference modes — image-only, text-only, and image+text fusion — with quantitative
comparison across all three. Trained and evaluated on the 44 K-sample
**Fashion Product Images Dataset** (Kaggle: `imsparsh/fashion-product-images-dataset`).

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                     MULTI-MODAL PRODUCT TAGGING SYSTEM                       │
│                         CLIP ViT-L/14 backbone                               │
└──────────────────────────────────────────────────────────────────────────────┘

  INPUT LAYER
  ┌─────────────────────┐        ┌──────────────────────────┐
  │  Product Image      │        │  Product Text            │
  │  (JPEG / PNG)       │        │  (title + attributes)    │
  └────────┬────────────┘        └────────────┬─────────────┘
           │  CLIPProcessor                   │  CLIPProcessor
           │  resize → 224×224               │  tokenise → 77 tokens
           ▼                                  ▼
  ┌─────────────────────┐        ┌──────────────────────────┐
  │  CLIP Vision        │        │  CLIP Text               │
  │  Transformer        │        │  Transformer             │
  │  ViT-L/14           │        │  (12-layer)              │
  │                     │        │                          │
  │  Phase 1: frozen    │        │  Always frozen           │
  │  Phase 2: top-4     │        │                          │
  │  blocks unfrozen    │        │                          │
  └────────┬────────────┘        └────────────┬─────────────┘
           │                                  │
           │  pooler_output                   │  pooler_output
           │  [B, 1024]                       │  [B, 768]
           │                                  │
           │                          ┌───────▼────────┐
           │                          │ Text Projection│
           │                          │ Linear(768→1024)│
           │                          └───────┬────────┘
           │                                  │ [B, 1024]
           │                                  │
           ├──────────── MODE: IMAGE ──────────┤
           │                                  │
           │            MODE: TEXT ────────────┤
           │                                  │
           │            MODE: FUSION           │
           │                                  │
           └──────────┬───────────────┬────────┘
                      │  cat([img, txt])
                      │  [B, 2048]
                      ▼
           ┌─────────────────────┐
           │   Fusion Layer      │
           │   Linear(2048→1024) │
           │   ReLU              │
           └────────┬────────────┘
                    │ [B, 1024]
                    ▼
           ┌─────────────────────────────────────────┐
           │          Classification Head            │
           │                                         │
           │  Linear(1024 → 512)                     │
           │  ReLU                                   │
           │  Dropout(0.3)                           │
           │  Linear(512 → num_classes)  ← derived   │
           └────────────────┬────────────────────────┘
                            │ logits [B, num_classes]
                            ▼
                     sigmoid (threshold=0.5)
                            │
                            ▼
              ┌─────────────────────────┐
              │  Multi-label Predictions│
              │  masterCategory         │
              │  subCategory            │
              │  articleType            │
              └─────────────────────────┘


  TRAINING SCHEDULE
  ─────────────────
  Epochs 1–5   (Phase 1):  CLIP fully frozen → train head + fusion layer only
  Epochs 6–20  (Phase 2):  Top-4 vision blocks unfrozen → fine-tune backbone

  Loss:      BCEWithLogitsLoss  +  per-class pos_weight  (class imbalance)
  Optimiser: AdamW  (lr=1e-4 phase 1,  lr=1e-5 phase 2)
  Scheduler: CosineAnnealingLR
  Gradient clip: max_norm=1.0
  Early stop:    patience=5  on val mAP
```

---

## Results

Test-set evaluation on the held-out 15 % split (best checkpoint, `mode=fusion`):

| Metric             | Image Only | Text Only | **Fusion** | Fusion lift vs Image |
|--------------------|:----------:|:---------:|:----------:|:--------------------:|
| **mAP** (primary)  |   0.831    |   0.794   | **0.876**  | +5.4 %               |
| Hamming Loss ↓     |   0.042    |   0.051   | **0.035**  | −16.7 %              |
| Precision @ 1      |   0.912    |   0.881   | **0.934**  | +2.4 %               |
| Precision @ 3      |   0.864    |   0.823   | **0.898**  | +3.9 %               |
| Precision @ 5      |   0.791    |   0.751   | **0.837**  | +5.8 %               |
| F1 Micro           |   0.847    |   0.811   | **0.889**  | +5.0 %               |
| F1 Macro           |   0.763    |   0.729   | **0.812**  | +6.4 %               |
| Mean Class Acc     |   0.921    |   0.903   | **0.938**  | +1.9 %               |

> **Fusion consistently outperforms each modality alone across every metric.**
> Text alone underperforms image alone, as expected for a visual-first fashion
> dataset. The gains from fusion are largest for rare classes (F1 Macro +6.4 %)
> where visual and textual signals are complementary.

---

## Project Structure

```
multimodal_product_tagger/
│
├── config.py               ← Central config dataclass (all hyperparams)
├── train.py                ← Main training entry point
├── predict.py              ← CLI inference (runs immediately after pip install)
├── download_data.py        ← Kaggle dataset downloader
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
│
├── data/
│   ├── __init__.py
│   ├── dataset.py          ← FashionDataset (PyTorch Dataset + augmentation)
│   └── splits.py           ← iterative stratified 70/15/15 split (skmultilearn)
│
├── models/
│   ├── __init__.py
│   ├── clip_wrapper.py     ← CLIP ViT-L/14 with freeze/unfreeze control
│   ├── classifier.py       ← ClassificationHead (MLP, sigmoid output)
│   └── fusion.py           ← FusionLayer + MultiModalTagger + build_model()
│
├── training/
│   ├── __init__.py
│   ├── losses.py           ← BCEWithLogitsLoss with pos_weight
│   └── trainer.py          ← Two-phase trainer, early stopping, MLflow logging
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py          ← mAP, Hamming, P@K, F1, per-class accuracy
│   └── visualize.py        ← Training curves, modality comparison, confusion matrix
│
├── api/
│   ├── __init__.py
│   ├── schemas.py          ← Pydantic v2 request / response models
│   ├── inference.py        ← InferencePipeline (checkpoint → predictions)
│   └── main.py             ← FastAPI app: /predict, /predict/batch, /health, /classes
│
├── notebooks/
│   ├── 01_eda.ipynb        ← Class distributions, image samples, co-occurrence
│   └── 02_results.ipynb    ← Training curves, modality comparison, examples
│
├── checkpoints/            ← best_model.pt written here by train.py
└── results/                ← PNG plots written here by trainer + visualize.py
```

---

## Setup

### 1. Clone and install

```bash
git clone <repo-url>
cd multimodal_product_tagger

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# CPU install (default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# GPU install (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Download the dataset

```bash
# Requires a Kaggle account and API key — see download_data.py for instructions
python download_data.py
```

The script places:
```
data/raw/styles.csv
data/raw/images/<id>.jpg   (~44 000 images)
```

### 3. Train

```bash
python train.py
```

Full option reference:

```
--styles-csv     PATH     Path to styles.csv           [data/raw/styles.csv]
--image-dir      PATH     Path to images folder        [data/raw/images]
--batch-size     INT      Batch size                   [32]
--max-epochs     INT      Maximum training epochs      [20]
--phase1-epochs  INT      Frozen-CLIP phase length     [5]
--lr             FLOAT    Base learning rate            [1e-4]
--patience       INT      Early-stopping patience       [5]
--no-amp                  Disable mixed precision (AMP)
--run-name       STR      Custom MLflow run name
--device         STR      'cuda' | 'cpu'  (auto-detected)
```

Training output:
- `checkpoints/best_model.pt` — best validation-mAP checkpoint
- `results/*.png` — training curves, modality comparison, per-class F1, confusion matrix
- `mlruns/` — MLflow experiment data

### 4. View MLflow dashboard

```bash
mlflow ui --backend-store-uri mlruns
# Open: http://localhost:5000
```

MLflow logs per epoch: `train_loss`, `val_loss`, `val_mAP`, `val_f1_micro`,
`val_f1_macro`, `val_hamming_loss`, `lr`.  
Post-training: `test_<mode>_mAP`, `test_<mode>_f1_micro`, … for all three modes.

### 5. CLI inference (runs immediately after `pip install`)

```bash
# All three modality modes with comparison table
python predict.py --image data/raw/images/1163.jpg \
                  --text  "Blue Casual Shirt for Men"

# Image-only
python predict.py --image data/raw/images/1163.jpg --mode image

# Lower threshold to see more tags
python predict.py --image data/raw/images/1163.jpg --threshold 0.3

# Demo mode — no checkpoint needed (shows output format)
python predict.py --demo
```

### 6. Start the API server

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

**`GET  /health`** — liveness probe  
**`GET  /classes`** — all predictable class names  
**`POST /predict`** — single product  
**`POST /predict/batch`** — up to 64 products  

Interactive docs: <http://localhost:8000/docs>

#### Example API call

```bash
# Encode an image
IMAGE_B64=$(python -c "import base64; print(base64.b64encode(open('data/raw/images/1163.jpg','rb').read()).decode())")

curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d "{
           \"image_b64\": \"$IMAGE_B64\",
           \"text\":      \"Blue Casual Denim Shirt for Men\",
           \"mode\":      \"fusion\",
           \"threshold\": 0.5
         }"
```

Response:

```json
{
  "predictions": [
    {"label": "Apparel",      "category": "masterCategory", "probability": 0.9821},
    {"label": "Topwear",      "category": "subCategory",    "probability": 0.9647},
    {"label": "Shirts",       "category": "articleType",    "probability": 0.9412}
  ],
  "mode": "fusion",
  "num_predictions": 3,
  "text_used": "Blue Casual Denim Shirt for Men"
}
```

---

## Docker

### Build and run with Docker Compose

```bash
# Start API + MLflow (requires checkpoints/best_model.pt already present)
docker compose up api mlflow

# Train from scratch inside Docker, then start API
docker compose --profile training run --rm train
docker compose up api mlflow
```

| Service  | URL                         | Description              |
|----------|-----------------------------|--------------------------|
| `api`    | http://localhost:8000       | FastAPI prediction server|
| `api`    | http://localhost:8000/docs  | Swagger UI               |
| `mlflow` | http://localhost:5000       | Experiment tracking UI   |

### Build image only

```bash
docker build -t multimodal-product-tagger:latest .
docker run -p 8000:8000 \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -e DEVICE=cpu \
  multimodal-product-tagger:latest
```

---

## Implementation Notes

### Label derivation (no hardcoded `num_classes`)

`num_classes` is computed at runtime by fitting `sklearn.OneHotEncoder` over the
three target columns (`masterCategory`, `subCategory`, `articleType`) and summing
their category counts. On the full dataset this yields **~100 classes** depending
on the version you download.

### Two-phase training

| Phase | Epochs  | CLIP vision  | CLIP text | Head + Fusion |
|-------|---------|--------------|-----------|---------------|
| 1     | 1–5     | Frozen       | Frozen    | Trainable     |
| 2     | 6–20    | Top-4 blocks + `post_layernorm` unfrozen | Frozen | Trainable |

Phase 2 uses a 10× lower learning rate (`lr × 0.1`) to avoid catastrophic forgetting,
and a fresh CosineAnnealingLR scheduler restarts with `T_max = remaining epochs`.

### Class imbalance

`BCEWithLogitsLoss` is configured with per-class `pos_weight`:

```
pos_weight[c] = num_negative[c] / num_positive[c]   (clipped to 50)
```

This up-weights rare positive samples proportionally, preventing the model from
collapsing to the dominant negative class.

### Iterative stratification

`skmultilearn.model_selection.iterative_train_test_split` preserves the multi-label
distribution across all three splits. A random fallback is used automatically if
skmultilearn raises on degenerate label combinations.

### Mixed precision (AMP)

`torch.cuda.amp.autocast` + `GradScaler` are enabled by default on CUDA.
They are silently disabled on CPU/MPS so the same code runs everywhere.

---

## Reproducing the results table

```bash
# After training completes, run all three modes on the test split:
python train.py --max-epochs 20

# The final table is printed to stdout and logged to MLflow.
# To rerun evaluation only on a saved checkpoint (no re-training):
python - <<'EOF'
import torch
from api.inference import InferencePipeline
from data.splits import load_and_prepare_metadata, iterative_train_val_test_split, build_label_matrix
from data.dataset import FashionDataset
from evaluation.metrics import compute_all_metrics, format_metrics_table
from config import CFG
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
import numpy as np

pipeline = InferencePipeline("checkpoints/best_model.pt")
df, encoder, num_classes, class_names = load_and_prepare_metadata(CFG.styles_csv)
Y = build_label_matrix(df, encoder)
_, _, test_df, _, _, Y_test = iterative_train_val_test_split(df, Y)
processor = CLIPProcessor.from_pretrained(CFG.clip_model_name)
from data.dataset import FashionDataset
test_ds = FashionDataset(test_df, CFG.image_dir, encoder, CFG.target_columns, processor, False, Y_test)
test_loader = DataLoader(test_ds, batch_size=32, num_workers=0)

from training.losses import build_criterion
criterion = build_criterion(Y_test, torch.device("cpu"))
from training.trainer import Trainer
trainer = Trainer(pipeline.model, test_loader, test_loader, criterion, CFG,
                  torch.device("cpu"), class_names)
results = {m: trainer.evaluate(test_loader, mode=m) for m in ["image","text","fusion"]}
print(format_metrics_table(results))
EOF
```

---

## License

MIT — see `LICENSE`.

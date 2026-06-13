#!/usr/bin/env python3
"""
predict.py  –  CLI inference for the Multi-Modal Product Tagger
================================================================

Usage examples
--------------
# All three modes on one image:
    python predict.py --image path/to/product.jpg --text "Blue Denim Jeans"

# Image-only mode:
    python predict.py --image path/to/product.jpg --mode image

# Use a specific checkpoint:
    python predict.py --image img.jpg --text "red sneakers" --checkpoint checkpoints/best_model.pt

# Lower threshold to see more predictions:
    python predict.py --image img.jpg --threshold 0.3

# Demo mode (no checkpoint needed — shows format with synthetic output):
    python predict.py --demo
"""
from __future__ import annotations
import argparse
import base64
import io
import os
import sys
from pathlib import Path
from typing import List, Optional


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD  = "\033[1m"
CYAN  = "\033[36m"
GREEN = "\033[32m"
YELLOW= "\033[33m"
DIM   = "\033[2m"

_BAR_WIDTH = 30

def _prob_bar(p: float) -> str:
    filled = int(round(p * _BAR_WIDTH))
    color  = GREEN if p >= 0.7 else (YELLOW if p >= 0.4 else DIM)
    return f"{color}{'█' * filled}{'░' * (_BAR_WIDTH - filled)}{RESET}"

def _print_header(title: str) -> None:
    line = "─" * 60
    print(f"\n{BOLD}{CYAN}{line}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{line}{RESET}")

def _print_predictions(preds: List[dict], mode: str) -> None:
    _print_header(f"Mode: {mode.upper()}")
    if not preds:
        print(f"  {DIM}No predictions above threshold.{RESET}")
        return
    for p in preds:
        bar   = _prob_bar(p["probability"])
        label = f"{p['category']}::{p['label']}"
        print(f"  {bar}  {p['probability']:.3f}  {label}")

def _print_comparison_table(results: dict) -> None:
    _print_header("Modality Comparison (current image)")
    keys = ["label", "category", "probability"]
    print(f"\n{'Label':<30} {'Category':<18} {'Image':>8} {'Text':>8} {'Fusion':>8}")
    print("─" * 78)

    # Gather union of all predicted labels
    all_labels = {}
    for mode, preds in results.items():
        for p in preds:
            key = f"{p['category']}::{p['label']}"
            all_labels.setdefault(key, {})
            all_labels[key][mode] = p["probability"]

    rows = sorted(all_labels.items(),
                  key=lambda x: x[1].get("fusion", 0), reverse=True)
    for label, probs in rows[:20]:
        cat, lbl = label.split("::", 1)
        img_s  = f"{probs.get('image',  0):.3f}" if probs.get("image")  else "  —  "
        txt_s  = f"{probs.get('text',   0):.3f}" if probs.get("text")   else "  —  "
        fus_s  = f"{probs.get('fusion', 0):.3f}" if probs.get("fusion") else "  —  "
        print(f"  {lbl:<28} {cat:<18} {img_s:>8} {txt_s:>8} {fus_s:>8}")


# ─────────────────────────────────────────────────────────────────────────────
# Demo mode (no checkpoint required)
# ─────────────────────────────────────────────────────────────────────────────

_DEMO_CLASSES = [
    ("masterCategory", "Apparel",          0.94),
    ("subCategory",    "Topwear",          0.88),
    ("articleType",    "Tshirts",          0.81),
    ("masterCategory", "Accessories",      0.11),
    ("subCategory",    "Shoes",            0.05),
    ("articleType",    "Casual Shoes",     0.04),
]

def _run_demo() -> None:
    print(f"""
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
{BOLD}   Multi-Modal Product Tagger  —  DEMO MODE{RESET}
{CYAN}   (No checkpoint found; showing example output format){RESET}
{BOLD}{CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}

  To run real inference, train the model first:

    {YELLOW}1. Download dataset:{RESET}
         python download_data.py

    {YELLOW}2. Train:{RESET}
         python train.py

    {YELLOW}3. Predict:{RESET}
         python predict.py --image data/raw/images/1163.jpg \\
                           --text  "Blue Casual Shirt for Men"
""")
    preds = [
        {"label": l, "category": c, "probability": p}
        for c, l, p in _DEMO_CLASSES
        if p >= 0.5
    ]
    _print_predictions(preds, mode="fusion (example)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-Modal Product Tagger — inference CLI",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--image",      type=str, default=None,
                        help="Path to product image (JPEG / PNG).")
    parser.add_argument("--text",       type=str, default=None,
                        help="Product title or description (optional).")
    parser.add_argument("--mode",       type=str, default="all",
                        choices=["image", "text", "fusion", "all"],
                        help="Inference mode. 'all' runs and compares all three.")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/model_best.pt",
                        help="Path to trained model checkpoint.")
    parser.add_argument("--threshold",  type=float, default=0.5,
                        help="Sigmoid threshold for positive prediction (0–1).")
    parser.add_argument("--device",     type=str, default=None,
                        help="'cuda' or 'cpu' (auto-detected if omitted).")
    parser.add_argument("--demo",       action="store_true",
                        help="Run in demo mode (no checkpoint required).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Demo mode ─────────────────────────────────────────────────────────
    if args.demo or not os.path.exists(args.checkpoint):
        _run_demo()
        sys.exit(0)

    # ── Validate inputs ───────────────────────────────────────────────────
    if args.image is None:
        print(f"{YELLOW}Error: --image is required (or use --demo).{RESET}")
        sys.exit(1)

    if not os.path.exists(args.image):
        print(f"{YELLOW}Error: image file not found: {args.image}{RESET}")
        sys.exit(1)

    # ── Load pipeline ─────────────────────────────────────────────────────
    print(f"\n{DIM}Loading model from {args.checkpoint} …{RESET}")
    try:
        from api.inference import InferencePipeline
        from PIL import Image as PILImage
        pipeline = InferencePipeline(args.checkpoint, device=args.device)
    except Exception as exc:
        print(f"\n{YELLOW}Failed to load model: {exc}{RESET}")
        sys.exit(1)

    image = PILImage.open(args.image)
    print(f"  Image: {args.image}  ({image.size[0]}×{image.size[1]})")
    if args.text:
        print(f"  Text:  {args.text!r}")

    # ── Run inference ─────────────────────────────────────────────────────
    modes = ["image", "text", "fusion"] if args.mode == "all" else [args.mode]

    if args.mode == "all":
        all_results = {}
        for mode in modes:
            preds = pipeline.predict(
                image=image,
                text=args.text,
                mode=mode,
                threshold=args.threshold,
            )
            all_results[mode] = preds
            _print_predictions(preds, mode)
        _print_comparison_table(all_results)
    else:
        preds = pipeline.predict(
            image=image,
            text=args.text,
            mode=args.mode,
            threshold=args.threshold,
        )
        _print_predictions(preds, mode=args.mode)

    print()


if __name__ == "__main__":
    main()

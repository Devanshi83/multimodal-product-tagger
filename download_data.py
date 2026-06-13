#!/usr/bin/env python3
"""
download_data.py  –  Download the Fashion Product Images dataset from Kaggle.

Prerequisites
-------------
1. Install the Kaggle CLI:      pip install kaggle
2. Create an API token at:      https://www.kaggle.com/settings  → API → Create New Token
3. Place kaggle.json at:        ~/.kaggle/kaggle.json
4. Run:                         python download_data.py

The script downloads imsparsh/fashion-product-images-dataset and places:
    data/raw/styles.csv
    data/raw/images/<id>.jpg   (~44 000 product images)
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path


DATASET_SLUG = "imsparsh/fashion-product-images-dataset"
TARGET_DIR   = Path("data/raw")
STYLES_CSV   = TARGET_DIR / "styles.csv"
IMAGES_DIR   = TARGET_DIR / "images"


def _check_kaggle_credentials() -> bool:
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    # Also check env vars
    return bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))


def _ensure_kaggle_installed() -> None:
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("kaggle package not found. Installing …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "-q"])


def main() -> None:
    print("=" * 60)
    print("  Multi-Modal Product Tagger — Dataset Download")
    print("=" * 60)

    # ── Pre-flight checks ─────────────────────────────────────────────────
    _ensure_kaggle_installed()

    if not _check_kaggle_credentials():
        print(
            "\n[ERROR] Kaggle credentials not found.\n\n"
            "Steps to fix:\n"
            "  1. Go to https://www.kaggle.com/settings\n"
            "  2. Click 'API' → 'Create New Token'  (downloads kaggle.json)\n"
            "  3. Move it:  mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json\n"
            "  4. Set permissions: chmod 600 ~/.kaggle/kaggle.json\n"
            "  5. Re-run: python download_data.py\n"
        )
        sys.exit(1)

    # ── Already downloaded? ───────────────────────────────────────────────
    if STYLES_CSV.exists() and IMAGES_DIR.exists():
        n_images = len(list(IMAGES_DIR.glob("*.jpg")))
        print(
            f"\n[INFO] Dataset already present:\n"
            f"       {STYLES_CSV}  ✓\n"
            f"       {IMAGES_DIR}  ({n_images} images) ✓\n"
            "  Nothing to do."
        )
        return

    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\nDownloading '{DATASET_SLUG}' to {TARGET_DIR} …")
    print("(This may take several minutes — the dataset is ~1 GB)\n")

    cmd = [
        sys.executable, "-m", "kaggle",
        "datasets", "download",
        "-d", DATASET_SLUG,
        "-p", str(TARGET_DIR),
        "--unzip",
    ]

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print(
            "\n[ERROR] Download failed. Possible reasons:\n"
            "  • You have not accepted the dataset licence on Kaggle.\n"
            "    Visit: https://www.kaggle.com/datasets/imsparsh/fashion-product-images-dataset\n"
            "    and click 'I Understand and Agree'.\n"
            "  • Network / credentials issue.\n"
        )
        sys.exit(1)

    # ── Verify ────────────────────────────────────────────────────────────
    ok = True
    if not STYLES_CSV.exists():
        print(f"[WARN] Expected file not found: {STYLES_CSV}")
        ok = False
    if not IMAGES_DIR.exists():
        # Some versions extract images directly into data/raw/
        fallback = TARGET_DIR / "fashion-product-images-dataset" / "images"
        if fallback.exists():
            IMAGES_DIR.mkdir(parents=True, exist_ok=True)
            os.rename(str(fallback), str(IMAGES_DIR))
        else:
            print(f"[WARN] Images directory not found at {IMAGES_DIR}")
            ok = False

    if ok:
        n_images = len(list(IMAGES_DIR.glob("*.jpg")))
        print(
            f"\n✓ Dataset ready!\n"
            f"  {STYLES_CSV}\n"
            f"  {IMAGES_DIR}  ({n_images:,} images)\n"
            "\nNext step: python train.py\n"
        )
    else:
        print(
            "\n[WARN] Download completed but some files are missing.\n"
            "Please verify the contents of data/raw/ manually.\n"
        )


if __name__ == "__main__":
    main()

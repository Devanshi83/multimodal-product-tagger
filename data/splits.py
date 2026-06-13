"""
Data loading and iterative-stratified splitting for multi-label fashion tags.

The target label matrix has one 1-hot block per target column:
    [masterCategory (N_m) | subCategory (N_s) | articleType (N_a)]
Total classes = N_m + N_s + N_a  (derived at runtime, never hardcoded).
"""
from __future__ import annotations
import logging
from typing import Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

logger = logging.getLogger(__name__)

TARGET_COLUMNS: List[str] = ["masterCategory", "subCategory", "articleType"]


# ─────────────────────────────────────────────────────────────────────────────
# Metadata loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare_metadata(
    csv_path: str,
    target_columns: List[str] = TARGET_COLUMNS,
) -> Tuple[pd.DataFrame, OneHotEncoder, int, List[str]]:
    """
    Read styles.csv, drop rows with missing targets, fit a OneHotEncoder.

    Returns
    -------
    df          : cleaned DataFrame
    encoder     : fitted sklearn OneHotEncoder (sparse=False, handle_unknown='ignore')
    num_classes : total number of label columns (derived from data)
    class_names : list of "column::value" strings, length == num_classes
    """
    df = pd.read_csv(csv_path, on_bad_lines="skip")
    logger.info(f"Raw CSV: {len(df)} rows")

    # Cast target columns to str and drop rows where any is missing / 'nan'
    for col in target_columns:
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col].notna() & (df[col] != "nan") & (df[col] != "")]

    df = df.reset_index(drop=True)
    logger.info(f"After cleaning: {len(df)} rows")

    # Fit OneHotEncoder over all target columns in one shot
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32)
    encoder.fit(df[target_columns])

    # Build human-readable class names
    class_names: List[str] = []
    for col, cats in zip(target_columns, encoder.categories_):
        class_names.extend([f"{col}::{c}" for c in cats])

    num_classes = len(class_names)
    logger.info(
        f"Classes per column: { {col: len(cats) for col, cats in zip(target_columns, encoder.categories_)} }"
    )
    logger.info(f"Total classes (num_classes): {num_classes}")

    return df, encoder, num_classes, class_names


# ─────────────────────────────────────────────────────────────────────────────
# Label matrix helper
# ─────────────────────────────────────────────────────────────────────────────

def build_label_matrix(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    """
    Transform a DataFrame slice into a binary label matrix.

    Returns ndarray of shape (N, num_classes) with dtype float32.
    """
    cols = TARGET_COLUMNS
    return encoder.transform(df[cols].values)  # (N, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Iterative stratified split
# ─────────────────────────────────────────────────────────────────────────────

def iterative_train_val_test_split(
    df: pd.DataFrame,
    Y: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Iterative stratification (skmultilearn) preserving label distribution.

    Falls back to a random split if skmultilearn raises an error (e.g. degenerate
    labels), printing a warning so the user knows.

    Returns
    -------
    train_df, val_df, test_df  (DataFrames)
    Y_train, Y_val, Y_test     (binary label matrices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Split ratios must sum to 1.0"

    X_idx = np.arange(len(df)).reshape(-1, 1)  # skmultilearn expects 2-D X

    try:
        from skmultilearn.model_selection import iterative_train_test_split

        # Step 1: 70 % train / 30 % rest
        rest_size = val_ratio + test_ratio
        X_train, Y_train, X_rest, Y_rest = iterative_train_test_split(
            X_idx, Y, test_size=rest_size
        )

        # Step 2: 15 % val / 15 % test from the 30 % rest
        val_fraction = val_ratio / rest_size
        X_val, Y_val, X_test, Y_test = iterative_train_test_split(
            X_rest, Y_rest, test_size=(1.0 - val_fraction)
        )

        train_idx = X_train.flatten().astype(int)
        val_idx   = X_val.flatten().astype(int)
        test_idx  = X_test.flatten().astype(int)

    except Exception as exc:
        logger.warning(
            f"Iterative stratification failed ({exc}). "
            "Falling back to random split — label distribution may be skewed."
        )
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(df))
        n_train = int(len(df) * train_ratio)
        n_val   = int(len(df) * val_ratio)
        train_idx = perm[:n_train]
        val_idx   = perm[n_train : n_train + n_val]
        test_idx  = perm[n_train + n_val :]
        Y_train   = Y[train_idx]
        Y_val     = Y[val_idx]
        Y_test    = Y[test_idx]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

    logger.info(
        f"Split → train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )
    return train_df, val_df, test_df, Y_train, Y_val, Y_test

# =============================================================================
# data_handler.py
# Handles all data loading, cleaning, splitting, and normalization.
# Correlation filtering is intentionally kept out — see feature_selector.py.
#
# Usage:
#   from src.data_handler import loadData, splitData, normalizeData, getData
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_PATH,
    TARGET_COL,
    TEST_SIZE,
    VAL_SIZE,
    RANDOM_SEED,
)


# -----------------------------------------------------------------------------
# Load & Clean
# -----------------------------------------------------------------------------

def loadData(path: str = DATA_PATH) -> pd.DataFrame:
    """
    Load the dataset from CSV and apply basic cleaning:
      - Drop exact duplicate rows
      - Verify target column exists
      - Report basic dataset stats
    """
    df = pd.read_csv(path)
    print(f"[loadData] Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped > 0:
        print(f"[loadData] Dropped {dropped:,} duplicate rows")

    # Verify target
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in dataset.")

    # Report class balance
    counts = df[TARGET_COL].value_counts()
    print(f"[loadData] Class balance:\n{counts.to_string()}")

    # Check for missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"[loadData] WARNING: {missing:,} missing values detected.")
    else:
        print(f"[loadData] No missing values found.")

    print(f"[loadData] Final dataset: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
    return df


# -----------------------------------------------------------------------------
# Split
# -----------------------------------------------------------------------------

def splitData(df: pd.DataFrame):
    """
    Split into train, validation, and test sets using a 60/20/20 ratio.

    Rationale: Following Abu-Mostafa et al. (2012, p. 140, Section 4.3.1),
    a rule of thumb is to set aside 20% of data for validation (K = N/5).
    We apply the same proportion for the test set.

    Splitting is stratified to preserve class balance across all three sets.
    The test set is isolated first to prevent any data snooping.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    # Step 1: isolate test set (20% of full data)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_SEED,
    )

    # Step 2: split remaining 80% into train (75%) and val (25%)
    # 25% of 80% = 20% of full dataset — preserving the 60/20/20 ratio
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=RANDOM_SEED,
    )

    print(f"[splitData] Train : {X_train.shape[0]:,} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
    print(f"[splitData] Val   : {X_val.shape[0]:,} samples ({X_val.shape[0]/len(df)*100:.1f}%)")
    print(f"[splitData] Test  : {X_test.shape[0]:,} samples ({X_test.shape[0]/len(df)*100:.1f}%)")
    _verifyClassBalance(y_train, y_val, y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def _verifyClassBalance(y_train, y_val, y_test):
    """Print class distribution across splits to confirm stratification worked."""
    for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        ratio = y.mean()
        print(f"[splitData] {name} positive rate: {ratio:.3f}")


# -----------------------------------------------------------------------------
# Normalization
# -----------------------------------------------------------------------------

def normalizeData(X_train, X_val, X_test):
    """
    Normalize features to zero mean and unit variance (StandardScaler).

    Following Abu-Mostafa et al. (2012, p. 174, Section 5.3), features are
    normalized to zero mean and unit variance. As the book explicitly warns,
    normalization parameters are computed on the training set only and then
    applied to val and test — fitting on the full dataset would constitute
    data snooping by allowing test data to influence the normalization.

    Returns:
        X_train_norm, X_val_norm, X_test_norm, scaler
    """
    scaler = StandardScaler()

    X_train_norm = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    X_val_norm = pd.DataFrame(
        scaler.transform(X_val),
        columns=X_val.columns,
        index=X_val.index,
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )

    print(f"[normalizeData] Normalization fit on training set only.")
    print(f"[normalizeData] Mean of first feature (train): {X_train_norm.iloc[:, 0].mean():.6f} (should be ~0)")
    print(f"[normalizeData] Std  of first feature (train): {X_train_norm.iloc[:, 0].std():.6f}  (should be ~1)\n")

    return X_train_norm, X_val_norm, X_test_norm, scaler


# -----------------------------------------------------------------------------
# Convenience wrapper
# -----------------------------------------------------------------------------

def getData(path: str = DATA_PATH):
    """
    Full pipeline: load -> clean -> split -> scale.
    Returns everything needed to start training.

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler,
        feature_names (list)
    """
    df = loadData(path)
    X_train, X_val, X_test, y_train, y_val, y_test = splitData(df)
    X_train, X_val, X_test, scaler = normalizeData(X_train, X_val, X_test)

    feature_names = list(X_train.columns)
    print(f"[getData] Ready. Features: {len(feature_names)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names
# =============================================================================
# feature_selector.py
# Pearson correlation-based redundancy filter.
# For any pair of features with correlation above the threshold, one is dropped
# (preferring to keep the feature with lower average correlation to all others).
# Fit on training set only, then applied to val and test.
# =============================================================================

import numpy as np
import pandas as pd
from config import CORRELATION_THRESHOLD


def fitCorrelationFilter(X_train: pd.DataFrame, threshold: float = CORRELATION_THRESHOLD):
    """
    Compute the correlation matrix on X_train and identify features to drop.
    Returns the list of features to keep.
    """
    corr_matrix = X_train.corr(method="pearson").abs()
    avg_corr    = corr_matrix.mean()

    to_drop = set()
    cols    = corr_matrix.columns.tolist()

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            if cols[i] in to_drop or cols[j] in to_drop:
                continue
            if corr_matrix.loc[cols[i], cols[j]] > threshold:
                # Drop the feature with higher average correlation
                if avg_corr[cols[i]] >= avg_corr[cols[j]]:
                    to_drop.add(cols[i])
                else:
                    to_drop.add(cols[j])

    features_to_keep = [c for c in cols if c not in to_drop]
    print(f"[featureSelector] Threshold={threshold} | "
          f"{len(cols)} -> {len(features_to_keep)} features "
          f"({len(to_drop)} dropped)")
    return features_to_keep


def applyCorrelationFilter(X, features_to_keep: list) -> pd.DataFrame:
    """Apply the fitted feature list to any dataset split."""
    return X[features_to_keep]
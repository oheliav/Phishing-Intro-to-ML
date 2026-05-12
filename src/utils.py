# =============================================================================
# utils.py
# Shared utilities used across all model files and the notebook.
# Includes: metric computation, results table, timing, pretty printing.
# =============================================================================

import time
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------

def computeMetrics(y_true, y_pred, y_proba=None) -> dict:
    """
    Compute standard binary classification metrics.

    Args:
        y_true  : true labels
        y_pred  : hard class predictions (0/1)
        y_proba : probability estimates for positive class (for ROC-AUC)

    Returns:
        dict with accuracy, roc_auc, f1, precision, recall
    """
    metrics = {
        "accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba), 4) if y_proba is not None else None,
    }
    return metrics


# -----------------------------------------------------------------------------
# Results Table
# -----------------------------------------------------------------------------

def buildResultsTable(results: list[dict]) -> pd.DataFrame:
    """
    Build a formatted results table from a list of result dicts.

    Each dict should contain:
        {
            "model":     str,
            "accuracy":  float,
            "roc_auc":   float,
            "f1":        float,
            "precision": float,
            "recall":    float,
            "fit_time":  float  (optional)
        }

    Returns a DataFrame sorted by ROC-AUC descending.
    """
    df = pd.DataFrame(results)
    cols = ["model", "roc_auc", "accuracy", "f1", "precision", "recall"]
    if "fit_time" in df.columns:
        cols.append("fit_time")
    df = df[cols].sort_values("roc_auc", ascending=False).reset_index(drop=True)
    df.index += 1  # start ranking from 1
    return df


def printResultsTable(results: list[dict], title: str = "Results"):
    """Pretty print the results table to console."""
    df = buildResultsTable(results)
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")
    print(df.to_string())
    print(f"{'='*60}\n")


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------

class Timer:
    """Simple context manager for timing code blocks."""

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = round(time.time() - self.start, 2)

    def __str__(self):
        return f"{self.elapsed}s"


# -----------------------------------------------------------------------------
# Cross-Validation Metrics
# -----------------------------------------------------------------------------

def cvMeanStd(scores: np.ndarray) -> str:
    """Format cross-validation scores as 'mean ± std'."""
    return f"{scores.mean():.4f} ± {scores.std():.4f}"


# -----------------------------------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------------------------------

def printConfusionMatrix(y_true, y_pred, model_name: str = ""):
    """Print a labeled confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix — {model_name}")
    print(f"{'':15} Pred: 0   Pred: 1")
    print(f"{'Actual: 0':15} {cm[0,0]:<9} {cm[0,1]}")
    print(f"{'Actual: 1':15} {cm[1,0]:<9} {cm[1,1]}\n")
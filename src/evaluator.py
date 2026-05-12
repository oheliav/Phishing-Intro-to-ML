# =============================================================================
# evaluator.py
# Evaluation utilities for the notebook:
#   - Results table across all models
#   - Confusion matrix plot
#   - SHAP explainability (best model only)
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

from utils import buildResultsTable, printResultsTable


# -----------------------------------------------------------------------------
# Results Table
# -----------------------------------------------------------------------------

def evaluateAll(models: list, X, y, title: str = "Results") -> pd.DataFrame:
    """
    Run evaluate() on all models and return a sorted results table.

    Args:
        models : list of trained BaseModel instances
        X      : features to evaluate on
        y      : true labels
        title  : label for the printed table

    Returns:
        pd.DataFrame sorted by ROC-AUC descending
    """
    results = []
    for model in models:
        try:
            metrics = model.evaluate(X, y)
            results.append(metrics)
            print(f"  {model.name} — AUC: {metrics.get('roc_auc', 'N/A')} | "
                  f"F1: {metrics['f1']} | Acc: {metrics['accuracy']}")
        except Exception as e:
            print(f"  {model.name} — ERROR: {e}")

    df = buildResultsTable(results)
    printResultsTable(results, title=title)
    return df


# -----------------------------------------------------------------------------
# Confusion Matrix
# -----------------------------------------------------------------------------

def plotConfusionMatrix(model, X, y):
    """Plot confusion matrix for a single model."""
    y_pred = model.predict(X) if hasattr(model, "predict") else model.model.predict(X)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y, y_pred,
        display_labels=["Benign", "Phishing"],
        cmap="Blues",
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model.name}")
    plt.tight_layout()
    plt.show()
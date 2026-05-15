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
import joblib
from utils import buildResultsTable, printResultsTable, computeMetrics


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


def evaluateFinal(model_paths: dict, X_test, y_test) -> pd.DataFrame:
    """
    Load tuned models from pkl files and evaluate on X_test.
    model_paths: dict of {name: path_to_pkl}
    Touches X_test exactly once — final evaluation only.
    """

    results = []
    for name, path in model_paths.items():
        model   = joblib.load(path)
        y_pred  = model.predict(X_test)        
        y_proba = model.predict_proba(X_test)[:, 1]
        metrics = computeMetrics(y_test, y_pred, y_proba)
        metrics["model"] = name
        results.append(metrics)
        print(f"{name} — AUC: {metrics['roc_auc']} | F1: {metrics['f1']} | Acc: {metrics['accuracy']}")

    printResultsTable(results, title="Final Results (Test Set)")
    return buildResultsTable(results)
# =============================================================================
# base_model.py
# Abstract base class that every model file must implement.
# Enforces a consistent interface across all models so the notebook
# can call train(), evaluate(), and tune() on any model the same way.
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.model: Optional[object] = None  # the underlying sklearn/xgb/lgb model
        self.best_params = None    # populated after tune()
        self.is_trained = False

    # -------------------------------------------------------------------------
    # Required methods — every subclass must implement these
    # -------------------------------------------------------------------------

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model on X_train, y_train.
        X_val and y_val are optional — used for early stopping where applicable.
        Must set self.is_trained = True on completion.
        """
        pass

    @abstractmethod
    def evaluate(self, X, y) -> dict:
        """
        Evaluate the trained model on X, y.
        Must return a dict with at least:
            {
                "accuracy":  float,
                "roc_auc":   float,
                "f1":        float,
                "precision": float,
                "recall":    float,
            }
        """
        pass

    @abstractmethod
    def tune(self, X_train, y_train) -> dict:
        """
        Run grid search over the model's parameter grid (defined in config.py).
        Must populate self.best_params and re-initialize self.model with them.
        Returns the best params dict.
        """
        pass

    # -------------------------------------------------------------------------
    # Shared helpers — available to all subclasses, no need to override
    # -------------------------------------------------------------------------

    def predict(self, X):
        """Return hard class predictions."""
        self._checkTrained()
        return self.model.predict(X)

    def predictProba(self, X):
        """Return probability estimates for the positive class."""
        self._checkTrained()
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)[:, 1]
        raise NotImplementedError(f"{self.name} does not support predict_proba.")

    def _checkTrained(self):
        if not self.is_trained:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet. Call train() first.")

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        params = self.best_params if self.best_params else "default"
        return f"{self.name} [{status}] | params: {params}"
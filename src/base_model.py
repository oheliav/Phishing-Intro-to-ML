# =============================================================================
# base_model.py
# Abstract base class that every model file must implement.
# Enforces a consistent interface: train() and evaluate().
# Tuning is handled separately in regularize.py.
# =============================================================================

from abc import ABC, abstractmethod
from typing import Optional


class BaseModel(ABC):

    def __init__(self, name: str):
        self.name = name
        self.model: Optional[object] = None
        self.is_trained = False

    @abstractmethod
    def train(self, X_train, y_train, X_val=None, y_val=None):
        pass

    @abstractmethod
    def evaluate(self, X, y) -> dict:
        pass

    def _checkTrained(self):
        if not self.is_trained:
            raise RuntimeError(f"Model '{self.name}' has not been trained yet. Call train() first.")

    def __repr__(self):
        status = "trained" if self.is_trained else "untrained"
        return f"{self.name} [{status}]"
# =============================================================================
# linear_models.py
# Linear classifiers:
#   - Logistic Regression (sklearn)
#   - Single Layer Perceptron with Pocket Algorithm (from scratch)
# =============================================================================

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from base_model import BaseModel
from utils import computeMetrics, Timer
from config import RANDOM_SEED


class LogisticRegressionModel(BaseModel):

    def __init__(self):
        super().__init__("Logistic Regression")
        self.model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        with Timer() as t:
            self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Trained in {t}")

    def evaluate(self, X, y) -> dict:
        self._checkTrained()
        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        metrics = computeMetrics(y, y_pred, y_proba)
        metrics["model"] = self.name
        return metrics


class PocketModel(BaseModel):
    """
    Single Layer Perceptron with Pocket Algorithm.

    Each iteration picks a random misclassified point and updates weights.
    Best weights seen (lowest train error) are kept in the pocket.
    w^T x is used as the continuous score for AUC computation.
    Labels converted internally from {0,1} to {-1,+1}.
    """

    def __init__(self, epochs=1000):
        super().__init__("SLP Pocket")
        self.epochs = epochs
        self.w_ = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        X    = np.array(X_train)
        y    = np.array(y_train)
        y_pm = np.where(y == 1, 1, -1)
        N, d = X.shape
        X_b  = np.hstack([np.ones((N, 1)), X])

        rng      = np.random.default_rng(RANDOM_SEED)
        w        = rng.normal(0, 0.01, d + 1)
        best_w   = w.copy()
        best_err = np.mean(np.sign(X_b @ w) != y_pm)

        with Timer() as t:
            for _ in range(self.epochs):
                preds         = np.sign(X_b @ w)
                preds[preds == 0] = 1
                misclassified = np.where(preds != y_pm)[0]
                if len(misclassified) == 0:
                    break
                idx = rng.choice(misclassified)
                w   = w + y_pm[idx] * X_b[idx]
                err = np.mean(np.sign(X_b @ w) != y_pm)
                if err < best_err:
                    best_err = err
                    best_w   = w.copy()

        self.w_         = best_w
        self.is_trained = True
        print(f"[{self.name}] Trained in {t} | Best train error: {best_err:.4f}")

    def evaluate(self, X, y) -> dict:
        self._checkTrained()
        X    = np.array(X)
        X_b  = np.hstack([np.ones((X.shape[0], 1)), X])
        scores  = X_b @ self.w_                          # continuous score for AUC
        y_pred  = np.where(scores >= 0, 1, 0)            # hard prediction
        metrics = computeMetrics(y, y_pred, scores)
        metrics["model"] = self.name
        return metrics



# =============================================================================
# Majority Class Baseline
# =============================================================================

class MajorityClassModel(BaseModel):

    def __init__(self):
        super().__init__("Majority Class")
        self.model = DummyClassifier(strategy="most_frequent")

    def train(self, X_train, y_train, X_val=None, y_val=None):
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Trained")

    def evaluate(self, X, y) -> dict:
        self._checkTrained()
        y_pred  = self.model.predict(X)
        metrics = computeMetrics(y, y_pred, y_proba=None)
        metrics["model"] = self.name
        return metrics


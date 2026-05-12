# =============================================================================
# knn.py
# K-Nearest Neighbors classifier (sklearn):
#   - Default k=5
#   - Distance-based, no training phase
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

from base_model import BaseModel
from utils import computeMetrics, Timer
from config import RANDOM_SEED


class KNNModel(BaseModel):

    def __init__(self, k=229):
        super().__init__("kNN")
        self.k     = k
        self.model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)

    def train(self, X_train, y_train, X_val=None, y_val=None):
        # kNN has no real training phase — it just stores the training data
        with Timer() as t:
            self.model.fit(X_train, y_train)
        self.is_trained = True
        print(f"[{self.name}] Fitted in {t} (k={self.k})")

    def evaluate(self, X, y) -> dict:
        self._checkTrained()
        y_pred  = self.model.predict(X)
        y_proba = self.model.predict_proba(X)[:, 1]
        metrics = computeMetrics(y, y_pred, y_proba)
        metrics["model"] = self.name
        return metrics

    def tune(self, X_train, y_train) -> dict:
        raise NotImplementedError
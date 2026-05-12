# =============================================================================
# svm.py
# Support Vector Machine classifiers (sklearn):
#   - Linear SVM
#   - RBF Kernel SVM (used in optimization phase)
# =============================================================================

from sklearn.svm import SVC

from base_model import BaseModel
from utils import computeMetrics, Timer
from config import RANDOM_SEED


class LinearSVMModel(BaseModel):

    def __init__(self):
        super().__init__("SVM (Linear)")
        self.model = SVC(kernel="linear", probability=True, random_state=RANDOM_SEED)

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

    def tune(self, X_train, y_train) -> dict:
        raise NotImplementedError


class RBFSVMModel(BaseModel):

    def __init__(self):
        super().__init__("SVM (RBF)")
        self.model = SVC(kernel="rbf", probability=True, random_state=RANDOM_SEED)

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

    def tune(self, X_train, y_train) -> dict:
        raise NotImplementedError
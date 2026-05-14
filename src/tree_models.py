# =============================================================================
# tree_models.py
# Tree-based classifiers (sklearn / LightGBM):
#   - Random Forest
#   - LightGBM
# =============================================================================

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

from base_model import BaseModel
from utils import computeMetrics, Timer
from config import RANDOM_SEED, CV_FOLDS, CV_SCORING


class RandomForestModel(BaseModel):

    def __init__(self):
        super().__init__("Random Forest")
        self.model = RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1)

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


class LightGBMModel(BaseModel):

    def __init__(self):
        super().__init__("LightGBM")
        self.model = lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=-1, verbose=-1)

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


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

    def tune(self, X_merged, y_merged) -> dict:
        best = {"random_state": RANDOM_SEED, "n_jobs": 2}

        for param, values in [
            ("n_estimators",    [50, 100, 200, 300, 500]),
            ("max_depth",       [5, 10, 20, 30, None]),
            ("min_samples_leaf",[1, 2, 4, 8]),
            ("max_features",    ["sqrt", "log2", 0.3, 0.5]),
        ]:
            gs = GridSearchCV(
                RandomForestClassifier(**best),
                {param: values},
                cv=10, scoring=CV_SCORING, n_jobs=2,
            )
            gs.fit(X_merged, y_merged)
            best[param] = gs.best_params_[param]
            print(f"Best {param}: {best[param]} | AUC: {gs.best_score_:.4f}")

        self.best_params = best
        self.model = RandomForestClassifier(**best)
        self.model.fit(X_merged, y_merged)
        self.is_trained = True
        print(f"\n[{self.name}] Final params: {self.best_params}")
        return self.best_params


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

    def tune(self, X_merged, y_merged) -> dict:
        best = {"random_state": RANDOM_SEED, "n_jobs": 2, "verbose": -1}

        for param, values in [
            ("n_estimators",    [50, 100, 200, 300]),
            ("num_leaves",      [15, 31, 63, 127]),
            ("learning_rate",   [0.01, 0.05, 0.1, 0.2]),
            ("min_child_samples",[10, 20, 50, 100]),
        ]:
            gs = GridSearchCV(
                lgb.LGBMClassifier(**best),
                {param: values},
                cv=10, scoring=CV_SCORING, n_jobs=2,
            )
            gs.fit(X_merged, y_merged)
            best[param] = gs.best_params_[param]
            print(f"Best {param}: {best[param]} | AUC: {gs.best_score_:.4f}")

        self.best_params = best
        self.model = lgb.LGBMClassifier(**best)
        self.model.fit(X_merged, y_merged)
        self.is_trained = True
        print(f"\n[{self.name}] Final params: {self.best_params}")
        return self.best_params
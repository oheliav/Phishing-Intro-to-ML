# =============================================================================
# regularize.py
# Hyperparameter tuning via full grid search with 10-fold CV.
# Selects best params by minimizing E_CV (1 - accuracy).
# Also reports CV AUC as additional metric.
# =============================================================================

from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import lightgbm as lgb
import numpy as np

from config import RANDOM_SEED

# -----------------------------------------------------------------------------
# Parameter Grids
# -----------------------------------------------------------------------------

RF_PARAM_GRID = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [10, 20, None],
    "min_samples_leaf": [1, 2, 4],
    "max_features":     ["sqrt", "log2"],
}

LGBM_PARAM_GRID = {
    "n_estimators":      [100, 200, 300],
    "num_leaves":        [31, 63, 127],
    "learning_rate":     [0.05, 0.1, 0.2],
    "min_child_samples": [10, 20, 50],
}

NN_PARAM_GRID = {
    "alpha":              [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.01],
}

# -----------------------------------------------------------------------------
# Estimator Factories
# -----------------------------------------------------------------------------

def _makeEstimator(name):
    if name == "Random Forest":
        return RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=2)
    elif name == "LightGBM":
        return lgb.LGBMClassifier(random_state=RANDOM_SEED, n_jobs=2, verbose=-1)
    elif name == "NN":
        return MLPClassifier(hidden_layer_sizes=(32,), activation="relu", random_state=RANDOM_SEED, max_iter=500)
    raise ValueError(f"Unknown model: {name}")

# -----------------------------------------------------------------------------
# Tuning Function
# -----------------------------------------------------------------------------

def tuneModel(name, param_grid, X_merged, y_merged):
    """
    Full grid search with 10-fold CV on X_merged.
    Selects best params by minimizing E_CV (1 - accuracy).
    Also reports CV AUC for the best configuration.
    Retrains best estimator on full X_merged.

    Args:
        name       : "Random Forest", "LightGBM", or "NN"
        param_grid : parameter grid dict
        X_merged, y_merged : full train+val data

    Returns:
        best_estimator, best_params, cv_results dict
    """
    gs = GridSearchCV(
        _makeEstimator(name),
        param_grid,
        cv=10,
        scoring="accuracy",
        n_jobs=4,
        refit=True,
        verbose=1,
    )
    gs.fit(X_merged, y_merged)

    best_acc = gs.best_score_
    best_std = gs.cv_results_["std_test_score"][gs.best_index_]
    e_cv     = round(1 - best_acc, 4)

    auc_scores = cross_val_score(gs.best_estimator_, X_merged, y_merged, cv=10, scoring="roc_auc", n_jobs=4)

    print(f"\n[{name}] Best params:  {gs.best_params_}")
    print(f"[{name}] CV Accuracy:  {best_acc:.4f} ± {best_std:.4f}")
    print(f"[{name}] E_CV:         {e_cv:.4f}")
    print(f"[{name}] CV AUC:       {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")

    return gs.best_estimator_, gs.best_params_, {
        "cv_accuracy": round(best_acc, 4),
        "cv_std":      round(best_std, 4),
        "e_cv":        e_cv,
        "cv_auc":      round(auc_scores.mean(), 4),
        "params":      gs.best_params_,
    }
# =============================================================================
# regularize.py
# Sequential hyperparameter search — one parameter at a time.
# 10-fold CV, no parallelism. Reports E_CV and AUC mean ± std.
# Saves best model to models/ directory.
# =============================================================================

import os
import joblib
import copy
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
from sklearn.neural_network import MLPClassifier
from config import RANDOM_SEED

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def tuneRF(X_merged, y_merged):
    best = {}

    for param, values in [
        ("n_estimators",     [100, 500, 700]),
        ("max_depth",        [5, 10, 20, None]),
        ("min_samples_leaf", [1, 2, 4, 8]),
        ("max_features",     ["sqrt", "log2"]),
    ]:
        gs = GridSearchCV(
            RandomForestClassifier(**best, random_state=RANDOM_SEED),
            {param: values},
            cv=10,
            scoring="accuracy",
            n_jobs=-1,
        )
        gs.fit(X_merged, y_merged)
        best[param] = gs.best_params_[param]
        std = gs.cv_results_["std_test_score"][gs.best_index_]
        e_cv = round(1 - gs.best_score_, 4)
        print(f"Best {param}: {best[param]} | Accuracy: {gs.best_score_:.4f} ± {std:.4f} | E_CV: {e_cv:.4f}")

    # Final model with best params, retrain on full X_merged
    final = RandomForestClassifier(**best, random_state=RANDOM_SEED)
    final.fit(X_merged, y_merged)

    # CV AUC on final model
    auc = cross_val_score(final, X_merged, y_merged, cv=10, scoring="roc_auc", n_jobs=-1)
    print(f"\n[RF] Final params: {best}")
    print(f"[RF] CV AUC: {auc.mean():.4f} ± {auc.std():.4f}")

    joblib.dump(final, os.path.join(MODELS_DIR, "rf_tuned.pkl"))
    print(f"[RF] Saved.")
    return final, best


def tuneLGBM(X_merged, y_merged):
    best = {}

    for param, values in [
        ("n_estimators",      [100, 300, 500]),
        ("num_leaves",        [15, 31, 63, 127]),
        ("learning_rate",     [0.05, 0.1, 0.2]),
        ("min_child_samples", [10, 20, 50]),
        ("reg_alpha",         [0, 0.1, 1.0]),
        ("reg_lambda",        [0, 0.1, 1.0]),
    ]:
        gs = GridSearchCV(
            lgb.LGBMClassifier(**best, random_state=RANDOM_SEED, verbose=-1),
            {param: values},
            cv=10,
            scoring="accuracy",
            n_jobs=-1,
        )
        gs.fit(X_merged, y_merged)
        best[param] = gs.best_params_[param]
        std = gs.cv_results_["std_test_score"][gs.best_index_]
        e_cv = round(1 - gs.best_score_, 4)
        print(f"Best {param}: {best[param]} | Accuracy: {gs.best_score_:.4f} ± {std:.4f} | E_CV: {e_cv:.4f}")

    final = lgb.LGBMClassifier(**best, random_state=RANDOM_SEED, verbose=-1)
    final.fit(X_merged, y_merged)

    auc = cross_val_score(final, X_merged, y_merged, cv=10, scoring="roc_auc", n_jobs=-1)
    print(f"\n[LightGBM] Final params: {best}")
    print(f"[LightGBM] CV AUC: {auc.mean():.4f} ± {auc.std():.4f}")

    joblib.dump(final, os.path.join(MODELS_DIR, "lgbm_tuned.pkl"))
    print(f"[LightGBM] Saved.")
    return final, best


def loadModel(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    model = joblib.load(path)
    print(f"Loaded {name} from {path}")
    return model


def tuneNNWeightDecay(X_merged, y_merged):
    """
    Weight decay regularization for NN.
    Minimizes E_aug = E_in + (lambda/N) * sum(w^2).
    sklearn's alpha parameter corresponds to lambda.
    Selected via 10-fold CV on X_merged.
    Retrains best model on full X_merged.
    """

    best = {}

    for param, values in [
        ("alpha",              [0.0001, 0.001, 0.01, 0.1]),
        ("learning_rate_init", [0.001, 0.01]),
    ]:
        gs = GridSearchCV(
            MLPClassifier(**best,
                          hidden_layer_sizes=(32,),
                          activation="relu",
                          max_iter=500,
                          random_state=RANDOM_SEED),
            {param: values},
            cv=10,
            scoring="accuracy",
            n_jobs=-1,
        )
        gs.fit(X_merged, y_merged)
        best[param] = gs.best_params_[param]
        std  = gs.cv_results_["std_test_score"][gs.best_index_]
        e_cv = round(1 - gs.best_score_, 4)
        print(f"Best {param}: {best[param]} | Accuracy: {gs.best_score_:.4f} ± {std:.4f} | E_CV: {e_cv:.4f}")

    final = MLPClassifier(**best,
                          hidden_layer_sizes=(32,),
                          activation="relu",
                          max_iter=500,
                          random_state=RANDOM_SEED)
    final.fit(X_merged, y_merged)

    auc = cross_val_score(final, X_merged, y_merged, cv=10, scoring="roc_auc", n_jobs=-1)
    print(f"\n[NN Weight Decay] Final params: {best}")
    print(f"[NN Weight Decay] CV AUC: {auc.mean():.4f} ± {auc.std():.4f}")

    joblib.dump(final, os.path.join(MODELS_DIR, "nn_wd_tuned.pkl"))
    print(f"[NN Weight Decay] Saved.")
    return final, best


def trainNNEarlyStopping(X_merged, y_merged):
    from sklearn.neural_network import MLPClassifier

    model = MLPClassifier(
        hidden_layer_sizes=(32,),
        activation="relu",
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=10,
        max_iter=500,
        random_state=RANDOM_SEED,
    )
    model.fit(np.array(X_merged), np.array(y_merged))

    print(f"[NN Early Stopping] Stopped at iteration: {model.n_iter_}")
    print(f"[NN Early Stopping] Best val accuracy: {model.best_validation_score_:.4f}")

    joblib.dump(model, os.path.join(MODELS_DIR, "nn_es.pkl"))
    print(f"[NN Early Stopping] Saved.")
    return model
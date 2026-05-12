# =============================================================================
# config.py
# Central configuration for all constants, paths, and hyperparameter grids.
# All other modules import from here — do not hardcode values elsewhere.
# =============================================================================

import os

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset_full.csv")

# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
RANDOM_SEED = 42

# -----------------------------------------------------------------------------
# Data Split
# Following Abu-Mostafa et al. (2012, p. 140, Section 4.3.1):
# "A rule of thumb in practice is to set K = N/5 (set aside 20% of the data
# for validation)." We apply this to both validation and test sets.
# Split: 60% train | 20% validation | 20% test
# -----------------------------------------------------------------------------
TEST_SIZE    = 0.20
VAL_SIZE     = 0.20   # fraction of the full dataset (not of the train remainder)

# -----------------------------------------------------------------------------
# Cross-Validation (used during grid search for hyperparameter tuning)
# -----------------------------------------------------------------------------
CV_FOLDS = 10
CV_SCORING = "roc_auc"

# -----------------------------------------------------------------------------
# Target column
# -----------------------------------------------------------------------------
TARGET_COL = "phishing"

# -----------------------------------------------------------------------------
# Correlation-based redundancy filter threshold
# Features with pairwise Pearson correlation above this are filtered.
# Based on prior work showing t=0.9 retains performance while reducing volume.
# -----------------------------------------------------------------------------
CORRELATION_THRESHOLD = 0.90

# -----------------------------------------------------------------------------
# Hyperparameter Grids (Grid Search)
# -----------------------------------------------------------------------------

PARAM_GRIDS = {

    "logistic_regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l1", "l2"],
        "solver": ["liblinear"],
        "max_iter": [1000],
    },

    "linear_svm": {
        "C": [0.001, 0.01, 0.1, 1, 10],
        "max_iter": [2000],
    },

    "svm_rbf": {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", "auto", 0.001, 0.01],
        "kernel": ["rbf"],
    },

    "decision_tree": {
        "max_depth": [3, 5, 10, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "criterion": ["gini", "entropy"],
    },

    "random_forest": {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    },

    "xgboost": {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 6, 9],
        "learning_rate": [0.01, 0.1, 0.3],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 5],
    },

    "lightgbm": {
        "n_estimators": [100, 200, 300],
        "num_leaves": [31, 63, 127],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 1.0],
        "colsample_bytree": [0.7, 1.0],
        "reg_alpha": [0, 0.1, 1],
        "reg_lambda": [1, 5],
    },

    "mlp": {
        "hidden_layer_sizes": [(64,), (128,), (64, 64), (128, 64)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],   # L2 regularization
        "learning_rate_init": [0.001, 0.01],
        "max_iter": [200],
        "early_stopping": [True],
        "validation_fraction": [0.1],
    },

    "knn": {
        "n_neighbors": [3, 5, 7, 11, 15],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    },
}

# -----------------------------------------------------------------------------
# Early stopping config (for XGBoost and LightGBM native API, if used)
# -----------------------------------------------------------------------------
EARLY_STOPPING_ROUNDS = 20
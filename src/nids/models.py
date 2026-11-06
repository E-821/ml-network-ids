"""Model definitions. Hyperparameters are deliberately close to library defaults so
results reflect the data and the evaluation protocol, not heavy tuning."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

MODEL_NAMES = ["logreg", "random_forest", "hist_gb"]

DISPLAY_NAMES = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "hist_gb": "Gradient Boosting",
}


def signed_log1p(X: np.ndarray) -> np.ndarray:
    """Compress heavy-tailed flow statistics (bytes/s, durations in microseconds)
    while keeping the sign of the few negative values in the raw data."""
    return np.sign(X) * np.log1p(np.abs(X))


def make_model(name: str, seed: int = 42, n_jobs: int = -1):
    if name == "logreg":
        # Linear baseline: needs scaling; trees below do not.
        return make_pipeline(
            FunctionTransformer(signed_log1p),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=100, n_jobs=n_jobs, random_state=seed)
    if name == "hist_gb":
        # l2_regularization matters here: without it the multiclass model diverges
        # on the tiny classes (Heartbleed, SQLi, Infiltration). Validation log-loss
        # went from 0.51 to 38.6 within 10 iterations and macro-F1 fell to 0.66.
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )
    raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}")

"""Candidate regression models for the pace prediction."""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(random_state: int = 42) -> dict:
    """Build the candidate regression pipelines.

    Each model is wrapped in a ``Pipeline`` whose first step is a
    ``StandardScaler``, so that the scaling is fitted on the training fold
    only.

    Parameters
    ----------
    random_state : int, optional
        Seed passed to the stochastic estimators. Default is 42.

    Returns
    -------
    dict
        Unfitted pipelines keyed by name: ``"linear"``, ``"lasso"``,
        ``"ridge"``, ``"random_forest"`` and ``"hist_gbm"``.
    """
    # Le scaler est dans le Pipeline : moyenne/écart-type calculés sur le train
    # de chaque fold uniquement (équivalent de ton scale.values en R, sans fuite).
    return {
        "linear": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LinearRegression()),
            ]
        ),
        "lasso": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "lasso",
                    LassoCV(
                        alphas=100,
                        max_iter=50_000,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "ridge": Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", RidgeCV(alphas=np.logspace(-2, 3, 30))),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        min_samples_leaf=5,
                        max_features=0.5,
                        n_jobs=-1,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "hist_gbm": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        learning_rate=0.05,
                        max_iter=300,
                        max_depth=4,
                        min_samples_leaf=20,
                        l2_regularization=1.0,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_models(random_state: int = 42) -> dict:
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

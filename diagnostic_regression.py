"""Diagnostics for grouped trail-running regression models.

The script is also suitable for execution cell-by-cell in a notebook:

    exec(Path("diagnostic_regression.py").read_text())

It intentionally loads the computed CSV directly so that min/max scaling in
Dataset does not happen before the train/test split.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DATA_PATH = Path("data/computed/dataset_18_100_50.csv")
TARGET = "pace"
RANDOM_STATE = 42
N_SPLITS = 5

TARGET_COLUMNS = {"pace", "speed", "log_pace", "log_speed"}
META_COLUMNS = {
    "activity_id",
    "name",
    "segment",
    "timestamp",
    "elapsed_time",
    "position_lat",
    "position_long",
}


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the computed dataset and restore temporal dtypes."""
    data = pd.read_csv(path)
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["elapsed_time"] = pd.to_timedelta(data["elapsed_time"])
    return data


def print_integrity_report(data: pd.DataFrame) -> None:
    """Print dimensions, missing values, infinities and target diagnostics."""
    numeric = data.select_dtypes(include="number")
    infinity_counts = {
        column: int(np.isinf(values).sum())
        for column, values in numeric.items()
        if np.isinf(values).any()
    }

    print("Shape:", data.shape)
    print("Activities:", data["activity_id"].nunique())
    print("Missing values:", int(data.isna().sum().sum()))
    print("Infinite values:", infinity_counts)
    print("Duplicate rows:", int(data.duplicated().sum()))
    print(
        "Duplicate activity/segment pairs:",
        int(data.duplicated(["activity_id", "segment"]).sum()),
    )

    cardinality = numeric.nunique(dropna=False).sort_values()
    print("Constant columns:", cardinality[cardinality <= 1].to_dict())
    print("Columns with <= 3 distinct values:", cardinality[cardinality <= 3].to_dict())

    for column in ["pace", "speed", "log_pace", "log_speed"]:
        values = data[column].replace([np.inf, -np.inf], np.nan).dropna()
        print(
            f"{column}: "
            f"min={values.min():.6g}, "
            f"q01={values.quantile(0.01):.6g}, "
            f"median={values.median():.6g}, "
            f"q99={values.quantile(0.99):.6g}, "
            f"max={values.max():.6g}, "
            f"skew={values.skew():.6g}"
        )

    print("pace <= 0:", int((data["pace"] <= 0).sum()))
    print("pace > 30 min/km:", int((data["pace"] > 30).sum()))
    print("speed <= 0:", int((data["speed"] <= 0).sum()))
    print("speed > 8 m/s:", int((data["speed"] > 8).sum()))


def filter_model_rows(
    data: pd.DataFrame,
    target: str = TARGET,
    max_pace: float = 30.0,
    max_speed: float = 8.0,
) -> pd.DataFrame:
    """Remove invalid and implausible target rows before model fitting."""
    valid = (
        np.isfinite(data[target])
        & np.isfinite(data["speed"])
        & (data[target] > 0)
        & (data["speed"] > 0)
        & (data[target] <= max_pace)
        & (data["speed"] <= max_speed)
    )
    return data.loc[valid].reset_index(drop=True)


def select_features(data: pd.DataFrame) -> list[str]:
    """Select numeric predictors while excluding metadata and target columns."""
    excluded = META_COLUMNS | TARGET_COLUMNS
    return [
        column
        for column in data.columns
        if column not in excluded
        and pd.api.types.is_numeric_dtype(data[column])
    ]


def print_feature_diagnostics(data: pd.DataFrame, features: list[str]) -> None:
    """Print rank, condition number and highly correlated feature pairs."""
    matrix = data[features].replace([np.inf, -np.inf], np.nan).dropna()
    matrix = matrix.loc[:, matrix.nunique() > 1]
    standardized = StandardScaler().fit_transform(matrix)
    singular_values = np.linalg.svd(standardized, compute_uv=False)

    print("Design rank:", np.linalg.matrix_rank(standardized), "/", matrix.shape[1])
    print("Condition number:", singular_values[0] / singular_values[-1])

    correlation = matrix.corr().abs()
    pairs = []
    for first in range(len(correlation)):
        for second in range(first):
            value = correlation.iloc[first, second]
            if value > 0.98:
                pairs.append(
                    (
                        correlation.index[first],
                        correlation.index[second],
                        round(float(value), 4),
                    )
                )
    print("Correlation pairs > 0.98:", pairs)


def check_window_features(data: pd.DataFrame, activity_id: int | None = None) -> None:
    """Compare one front/back window with values calculated by hand."""
    if activity_id is None:
        activity_id = int(data["activity_id"].iloc[0])

    activity = (
        data.loc[data["activity_id"] == activity_id]
        .sort_values("segment")
        .reset_index(drop=True)
    )
    positive = activity["altitude_drop_pos"].to_numpy()
    window_size = 1000 // 50

    expected_front = [
        positive[index + 1 : index + 1 + window_size].sum()
        for index in range(min(4, len(activity)))
    ]
    expected_back = [
        positive[max(0, index - window_size) : index].sum()
        for index in range(min(4, len(activity)))
    ]

    print("Activity:", activity_id)
    print("Actual front_1000:", activity["altitude_drop_pos_front_1000"].head(4).tolist())
    print("Expected front_1000:", expected_front)
    print("Actual back_1000:", activity["altitude_drop_pos_back_1000"].head(4).tolist())
    print("Expected back_1000:", expected_back)
    print(
        "done + remaining - total at boundaries:",
        (
            activity["altitude_drop_pos_done"]
            + activity["altitude_drop_pos_remaining"]
            - activity["altitude_drop_pos_total"]
        ).iloc[[0, -1]].tolist(),
    )


def make_preprocess(features: list[str]) -> ColumnTransformer:
    """Build a fold-safe scaler for every numeric predictor."""
    return ColumnTransformer(
        [("scale", StandardScaler(), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")


def make_pipeline(
    estimator: object,
    features: list[str],
) -> Pipeline:
    return Pipeline(
        [
            ("preprocess", make_preprocess(features)),
            ("model", estimator),
        ]
    )


def evaluate_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    features: list[str],
) -> None:
    """Compare linear and Ridge models with grouped and random CV."""
    estimators = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
    }

    for name, estimator in estimators.items():
        for split_name, cv, cv_groups in [
            ("group", GroupKFold(N_SPLITS), groups),
            ("random", KFold(N_SPLITS, shuffle=True, random_state=RANDOM_STATE), None),
        ]:
            model = make_pipeline(estimator, features)
            prediction = cross_val_predict(
                model,
                X,
                y,
                cv=cv,
                groups=cv_groups,
            )
            fitted = model.fit(X, y)
            print(
                name,
                split_name,
                {
                    "r2": round(r2_score(y, prediction), 6),
                    "mae": round(mean_absolute_error(y, prediction), 6),
                    "train_r2": round(fitted.score(X, y), 6),
                },
            )

    baseline = np.full(len(y), y.mean())
    print(
        "baseline_mean",
        {
            "r2": round(r2_score(y, baseline), 6),
            "mae": round(mean_absolute_error(y, baseline), 6),
        },
    )


def fit_tuned_models(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    features: list[str],
) -> tuple[GridSearchCV, GridSearchCV]:
    """Tune Ridge and Lasso with activity-aware inner cross-validation."""
    cv = GroupKFold(N_SPLITS)

    ridge_search = GridSearchCV(
        make_pipeline(Ridge(), features),
        {"model__alpha": np.logspace(-4, 4, 50)},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    lasso_search = GridSearchCV(
        make_pipeline(
            Lasso(max_iter=100_000, random_state=RANDOM_STATE),
            features,
        ),
        {"model__alpha": np.logspace(-4, 1, 80)},
        cv=cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        ridge_search.fit(X, y, groups=groups)
        lasso_search.fit(X, y, groups=groups)

    print("Ridge alpha:", ridge_search.best_params_)
    print("Ridge CV MAE:", -ridge_search.best_score_)
    print("Lasso alpha:", lasso_search.best_params_)
    print("Lasso CV MAE:", -lasso_search.best_score_)
    print("Lasso convergence warnings:", len(caught))

    lasso = lasso_search.best_estimator_
    names = lasso["preprocess"].get_feature_names_out()
    coefficients = pd.Series(lasso["model"].coef_, index=names)
    print("Lasso non-zero coefficients:", int((coefficients != 0).sum()))
    print(coefficients.sort_values(key=np.abs, ascending=False))

    return ridge_search, lasso_search


def save_and_check_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    output_path: Path = Path("ridge_pipeline.joblib"),
) -> None:
    """Fit, serialize and verify prediction equality after reloading."""
    first = model.fit(X, y)
    prediction = first.predict(X)
    joblib.dump(first, output_path)
    loaded = joblib.load(output_path)
    loaded_prediction = loaded.predict(X)
    print("Reloaded predictions identical:", np.array_equal(prediction, loaded_prediction))


def main() -> None:
    data = load_data()
    print_integrity_report(data)

    clean = filter_model_rows(data)
    features = select_features(clean)
    X = clean[features]
    y = clean[TARGET]
    groups = clean["activity_id"]

    print("Rows after filtering:", len(clean))
    print("Features:", len(features))
    print_feature_diagnostics(clean, features)
    check_window_features(clean)
    evaluate_models(X, y, groups, features)

    ridge_search, _ = fit_tuned_models(X, y, groups, features)
    save_and_check_model(ridge_search.best_estimator_, X, y)


if __name__ == "__main__":
    main()

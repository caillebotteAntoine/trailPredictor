"""Tests for :mod:`TrailPredictor.modeling.lasso`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError
from sklearn import config_context
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling import lasso as lasso_module
from TrailPredictor.modeling.lasso import (
    SELECTION_COLUMNS,
    LassoConfig,
    ModelingLasso,
    selection_frequency,
)
from TrailPredictor.modeling.models import Modeling

SEED = 0
N_ACTIVITIES = 6
N_SEGMENTS = 8
N_SPLITS = 2  # outer folds: 3 training activities each
INNER_N_SPLITS = 2
MAX_STEP_M = 15  # altitude change between two segments, in metres
SLOPE_SPEED_FACTOR = 0.05  # m/s lost per metre climbed in a segment
BASE_SPEED = 3.0  # m/s


@pytest.fixture
def lasso_data(make_segments, make_dataset) -> pd.DataFrame:
    """Activities whose speed decreases with the climb of each segment."""
    rng = np.random.default_rng(SEED)
    activities = {}
    for activity_id in range(N_ACTIVITIES):  # few activities: loop over inputs
        steps = rng.integers(-MAX_STEP_M, MAX_STEP_M + 1, size=N_SEGMENTS)
        steps[0] = 0
        # climb and descend in every activity: a zero total climb or descent
        # makes the done / remaining ratios 0 / 0 = NaN in build_features
        steps[1:3] = [MAX_STEP_M, -MAX_STEP_M]
        altitude = 100.0 + np.cumsum(steps)
        speed = BASE_SPEED - SLOPE_SPEED_FACTOR * steps
        activities[activity_id] = make_segments(altitude, speed=speed)
    return make_dataset(activities)


@pytest.fixture
def config() -> LassoConfig:
    return LassoConfig(n_splits=N_SPLITS, inner_n_splits=INNER_N_SPLITS, n_alphas=20)


@pytest.fixture
def model(
    lasso_data: pd.DataFrame, config: LassoConfig, tmp_path: Path
) -> ModelingLasso:
    return ModelingLasso(Dataset(lasso_data), config, models_dir=tmp_path)


# --- LassoConfig


def test_config_defaults() -> None:
    config = LassoConfig()

    assert (config.inner_n_splits, config.n_alphas, config.max_iter) == (5, 100, 10_000)
    assert config.selection_threshold == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("inner_n_splits", 1),
        ("n_alphas", 0),
        ("selection_threshold", -1.0),
        ("max_iter", 0),
    ],
)
def test_config_invalid_field_raises(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        LassoConfig(**{field: value})


def test_config_unknown_field_raises() -> None:
    with pytest.raises(ValidationError, match="scale_all"):
        LassoConfig(scale_all=True)


def test_config_key_differs_from_base_fields_only() -> None:
    assert LassoConfig().key != LassoConfig(inner_n_splits=3).key


# --- pipeline


def test_pipeline_steps(model: ModelingLasso) -> None:
    pipeline = model.build_pipeline()

    assert [name for name, _ in pipeline.steps] == [
        "preprocess",
        "scale",
        "selection",
        "regression",
    ]
    assert isinstance(pipeline.named_steps["regression"], LinearRegression)


def test_selection_finds_the_informative_columns(model: ModelingLasso) -> None:
    rng = np.random.default_rng(SEED)
    n_rows, n_groups = 300, 10
    predictors = pd.DataFrame(
        rng.normal(size=(n_rows, 6)), columns=[f"x{i}" for i in range(6)]
    )
    # informative columns on very different scales: selection must not depend on it
    predictors["x0"] *= 100.0
    target = (
        0.03 * predictors["x0"]
        - 2.0 * predictors["x1"]
        + rng.normal(scale=0.1, size=n_rows)
    )
    groups = np.repeat(np.arange(n_groups), n_rows // n_groups)
    pipeline = Pipeline(
        [*model.make_pipeline_steps(), ("regression", LinearRegression())]
    )

    with config_context(enable_metadata_routing=True):
        pipeline.fit(predictors, target, groups=groups)

    selector = pipeline.named_steps["selection"]
    abs_coef = pd.Series(np.abs(selector.estimator_.coef_), index=predictors.columns)
    assert selector.get_support()[:2].all()
    assert abs_coef[["x0", "x1"]].min() > abs_coef.drop(["x0", "x1"]).max()


def test_inner_folds_require_groups(model: ModelingLasso) -> None:
    pipeline = model.build_pipeline()

    with config_context(enable_metadata_routing=True):
        with pytest.raises(ValueError, match="groups"):
            pipeline.fit(model.dataset.predictors, model.dataset.target)


def test_inner_folds_only_see_training_activities(
    model: ModelingLasso, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen_groups: list[int] = []

    class SpyGroupKFold(GroupKFold):
        """GroupKFold recording the number of groups of each split."""

        def split(self, X, y=None, groups=None):  # noqa: N803 (sklearn signature)
            seen_groups.append(len(np.unique(groups)))
            return super().split(X, y, groups)

    monkeypatch.setattr(lasso_module, "GroupKFold", SpyGroupKFold)

    model.cross_validate()

    # one LassoCV fit per outer fold, on its training activities only
    assert seen_groups == [N_ACTIVITIES - N_ACTIVITIES // N_SPLITS] * N_SPLITS


# --- evaluate_selection


def test_evaluate_selection_result(model: ModelingLasso) -> None:
    result = model.evaluate_selection()

    n_features = result["feature"].nunique()
    assert result.columns.tolist() == SELECTION_COLUMNS
    assert len(result) == N_SPLITS * n_features
    assert result.groupby("fold")["feature"].nunique().eq(n_features).all()
    assert (result["alpha"] > 0.0).all()
    assert (result["rmse"] > 0.0).all()
    # unselected features have a (near) zero Lasso coefficient
    unselected = result.loc[~result["selected"], "lasso_coef"].abs()
    assert (unselected < model.config.selection_threshold).all()


def test_evaluate_selection_keeps_the_climb(model: ModelingLasso) -> None:
    model.evaluate_selection()

    frequency = model.selection_frequency["frequency"]
    climb_features = frequency.loc[frequency.index.str.contains("drop")]
    assert climb_features.max() == pytest.approx(1.0)


def test_evaluate_selection_saves_and_reloads(
    model: ModelingLasso, tmp_path: Path
) -> None:
    result = model.evaluate_selection()

    reloaded = ModelingLasso(model.dataset, model.config, models_dir=tmp_path)

    assert reloaded.result is not None
    pd.testing.assert_frame_equal(reloaded.result, result)


def test_evaluate_selection_does_not_modify_dataset(model: ModelingLasso) -> None:
    before = model.dataset.data

    model.evaluate_selection()

    pd.testing.assert_frame_equal(model.dataset.data, before)


def test_modeling_factory(
    lasso_data: pd.DataFrame, config: LassoConfig, tmp_path: Path
) -> None:
    model = Modeling(Dataset(lasso_data), models_dir=tmp_path).lasso(config)

    assert isinstance(model, ModelingLasso)
    assert model.config is config
    assert model.result is None


# --- selection_frequency


def test_selection_frequency() -> None:
    selection = pd.DataFrame(
        {
            "feature": ["a", "b", "c", "a", "b", "c"],
            "selected": [True, True, False, True, False, False],
            "lasso_coef": [2.0, -0.5, 0.0, 1.0, 0.0, 0.0],
        }
    )

    frequency = selection_frequency(selection)

    assert frequency.index.tolist() == ["a", "b", "c"]
    np.testing.assert_allclose(frequency["frequency"], [1.0, 0.5, 0.0])
    np.testing.assert_allclose(frequency["mean_abs_coef"], [1.5, 0.25, 0.0])


def test_selection_frequency_missing_column_raises() -> None:
    with pytest.raises(ValueError, match="selected"):
        selection_frequency(pd.DataFrame({"feature": ["a"], "lasso_coef": [1.0]}))


def test_selection_frequency_without_result_raises(model: ModelingLasso) -> None:
    with pytest.raises(ValueError, match="no result"):
        _ = model.selection_frequency

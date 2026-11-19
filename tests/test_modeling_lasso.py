"""Tests for :mod:`TrailPredictor.modeling.lasso`."""

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import LASSO_N_ACTIVITIES, SEED
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
    LassoPath,
    ModelingLasso,
    selection_frequency,
    summarize_folds,
)
from TrailPredictor.modeling.models import Modeling

N_SPLITS = 2  # outer folds: 3 training activities each
INNER_N_SPLITS = 2
OBJECTIVE_RTOL = 1e-3  # coordinate descent stops at a duality-gap tolerance


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

        def split(self, X, y=None, groups=None):
            seen_groups.append(len(np.unique(groups)))
            return super().split(X, y, groups)

    monkeypatch.setattr(lasso_module, "GroupKFold", SpyGroupKFold)

    model.cross_validate()

    # one LassoCV fit per outer fold, on its training activities only
    assert (
        seen_groups == [LASSO_N_ACTIVITIES - LASSO_N_ACTIVITIES // N_SPLITS] * N_SPLITS
    )


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


# --- regularisation path


@pytest.fixture
def toy_path() -> LassoPath:
    """Three alphas, two features, two folds with hand-checked statistics."""
    return LassoPath(
        coef_path=pd.DataFrame(
            {"a": [0.0, 1.0, 2.0], "b": [0.0, 0.0, -0.5]}, index=[1.0, 0.1, 0.01]
        ),
        mse_path=np.array([[4.0, 6.0], [2.0, 2.4], [1.8, 2.2]]),
        alpha=0.01,
    )


def test_lasso_path_statistics(toy_path: LassoPath) -> None:
    np.testing.assert_allclose(toy_path.alphas, [1.0, 0.1, 0.01])
    np.testing.assert_allclose(toy_path.mse_mean, [5.0, 2.2, 2.0])
    np.testing.assert_allclose(
        toy_path.mse_se, [1.0 / np.sqrt(2), 0.2 / np.sqrt(2)] + [0.2 / np.sqrt(2)]
    )
    assert toy_path.mse_threshold == pytest.approx(2.0 + 0.2 / np.sqrt(2))
    # 2.2 <= 2.141 is false: the 1-SE rule keeps the selected alpha
    assert toy_path.alpha_1se == pytest.approx(0.01)
    assert toy_path.n_active.tolist() == [0, 1, 2]


def test_lasso_path_alpha_1se_picks_sparser_model() -> None:
    path = LassoPath(
        coef_path=pd.DataFrame({"a": [0.0, 1.0, 2.0]}, index=[1.0, 0.1, 0.01]),
        mse_path=np.array([[4.0, 6.0], [2.0, 2.2], [1.0, 3.0]]),
        alpha=0.01,
    )

    # threshold 2.0 + 1 / sqrt(2) > 2.1, the mean error at alpha = 0.1
    assert path.alpha_1se == pytest.approx(0.1)


@pytest.mark.parametrize(
    ("alpha", "expected"), [(0.5, [0.0, 0.0]), (0.02, [2.0, -0.5])]
)
def test_lasso_path_coefs_at_closest_log_alpha(
    toy_path: LassoPath, alpha: float, expected: list[float]
) -> None:
    np.testing.assert_allclose(toy_path.coefs_at(alpha), expected)


def test_lasso_path_misaligned_mse_raises() -> None:
    with pytest.raises(ValueError, match="does not match"):
        LassoPath(
            coef_path=pd.DataFrame({"a": [0.0, 1.0]}, index=[1.0, 0.1]),
            mse_path=np.ones((3, 2)),
            alpha=0.1,
        )


def test_fit_regularization_path(model: ModelingLasso) -> None:
    path = model.fit_regularization_path()

    n_features = path.coef_path.shape[1]
    assert path.mse_path.shape == (model.config.n_alphas, INNER_N_SPLITS)
    assert len(path.coef_path) == model.config.n_alphas
    assert (np.diff(path.alphas) < 0).all()
    assert path.threshold == pytest.approx(model.config.selection_threshold)
    # the strongest penalty zeroes every coefficient, the weakest keeps some
    assert path.n_active.iloc[0] == 0
    assert 0 < path.n_active.iloc[-1] <= n_features
    assert path.alphas.min() <= path.alpha <= path.alpha_1se <= path.alphas.max()


def test_fit_regularization_path_matches_lasso_cv(model: ModelingLasso) -> None:
    pipeline = Pipeline(
        [("preprocess", model.dataset.get_preprocess()), *model.make_pipeline_steps()]
    )
    with config_context(enable_metadata_routing=True):
        pipeline.fit(
            model.dataset.predictors,
            model.dataset.target,
            groups=model.dataset.data["activity_id"],
        )
    lasso_cv = pipeline.named_steps["selection"].estimator_
    design = pipeline[:-1].transform(model.dataset.predictors).to_numpy(dtype=float)
    design = design - design.mean(axis=0)
    response = model.dataset.target.to_numpy(dtype=float)
    response = response - response.mean()

    def lasso_objective(coefs: np.ndarray, alpha: float) -> float:
        residuals = response - design @ coefs
        return residuals @ residuals / (2 * len(response)) + alpha * np.abs(coefs).sum()

    path = model.fit_regularization_path()

    assert path.alpha == pytest.approx(lasso_cv.alpha_)
    np.testing.assert_allclose(path.mse_path, lasso_cv.mse_path_)
    # the test design is rank deficient, so the Lasso solution is not unique:
    # compare the objective reached, not the coefficients
    path_objective = lasso_objective(path.coefs_at(path.alpha).to_numpy(), path.alpha)
    assert path_objective <= lasso_objective(lasso_cv.coef_, path.alpha) * (
        1 + OBJECTIVE_RTOL
    )


def test_fit_regularization_path_does_not_modify_dataset(model: ModelingLasso) -> None:
    before = model.dataset.data

    model.fit_regularization_path()

    pd.testing.assert_frame_equal(model.dataset.data, before)


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


# --- summarize_folds


def test_summarize_folds_counts_selected_features() -> None:
    selection = pd.DataFrame(
        {
            "fold": [0, 0, 0, 1, 1, 1],
            "feature": ["a", "b", "c"] * 2,
            "lasso_coef": [1.0, 0.0, 2.0, 1.0, 0.0, 0.0],
            "selected": [True, False, True, True, False, False],
            "alpha": [0.1] * 3 + [0.2] * 3,
            "rmse": [1.5] * 3 + [2.5] * 3,
        }
    )

    folds = summarize_folds(selection)

    expected = pd.DataFrame(
        {"alpha": [0.1, 0.2], "rmse": [1.5, 2.5], "n_selected": [2, 1]},
        index=pd.Index([0, 1], name="fold"),
    )
    pd.testing.assert_frame_equal(folds, expected)


def test_summarize_folds_empty_selection() -> None:
    assert summarize_folds(pd.DataFrame(columns=SELECTION_COLUMNS)).empty


def test_summarize_folds_missing_column() -> None:
    with pytest.raises(ValueError, match="rmse"):
        summarize_folds(pd.DataFrame(columns=["fold", "selected", "alpha"]))


# --- list_saved_results


@pytest.fixture
def saved_model(lasso_data: pd.DataFrame, tmp_path: Path) -> ModelingLasso:
    """Lasso model with a (fake) saved result, without running the selection."""
    model = ModelingLasso(
        Dataset(lasso_data, name="lasso_dataset"),
        LassoConfig(n_splits=N_SPLITS, inner_n_splits=INNER_N_SPLITS),
        models_dir=tmp_path,
    )
    model.result = pd.DataFrame(columns=SELECTION_COLUMNS)
    model.save_result()
    return model


def test_list_saved_results_empty_dir(tmp_path: Path) -> None:
    assert not lasso_module.list_saved_results("missing", tmp_path)


def test_list_saved_results_finds_saved_model(
    saved_model: ModelingLasso, tmp_path: Path
) -> None:
    dataset_key = saved_model.dataset.key

    saved = lasso_module.list_saved_results(dataset_key, tmp_path)

    assert saved == [
        lasso_module.SavedLassoResult(
            saved_model.dataset.target_name,
            saved_model.config,
            saved_model.result_path,
            dataset_key=dataset_key,
            dataset_name="lasso_dataset",
        )
    ]


def test_list_saved_results_filters_by_dataset(
    saved_model: ModelingLasso, tmp_path: Path
) -> None:
    other_dir = tmp_path / ModelingLasso.name / "other_key"
    shutil.copytree(saved_model.result_path.parent.parent, other_dir)

    saved = lasso_module.list_saved_results(saved_model.dataset.key, tmp_path)

    assert [result.result_path for result in saved] == [saved_model.result_path]


def test_list_saved_results_skips_invalid_files(
    saved_model: ModelingLasso, tmp_path: Path
) -> None:
    target_dir = saved_model.result_path.parent
    metadata = saved_model.result_path.with_suffix(".json").read_text()
    invalid_files = {
        "bad_config": '{"config": {"unknown_field": 1}}',
        "not_json": "{",
        "not_a_dict": "[]",
    }
    for stem, content in invalid_files.items():
        (target_dir / f"{stem}.json").write_text(content)
        (target_dir / f"{stem}.csv").write_text("fold\n0\n")
    (target_dir / "no_csv.json").write_text(metadata)
    unknown_target = target_dir.parent / "not_a_target"
    unknown_target.mkdir()
    (unknown_target / "key.json").write_text(metadata)
    (unknown_target / "key.csv").write_text("fold\n0\n")

    saved = lasso_module.list_saved_results(saved_model.dataset.key, tmp_path)

    assert [result.result_path for result in saved] == [saved_model.result_path]

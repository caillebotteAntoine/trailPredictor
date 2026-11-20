"""Tests for :class:`TrailPredictor.modeling.base_model.BaseModeling`."""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import sklearn
from conftest import DEFAULT_HEART_RATE, N_SPLITS, SEED, TEST_SIZE
from sklearn.linear_model import LinearRegression

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.base_model import BaseModeling
from TrailPredictor.modeling.config import ModelConfig

DATASET_NAME = "dataset_40_60_1000"


class LinearModeling(BaseModeling[ModelConfig]):
    """Smallest concrete model."""

    name: ClassVar[str] = "linear"

    def make_estimator(self) -> LinearRegression:
        return LinearRegression()


@pytest.fixture
def model(
    dataset: pd.DataFrame, model_config: ModelConfig, tmp_path: Path
) -> LinearModeling:
    # heart rate inside the default zones: an "Unknown" zone breaks the one-hot
    in_zone = dataset.assign(heart_rate=DEFAULT_HEART_RATE)
    return LinearModeling(Dataset(in_zone), model_config, models_dir=tmp_path)


def test_cannot_instantiate_abstract_base(
    dataset: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="abstract"):
        # pylint: disable-next=abstract-class-instantiated
        BaseModeling(Dataset(dataset), ModelConfig(), models_dir=tmp_path)  # type: ignore[abstract]


def test_result_path(model: LinearModeling, tmp_path: Path) -> None:
    assert model.result_path == (
        tmp_path / "linear" / model.dataset.key / "pace" / f"{model.config.key}.csv"
    )


def test_no_saved_result(model: LinearModeling) -> None:
    assert model.result is None


def test_save_without_result_raises(model: LinearModeling) -> None:
    with pytest.raises(ValueError, match="no result"):
        model.save_result()


def test_save_and_reload(model: LinearModeling, tmp_path: Path) -> None:
    model.result = pd.DataFrame({"rmse": [0.5, 0.4]})

    path = model.save_result()
    reloaded = LinearModeling(model.dataset, model.config, models_dir=tmp_path)

    assert reloaded.result is not None
    pd.testing.assert_frame_equal(reloaded.result, model.result)
    assert json.loads(path.with_suffix(".json").read_text()) == {
        "dataset_name": None,
        "dataset_key": model.dataset.key,
        "config": {"seed": SEED, "n_splits": N_SPLITS, "test_size": TEST_SIZE},
    }


def test_saved_metadata_holds_dataset_name(
    dataset: pd.DataFrame, model_config: ModelConfig, tmp_path: Path
) -> None:
    in_zone = dataset.assign(heart_rate=DEFAULT_HEART_RATE)
    model = LinearModeling(
        Dataset(in_zone, name=DATASET_NAME), model_config, models_dir=tmp_path
    )
    model.result = pd.DataFrame({"rmse": [0.5]})

    metadata = json.loads(model.save_result().with_suffix(".json").read_text())

    assert metadata["dataset_name"] == DATASET_NAME


def test_other_config_does_not_reload(model: LinearModeling, tmp_path: Path) -> None:
    model.result = pd.DataFrame({"rmse": [0.5]})
    model.save_result()

    other = LinearModeling(model.dataset, ModelConfig(seed=0), models_dir=tmp_path)

    assert other.result is None


def test_other_dataset_does_not_reload(
    model: LinearModeling, dataset: pd.DataFrame, tmp_path: Path
) -> None:
    model.result = pd.DataFrame({"rmse": [0.5]})
    model.save_result()
    one_activity = dataset.loc[dataset["activity_id"] == 1].assign(
        heart_rate=DEFAULT_HEART_RATE
    )

    other = LinearModeling(Dataset(one_activity), model.config, models_dir=tmp_path)

    assert other.dataset.key != model.dataset.key
    assert other.result is None


def test_same_data_reloads_after_csv_round_trip(
    model: LinearModeling, tmp_path: Path
) -> None:
    # the second loading of Dataset.load reads the raw data from a CSV cache
    model.result = pd.DataFrame({"rmse": [0.5]})
    model.save_result()
    csv_path = tmp_path / "raw.csv"
    model.dataset.data.to_csv(csv_path, index=False)

    read_back = Dataset(pd.read_csv(csv_path))
    reloaded = LinearModeling(read_back, model.config, models_dir=tmp_path)

    assert reloaded.result is not None


def test_cross_validate_one_score_per_fold(model: LinearModeling) -> None:
    scores = model.cross_validate()

    assert scores["test_score"].shape == (N_SPLITS,)
    assert np.all(scores["test_score"] <= 0.0)  # negative RMSE


def test_default_pipeline_has_no_intermediate_step(model: LinearModeling) -> None:
    assert model.make_pipeline_steps() == []
    assert [name for name, _ in model.build_pipeline().steps] == [
        "preprocess",
        "regression",
    ]


def test_cross_validate_returns_fitted_pipelines(model: LinearModeling) -> None:
    scores = model.cross_validate(return_estimator=True)

    assert len(scores["estimator"]) == N_SPLITS
    assert all(
        hasattr(pipeline.named_steps["regression"], "coef_")
        for pipeline in scores["estimator"]
    )


def test_cross_validate_leaves_metadata_routing_disabled(
    model: LinearModeling,
) -> None:
    model.cross_validate()

    assert not sklearn.get_config()["enable_metadata_routing"]


def test_cross_validate_does_not_modify_dataset(model: LinearModeling) -> None:
    before = model.dataset.data

    model.cross_validate()

    pd.testing.assert_frame_equal(model.dataset.data, before)


def test_fit_fits_pipeline_on_whole_dataset(model: LinearModeling) -> None:
    pipeline = model.fit()

    predicted = pipeline.predict(model.dataset.predictors)
    assert predicted.shape == (len(model.dataset.data),)
    assert np.isfinite(predicted).all()

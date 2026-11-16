"""Tests for :class:`TrailPredictor.modeling.base_model.BaseModeling`."""

import json
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from conftest import DEFAULT_HEART_RATE
from sklearn.linear_model import LinearRegression

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.base_model import BaseModeling
from TrailPredictor.modeling.config import ModelConfig

N_SPLITS = 2  # the test dataset holds two activities
N_ACTIVITIES_SPLIT = 10  # 20 % test size: two test activities


class LinearModeling(BaseModeling[ModelConfig]):
    """Smallest concrete model."""

    name: ClassVar[str] = "linear"

    def make_estimator(self) -> LinearRegression:
        return LinearRegression()


@pytest.fixture
def model(dataset: pd.DataFrame, tmp_path: Path) -> LinearModeling:
    # heart rate inside the default zones: an "Unknown" zone breaks the one-hot
    in_zone = dataset.assign(heart_rate=DEFAULT_HEART_RATE)
    return LinearModeling(
        Dataset(in_zone), ModelConfig(n_splits=N_SPLITS), models_dir=tmp_path
    )


def test_cannot_instantiate_abstract_base(
    dataset: pd.DataFrame, tmp_path: Path
) -> None:
    with pytest.raises(TypeError, match="abstract"):
        # pylint: disable-next=abstract-class-instantiated
        BaseModeling(Dataset(dataset), ModelConfig(), models_dir=tmp_path)  # type: ignore[abstract]


def test_result_path(model: LinearModeling, tmp_path: Path) -> None:
    assert model.result_path == tmp_path / "linear" / "pace" / f"{model.config.key}.csv"


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
        "seed": 42,
        "n_splits": N_SPLITS,
    }


def test_other_config_does_not_reload(model: LinearModeling, tmp_path: Path) -> None:
    model.result = pd.DataFrame({"rmse": [0.5]})
    model.save_result()

    other = LinearModeling(model.dataset, ModelConfig(seed=0), models_dir=tmp_path)

    assert other.result is None


def test_cross_validate_one_score_per_fold(model: LinearModeling) -> None:
    scores = model.cross_validate()

    assert scores["test_score"].shape == (N_SPLITS,)
    assert np.all(scores["test_score"] <= 0.0)  # negative RMSE


def test_cross_validate_does_not_modify_dataset(model: LinearModeling) -> None:
    before = model.dataset.data

    model.cross_validate()

    pd.testing.assert_frame_equal(model.dataset.data, before)


def test_split_test_train_keeps_activities_whole(
    make_segments, make_dataset, tmp_path: Path
) -> None:
    # enough activities for both sets to hold several of them: build_features
    # drops `activity_id` from a single-activity dataset
    activity_ids = range(N_ACTIVITIES_SPLIT)
    data = make_dataset(
        {i: make_segments([100.0, 110.0 + i, 105.0, 120.0]) for i in activity_ids}
    )
    model = LinearModeling(Dataset(data, target="speed"), ModelConfig(), tmp_path)

    train, test = model.split_test_train()

    train_ids = set(train.data["activity_id"])
    test_ids = set(test.data["activity_id"])
    assert not train_ids & test_ids
    assert train_ids | test_ids == set(activity_ids)
    assert train.target_name == test.target_name == "speed"

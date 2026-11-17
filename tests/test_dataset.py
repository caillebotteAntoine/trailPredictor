"""Tests for :class:`TrailPredictor.dataset.Dataset` feature selection and zones."""

from pathlib import Path

import pandas as pd
import pytest
from conftest import CUSTOM_DROP_THRESHOLD, N_ACTIVITIES_SPLIT, TWO_ZONES

from TrailPredictor.dataset import Dataset, dataset_name

DATASET_NAME = "trail"
SPLIT_SEED = 0


@pytest.fixture
def ds(dataset: pd.DataFrame) -> Dataset:
    return Dataset(dataset)


def test_with_heart_rate_zones_returns_rezoned_copy(ds: Dataset) -> None:
    zoned = ds.with_heart_rate_zones(TWO_ZONES)

    assert zoned is not ds
    assert zoned.data["heart_rate_zone"].tolist() == [
        "low", "low", "high", "high", "high", "high",  # activity 1
        "low", "low", "high", "low",  # activity 2
    ]  # fmt: skip


def test_with_heart_rate_zones_leaves_dataset_unchanged(ds: Dataset) -> None:
    before = ds.data

    ds.with_heart_rate_zones(TWO_ZONES)

    pd.testing.assert_frame_equal(ds.data, before)


def test_with_heart_rate_zones_keeps_target_and_features(ds: Dataset) -> None:
    ds.target = "speed"
    ds.features = ["slope", "heart_rate_zone"]

    zoned = ds.with_heart_rate_zones(TWO_ZONES)

    assert zoned.target_name == "speed"
    assert zoned.features == ["slope", "heart_rate_zone"]
    assert zoned.get_preprocess().transformers[1][1].categories == [
        ["low", "high"],
        ["Flat", "Uphill", "Downhill"],
    ]


def test_with_heart_rate_zones_only_changes_zone_column(ds: Dataset) -> None:
    zoned = ds.with_heart_rate_zones(TWO_ZONES)

    pd.testing.assert_frame_equal(
        zoned.data.drop(columns="heart_rate_zone"),
        ds.data.drop(columns="heart_rate_zone"),
    )


def test_with_heart_rate_zones_matches_full_rebuild(ds: Dataset) -> None:
    rebuilt = ds.with_heart_rate_zones(TWO_ZONES)  # copy before the in-place rebuild

    ds.build_features(heart_rate_zones=TWO_ZONES)

    pd.testing.assert_frame_equal(rebuilt.data, ds.data)


def test_with_heart_rate_zones_keeps_custom_drop_threshold(ds: Dataset) -> None:
    default_categories = ds.data["drop_category"]
    ds.build_features(drop_threshold=CUSTOM_DROP_THRESHOLD)

    zoned = ds.with_heart_rate_zones(TWO_ZONES)

    # not reverted to the default threshold
    assert not zoned.data["drop_category"].equals(default_categories)
    pd.testing.assert_series_equal(
        zoned.data["drop_category"], ds.data["drop_category"]
    )


def test_predictors_are_the_features(ds: Dataset) -> None:
    ds.features = ["slope", "running_cost"]

    assert ds.predictors.columns.tolist() == ["slope", "running_cost"]


@pytest.mark.parametrize(
    ("feature", "message"),
    [("missing", "not found"), ("pace", "cannot be used")],
)
def test_invalid_feature_raises(ds: Dataset, feature: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ds.features = ["slope", feature]


def test_split_test_train_keeps_activities_whole(
    model_config, make_segments, make_dataset
) -> None:
    # enough activities for both sets to hold several of them: build_features
    # drops `activity_id` from a single-activity dataset
    activity_ids = range(N_ACTIVITIES_SPLIT)
    data = Dataset(
        make_dataset(
            {i: make_segments([100.0, 110.0 + i, 105.0, 120.0]) for i in activity_ids}
        )
    )
    data.target = "speed"

    train, test = data.split_test_train(
        test_size=model_config.test_size, random_state=model_config.seed
    )

    train_ids = set(train.data["activity_id"])
    test_ids = set(test.data["activity_id"])
    assert not train_ids & test_ids
    assert train_ids | test_ids == set(activity_ids)
    assert train.target_name == test.target_name == "speed"


def test_key_survives_csv_round_trip(dataset: pd.DataFrame, tmp_path: Path) -> None:
    # Dataset.load builds the raw data in memory the first time, then reads
    # it back from the CSV cache: both must share the same key
    path = tmp_path / "raw.csv"
    dataset.to_csv(path, index=False)

    read_back = Dataset(pd.read_csv(path))

    assert Dataset(dataset).key == read_back.key


def test_key_ignores_target_features_and_prediction(ds: Dataset) -> None:
    key = ds.key

    ds.target = "speed"
    ds.features = ["slope"]
    ds.prediction = ds.target

    assert ds.key == key


def test_key_changes_with_data(dataset: pd.DataFrame) -> None:
    fewer_activities = dataset.loc[dataset["activity_id"] == 1]

    assert Dataset(dataset).key != Dataset(fewer_activities).key


def test_single_activity_has_key(dataset: pd.DataFrame) -> None:
    # build_features drops activity_id when there is a single activity
    single = Dataset(dataset.loc[dataset["activity_id"] == 1])

    assert "activity_id" not in single.data.columns
    assert single.key


def test_key_changes_with_heart_rate_zones(ds: Dataset) -> None:
    assert ds.with_heart_rate_zones(TWO_ZONES).key != ds.key


def test_dataset_name_from_load_parameters() -> None:
    assert dataset_name(40, 60, 1000) == "dataset_40_60_1000"


def test_name_defaults_to_none(ds: Dataset) -> None:
    assert ds.name is None


def test_split_test_train_suffixes_name(dataset: pd.DataFrame) -> None:
    train, test = Dataset(dataset, name=DATASET_NAME).split_test_train(0.5, SPLIT_SEED)

    assert (train.name, test.name) == (f"{DATASET_NAME}_train", f"{DATASET_NAME}_test")


def test_split_test_train_keeps_missing_name(ds: Dataset) -> None:
    train, test = ds.split_test_train(0.5, SPLIT_SEED)

    assert (train.name, test.name) == (None, None)


def test_with_heart_rate_zones_keeps_name(dataset: pd.DataFrame) -> None:
    named = Dataset(dataset, name=DATASET_NAME)

    assert named.with_heart_rate_zones(TWO_ZONES).name == DATASET_NAME

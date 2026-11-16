"""Tests for :class:`TrailPredictor.dataset.Dataset` feature selection and zones."""

import pandas as pd
import pytest

from TrailPredictor.dataset import Dataset

TWO_ZONES = {"low": (0, 160), "high": (160, 250)}
CUSTOM_DROP_THRESHOLD = 8.0  # m: the -5 m drops become "Flat"


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

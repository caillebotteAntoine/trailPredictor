"""Tests for :mod:`TrailPredictor.api.dataset_service`."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.api.dataset_service import (
    CHART_HEIGHT_PX,
    DatasetNotLoadedError,
    DatasetParams,
    DatasetSelection,
    DatasetService,
    FeatureFilters,
    dataset_chart,
    features_view,
    preset_features,
    saved_zones,
)
from TrailPredictor.dataset import Dataset
from TrailPredictor.feature_selection import FeatureGroup, Preset, correlation_table
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.modeling.heart_rate import HeartRateConfig, SavedHeartRateResult

PARAMS = DatasetParams(40, 60, 1000)
OTHER_PARAMS = DatasetParams(10, 30, 500)
CUTS = [150, 170]


@pytest.fixture
def loads() -> list[DatasetParams]:
    """Parameters passed to the fake loader of `service`."""
    return []


@pytest.fixture
def service(dataset: pd.DataFrame, loads: list[DatasetParams]) -> DatasetService:
    def load(
        min_distance: int, max_distance: int, width: int, progress: ProgressCallback
    ) -> Dataset:
        loads.append(DatasetParams(min_distance, max_distance, width))
        progress(0.5, "Reading FIT files")
        return Dataset(dataset)

    return DatasetService(load, max_loaded=1, saved_results=list)


def _no_progress(_fraction: float, _message: str) -> None:
    pass


@pytest.fixture
def loaded(service: DatasetService) -> DatasetService:
    service.load(PARAMS, _no_progress)
    return service


@pytest.fixture
def trials_csv(tmp_path: Path) -> Path:
    """Trials whose one-standard-error rule keeps N=2, cuts [140]."""
    path = tmp_path / "trials.csv"
    pd.DataFrame(
        {
            "n_zones": [2, 2, 3],
            "trial_number": [0, 1, 0],
            "rmse": [0.40, 0.30, 0.28],
            "rmse_se": [0.01, 0.02, 0.03],
            "cuts": ["[150]", "[140]", "[140, 170]"],
            "state": ["COMPLETE"] * 3,
        }
    ).to_csv(path, index=False)
    return path


# --- DatasetService: loading and cache


def test_load_caches_the_dataset(
    service: DatasetService, loads: list[DatasetParams]
) -> None:
    reports: list[tuple[float, str]] = []

    service.load(PARAMS, lambda fraction, text: reports.append((fraction, text)))
    service.load(PARAMS, _no_progress)

    assert loads == [PARAMS]
    assert reports == [(0.5, "Reading FIT files")]


def test_load_evicts_the_least_recently_used(
    loaded: DatasetService, loads: list[DatasetParams]
) -> None:
    loaded.load(OTHER_PARAMS, _no_progress)

    loaded.dataset(OTHER_PARAMS, "pace")
    with pytest.raises(DatasetNotLoadedError):
        loaded.dataset(PARAMS, "pace")
    assert loads == [PARAMS, OTHER_PARAMS]


def test_dataset_not_loaded_raises(service: DatasetService) -> None:
    with pytest.raises(DatasetNotLoadedError):
        service.dataset(PARAMS, "pace")


def test_dataset_with_target_and_zones(loaded: DatasetService) -> None:
    dataset = loaded.dataset(PARAMS, "speed", CUTS)

    assert dataset.target_name == "speed"
    assert set(dataset.data["heart_rate_zone"]) <= {"Z0", "Z1", "Z2", "Unknown"}


def test_dataset_does_not_modify_the_cache(loaded: DatasetService) -> None:
    before = loaded.dataset(PARAMS, "pace").data

    loaded.dataset(PARAMS, "speed", CUTS)

    pd.testing.assert_frame_equal(loaded.dataset(PARAMS, "pace").data, before)


@pytest.mark.parametrize("cuts", [[170, 150], [0, 150], [150, 300]])
def test_dataset_invalid_cuts_raise(loaded: DatasetService, cuts: list[int]) -> None:
    with pytest.raises(ValueError):
        loaded.dataset(PARAMS, "pace", cuts)


def test_correlations_are_cached_per_target(loaded: DatasetService) -> None:
    pace = loaded.correlations(PARAMS, "pace")

    assert loaded.correlations(PARAMS, "pace") is pace
    assert loaded.correlations(PARAMS, "speed") is not pace


def test_correlations_not_loaded_raise(service: DatasetService) -> None:
    with pytest.raises(DatasetNotLoadedError):
        service.correlations(PARAMS, "pace")


# --- DatasetService: current selection


def _selection(cuts: list[int] | None = None) -> DatasetSelection:
    return DatasetSelection(
        min_distance=PARAMS.min_distance,
        max_distance=PARAMS.max_distance,
        segment_width=PARAMS.segment_width,
        target="speed",
        cuts=CUTS if cuts is None else cuts,
    )


def test_select_sets_the_current_dataset(loaded: DatasetService) -> None:
    assert loaded.current is None

    loaded.select(_selection())

    assert loaded.current == _selection()
    assert loaded.current.params == PARAMS


def test_select_a_dataset_not_loaded_raises(service: DatasetService) -> None:
    with pytest.raises(DatasetNotLoadedError):
        service.select(_selection())
    assert service.current is None


def test_select_invalid_cuts_raise(loaded: DatasetService) -> None:
    loaded.select(_selection())

    with pytest.raises(ValueError):
        loaded.select(_selection(cuts=[170, 150]))
    assert loaded.current == _selection()


# --- features_view and preset_features


def test_features_view(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset)
    correlations = correlation_table(data)

    view = features_view(data, correlations, FeatureFilters())

    assert view.n_rows == len(dataset)
    assert view.ranked == correlations["name"].tolist()
    assert [row.name for row in view.rows] == view.ranked
    assert view.groups == sorted(set(view.groups))
    assert set(view.non_numeric).isdisjoint(view.ranked)
    assert set(view.non_numeric) | set(view.ranked) == set(data.features)
    json.dumps(view.model_dump(mode="json"), allow_nan=False)  # NaN -> null


def test_features_view_applies_the_filters(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset)
    correlations = correlation_table(data)
    filters = FeatureFilters(groups=[FeatureGroup.PAST], min_abs=0.1)

    view = features_view(data, correlations, filters)

    assert view.rows
    assert all(row.group is FeatureGroup.PAST for row in view.rows)
    assert all(abs(row.spearman) >= 0.1 for row in view.rows if row.spearman)
    assert len(view.ranked) == len(correlations)  # unfiltered


def test_features_view_nan_correlation_is_none(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset)
    correlations = correlation_table(data).assign(spearman=np.nan)

    view = features_view(data, correlations, FeatureFilters())

    assert all(row.spearman is None for row in view.rows)


def test_preset_features_picks_among_the_filtered(dataset: pd.DataFrame) -> None:
    correlations = correlation_table(Dataset(dataset))
    filters = FeatureFilters(search="back")

    names = preset_features(correlations, Preset.ALL, filters)

    assert names
    assert all("back" in name for name in names)


# --- dataset_chart


def test_target_histogram_chart(dataset: pd.DataFrame) -> None:
    figure = dataset_chart(Dataset(dataset)).model_dump()["figure"]

    assert figure is not None
    assert figure["layout"]["height"] == CHART_HEIGHT_PX
    assert len(figure["data"]) >= 8  # one histogram per target


def test_feature_chart(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset)
    feature = correlation_table(data)["name"].iloc[0]

    figure = dataset_chart(data, feature).model_dump()["figure"]

    assert figure is not None
    assert feature in figure["layout"]["title"]["text"]


def test_feature_chart_does_not_change_the_target(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset).with_target("speed")

    dataset_chart(data, correlation_table(data)["name"].iloc[0])

    assert data.target_name == "speed"


@pytest.mark.parametrize("feature", ["missing", "heart_rate_zone"])
def test_invalid_feature_chart_reports_the_error(
    dataset: pd.DataFrame, feature: str
) -> None:
    view = dataset_chart(Dataset(dataset), feature).model_dump()

    assert view["figure"] is None
    assert feature in view["error"]


# --- saved_zones


def test_saved_zones(trials_csv: Path, tmp_path: Path) -> None:
    config = HeartRateConfig(zone_counts=(2, 3))
    saved = [
        SavedHeartRateResult("pace", config, trials_csv, "abc", "dataset_40_60"),
        SavedHeartRateResult("pace", config, tmp_path / "missing.csv", "abc"),
    ]

    zones = saved_zones(saved)

    assert len(zones) == 1
    assert zones[0].key == str(trials_csv)
    assert zones[0].cuts == [140]
    assert zones[0].dataset_label == "dataset_40_60 (abc)"
    assert zones[0].config_key == config.key


def test_service_saved_zones_reads_the_saved_results(trials_csv: Path) -> None:
    config = HeartRateConfig()
    service = DatasetService(
        saved_results=lambda: [SavedHeartRateResult("speed", config, trials_csv, "k")]
    )

    assert [zones.target_name for zones in service.saved_zones()] == ["speed"]


def test_saved_zones_empty() -> None:
    assert not saved_zones([])

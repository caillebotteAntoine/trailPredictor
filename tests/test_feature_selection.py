"""Tests for :mod:`TrailPredictor.feature_selection`."""

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.dataset import Dataset
from TrailPredictor.feature_selection import (
    CORRELATION_COLUMNS,
    FeatureGroup,
    Preset,
    apply_filters,
    apply_preset,
    correlation_table,
    feature_group,
    merge_edits,
)


@pytest.fixture
def correlations() -> pd.DataFrame:
    """Table sorted by decreasing |rho|, with a NaN correlation last."""
    return pd.DataFrame(
        {
            "name": ["climb_back_1km", "Climb_front_1km", "drop_done", "x_ratio", "y"],
            "spearman": [0.9, -0.6, 0.3, -0.1, np.nan],
            "missing_pct": [0.0, 50.0, 10.0, 0.0, 100.0],
            "group": ["past", "upcoming", "cumulative", "ratio", "other"],
        }
    )


# --- feature_group


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("altitude_drop_back_1km", FeatureGroup.PAST),
        ("altitude_drop_front_2km", FeatureGroup.UPCOMING),
        ("distance_done", FeatureGroup.CUMULATIVE),
        ("distance_remaining", FeatureGroup.CUMULATIVE),
        ("altitude_drop_pos_total", FeatureGroup.CUMULATIVE),
        ("altitude_drop_ratio", FeatureGroup.RATIO),
        ("heart_rate", FeatureGroup.OTHER),
        ("", FeatureGroup.OTHER),
    ],
)
def test_feature_group(name: str, expected: FeatureGroup) -> None:
    assert feature_group(name) is expected


# --- correlation_table


def test_correlation_table_layout(dataset: pd.DataFrame) -> None:
    table = correlation_table(Dataset(dataset).with_target("speed"))

    assert table.columns.tolist() == CORRELATION_COLUMNS
    assert not table.empty
    assert table["name"].is_unique
    assert table["missing_pct"].between(0, 100).all()
    assert set(table["group"]) <= {group.value for group in FeatureGroup}
    abs_rho = table["spearman"].abs()
    ranked = abs_rho.dropna()
    assert ranked.is_monotonic_decreasing
    assert abs_rho.isna().sum() == len(abs_rho) - len(ranked)
    assert abs_rho.iloc[len(ranked) :].isna().all()  # NaN last


def test_correlation_table_pace_is_expressed_as_speed(dataset: pd.DataFrame) -> None:
    # pace is a decreasing function of speed: the ranks are reversed, and the
    # flipped sign makes both tables equal
    by_speed = correlation_table(Dataset(dataset).with_target("speed"))
    by_pace = correlation_table(Dataset(dataset).with_target("pace"))

    pd.testing.assert_frame_equal(
        by_pace.set_index("name").sort_index(),
        by_speed.set_index("name").sort_index(),
    )


def test_correlation_table_does_not_modify_dataset(dataset: pd.DataFrame) -> None:
    data = Dataset(dataset).with_target("speed")
    before = data.data.copy()

    correlation_table(data)

    pd.testing.assert_frame_equal(data.data, before)


# --- apply_filters


def test_apply_filters_defaults_keep_everything(correlations: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(apply_filters(correlations), correlations)


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        pytest.param(
            {"search": "CLIMB"},
            ["climb_back_1km", "Climb_front_1km"],
            id="search_ignores_case",
        ),
        pytest.param({"search": "_["}, [], id="search_is_not_a_regex"),
        pytest.param(
            {"groups": ["past", "ratio"]},
            ["climb_back_1km", "x_ratio"],
            id="groups",
        ),
        pytest.param(
            {"min_abs": 0.3},
            ["climb_back_1km", "Climb_front_1km", "drop_done"],
            id="min_abs_drops_nan",
        ),
        pytest.param(
            {"max_missing": 10.0},
            ["climb_back_1km", "drop_done", "x_ratio"],
            id="max_missing",
        ),
        pytest.param(
            {"search": "climb", "min_abs": 0.7}, ["climb_back_1km"], id="combined"
        ),
    ],
)
def test_apply_filters(
    correlations: pd.DataFrame, kwargs: dict[str, object], expected: list[str]
) -> None:
    filtered = apply_filters(correlations, **kwargs)  # type: ignore[arg-type]

    assert filtered["name"].tolist() == expected
    assert filtered.index.tolist() == list(range(len(expected)))


def test_apply_filters_does_not_modify_input(correlations: pd.DataFrame) -> None:
    before = correlations.copy()

    apply_filters(correlations, search="climb", min_abs=0.5)

    pd.testing.assert_frame_equal(correlations, before)


# --- apply_preset


@pytest.mark.parametrize(
    ("preset", "expected"),
    [
        (Preset.TOP_5, ["climb_back_1km", "Climb_front_1km", "drop_done", "x_ratio"]),
        (Preset.TOP_10, ["climb_back_1km", "Climb_front_1km", "drop_done", "x_ratio"]),
        (Preset.POSITIVE, ["climb_back_1km", "drop_done"]),
        (Preset.NEGATIVE, ["Climb_front_1km", "x_ratio"]),
        (
            Preset.ALL,
            ["climb_back_1km", "Climb_front_1km", "drop_done", "x_ratio", "y"],
        ),
    ],
)
def test_apply_preset(
    correlations: pd.DataFrame, preset: Preset, expected: list[str]
) -> None:
    assert apply_preset(preset, correlations) == expected


def test_apply_preset_top_keeps_the_first_rows() -> None:
    pool = pd.DataFrame({"name": [f"f{i}" for i in range(30)], "spearman": 0.5})

    assert apply_preset(Preset.TOP_20, pool) == [f"f{i}" for i in range(20)]


def test_every_preset_handles_an_empty_pool(correlations: pd.DataFrame) -> None:
    empty = correlations.iloc[0:0]

    assert all(apply_preset(preset, empty) == [] for preset in Preset)


# --- merge_edits


@pytest.mark.parametrize(
    ("current", "flags", "expected"),
    [
        pytest.param(["a", "b"], {"a": False}, ["b"], id="untick"),
        pytest.param(["a"], {"b": True, "c": True}, ["a", "b", "c"], id="append"),
        pytest.param(["b", "a"], {"a": True, "b": True}, ["b", "a"], id="keep_order"),
        pytest.param(["a", "hidden"], {"a": True}, ["a", "hidden"], id="hidden_kept"),
        pytest.param([], {}, [], id="empty"),
    ],
)
def test_merge_edits(
    current: list[str], flags: dict[str, bool], expected: list[str]
) -> None:
    assert merge_edits(current, flags) == expected

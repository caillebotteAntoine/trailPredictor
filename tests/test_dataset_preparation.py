"""Tests for :func:`TrailPredictor.features.build_features` and its helpers."""

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.features import (
    HEART_RATE_ZONES,
    OUT_OF_ZONES_LABEL,
    assign_heart_rate_zones,
    build_features,
)

# --------------------------------------------------------------------------- #
# Valeurs attendues (calculées à la main), par activité
# --------------------------------------------------------------------------- #
# Activité 1 : drop = [0, 10, -5, 10, 0, -15]
# Activité 2 : drop = [0, 0, 10, -5]

EXPECTED = {
    1: {
        # --- altitude_drop ---
        "altitude_drop": [0, 10, -5, 10, 0, -15],
        "altitude_drop_pos": [0, 10, 0, 10, 0, 0],
        "altitude_drop_neg": [0, 0, -5, 0, 0, -15],
        # --- done / remaining / total ---
        "altitude_drop_pos_done": [0, 10, 10, 20, 20, 20],
        "altitude_drop_pos_remaining": [20, 10, 10, 0, 0, 0],
        "altitude_drop_pos_total": [20] * 6,
        "altitude_drop_neg_done": [0, 0, -5, -5, -5, -20],
        "altitude_drop_neg_remaining": [-20, -20, -15, -15, -15, 0],
        "altitude_drop_neg_total": [-20] * 6,
        "distance_remaining": [5000, 4000, 3000, 2000, 1000, 0],
        "distance_total": [6000] * 6,
        # --- ratios (= done / total, remaining / total) ---
        "altitude_drop_pos_done_ratio": [0, 0.5, 0.5, 1, 1, 1],
        "altitude_drop_pos_remaining_ratio": [1, 0.5, 0.5, 0, 0, 0],
        "altitude_drop_neg_done_ratio": [0, 0, 0.25, 0.25, 0.25, 1],
        "altitude_drop_neg_remaining_ratio": [1, 1, 0.75, 0.75, 0.75, 0],
        # --- fenêtres de 3 km, k = 3 (segment courant exclu) ---
        "altitude_drop_pos_back_3km": [0, 0, 10, 10, 20, 10],
        "altitude_drop_pos_front_3km": [20, 10, 10, 0, 0, 0],
        "altitude_drop_neg_back_3km": [0, 0, 0, -5, -5, -5],
        "altitude_drop_neg_front_3km": [-5, -5, -15, -15, -15, 0],
    },
    2: {
        "altitude_drop": [0, 0, 10, -5],
        "altitude_drop_pos": [0, 0, 10, 0],
        "altitude_drop_neg": [0, 0, 0, -5],
        "altitude_drop_pos_done": [0, 0, 10, 10],
        "altitude_drop_pos_remaining": [10, 10, 0, 0],
        "altitude_drop_pos_total": [10] * 4,
        "altitude_drop_neg_done": [0, 0, 0, -5],
        "altitude_drop_neg_remaining": [-5, -5, -5, 0],
        "altitude_drop_neg_total": [-5] * 4,
        "distance_remaining": [3000, 2000, 1000, 0],
        "distance_total": [4000] * 4,
        "altitude_drop_pos_done_ratio": [0, 0, 1, 1],
        "altitude_drop_pos_remaining_ratio": [1, 1, 0, 0],
        "altitude_drop_neg_done_ratio": [0, 0, 0, 1],
        "altitude_drop_neg_remaining_ratio": [1, 1, 1, 0],
        "altitude_drop_pos_back_3km": [0, 0, 0, 10],
        "altitude_drop_pos_front_3km": [10, 10, 0, 0],
        "altitude_drop_neg_back_3km": [0, 0, 0, 0],
        "altitude_drop_neg_front_3km": [-5, -5, -5, 0],
    },
}

ALL_COLUMNS = sorted(EXPECTED[1])
DROP_COLUMNS = [
    c
    for c in ALL_COLUMNS
    if c.startswith("altitude_drop") and c.count("_") == 2 or c == "altitude_drop"
]
CUMULATIVE_COLUMNS = [
    c for c in ALL_COLUMNS if c.endswith(("_done", "_remaining", "_total"))
]
RATIO_COLUMNS = [c for c in ALL_COLUMNS if c.endswith("_ratio")]
WINDOW_COLUMNS = [c for c in ALL_COLUMNS if "_back_" in c or "_front_" in c]


@pytest.fixture
def features(dataset: pd.DataFrame) -> pd.DataFrame:
    return build_features(dataset, windows_km=(3,))


def _assert_column(features: pd.DataFrame, col: str) -> None:
    """Compare `col` à la valeur attendue, activité par activité."""
    for activity_id, expected in EXPECTED.items():
        actual = features.loc[features["activity_id"] == activity_id, col]
        np.testing.assert_allclose(
            actual.to_numpy(dtype=float),
            np.asarray(expected[col], dtype=float),
            err_msg=f"{col} (activité {activity_id})",
            atol=1e-9,
        )


# --------------------------------------------------------------------------- #
# Structure générale
# --------------------------------------------------------------------------- #
def test_does_not_mutate_input(dataset):
    before = dataset.copy(deep=True)
    build_features(dataset, windows_km=(3,))
    pd.testing.assert_frame_equal(dataset, before)


def test_preserves_rows_and_index(dataset, features):
    assert len(features) == len(dataset)
    pd.testing.assert_index_equal(features.index, dataset.index)


def test_all_expected_columns_present(features):
    missing = set(ALL_COLUMNS) - set(features.columns)
    assert not missing, f"colonnes manquantes : {sorted(missing)}"


def test_no_nan_or_inf(features):
    values = features[ALL_COLUMNS].to_numpy(dtype=float)
    assert np.isfinite(values).all()


# --------------------------------------------------------------------------- #
# Différences d'altitude
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "col", ["altitude_drop", "altitude_drop_pos", "altitude_drop_neg"]
)
def test_altitude_drop(features, col):
    _assert_column(features, col)


def test_altitude_drop_is_zero_at_start_of_each_activity(features):
    """Pas de fuite entre activités : le 1er segment de chacune vaut 0."""
    first_rows = features.groupby("activity_id").head(1)
    assert (first_rows["altitude_drop"] == 0).all()


# --------------------------------------------------------------------------- #
# done / remaining / total
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col", CUMULATIVE_COLUMNS)
def test_cumulative_features(features, col):
    _assert_column(features, col)


@pytest.mark.parametrize("feature", ["altitude_drop_pos", "altitude_drop_neg"])
def test_done_plus_remaining_equals_total(features, feature):
    """Invariant: done (current segment included) + remaining == total."""
    lhs = features[f"{feature}_done"] + features[f"{feature}_remaining"]
    np.testing.assert_allclose(lhs, features[f"{feature}_total"])


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col", RATIO_COLUMNS)
def test_ratio_features(features, col):
    _assert_column(features, col)


@pytest.mark.parametrize("col", RATIO_COLUMNS)
def test_ratios_are_between_0_and_1(features, col):
    assert features[col].between(0, 1).all()


@pytest.mark.parametrize(
    ("altitude", "zero_total_feature"),
    [
        ([100, 110, 120, 130], "altitude_drop_neg"),  # climbs only
        ([130, 120, 110, 100], "altitude_drop_pos"),  # descends only
        ([100, 100, 100, 100], None),  # flat: both totals are zero
    ],
)
def test_ratios_are_zero_when_total_is_zero(
    make_segments, make_dataset, altitude, zero_total_feature
):
    """No 0 / 0 = NaN when an activity never climbs or never descends."""
    out = build_features(make_dataset({1: make_segments(altitude)}), windows_km=(3,))
    assert np.isfinite(out[RATIO_COLUMNS].to_numpy(dtype=float)).all()
    zero_features = (
        ["altitude_drop_pos", "altitude_drop_neg"]
        if zero_total_feature is None
        else [zero_total_feature]
    )
    for feature in zero_features:
        for suffix in ("done", "remaining"):
            np.testing.assert_allclose(out[f"{feature}_{suffix}_ratio"], 0.0)


# --------------------------------------------------------------------------- #
# Fenêtres back / front
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col", WINDOW_COLUMNS)
def test_window_features(features, col):
    _assert_column(features, col)


def test_windows_do_not_leak_across_activities(features):
    """A leak would add the uphill of the neighbouring activity (10 m each way)."""
    first = features.groupby("activity_id").head(1)
    last = features.groupby("activity_id").tail(1)
    assert (first["altitude_drop_pos_back_3km"] == 0).all()
    assert (last["altitude_drop_pos_front_3km"] == 0).all()


def test_window_columns_follow_windows_km_argument(dataset):
    """windows_km=(1, 3) with 1 km segments -> k = 1 and k = 3."""
    out = build_features(dataset, windows_km=(1, 3))
    for w in (1, 3):
        assert f"altitude_drop_pos_back_{w}km" in out.columns
        assert f"altitude_drop_pos_front_{w}km" in out.columns
    # k = 1: only the previous segment
    np.testing.assert_allclose(
        out.loc[out["activity_id"] == 1, "altitude_drop_pos_back_1km"],
        [0, 0, 10, 0, 10, 0],
    )


# --------------------------------------------------------------------------- #
# Pauses
# --------------------------------------------------------------------------- #
def test_pause_on_slow_flat_segments(features):
    expected = {
        1: [False, False, True, False, False, False],
        2: [False, False, False, True],
    }
    for activity_id, pause in expected.items():
        actual = features.loc[features["activity_id"] == activity_id, "pause"]
        assert actual.tolist() == pause


def test_slow_steep_segment_is_not_a_pause(make_segments):
    """Walking slowly up a 20 % slope is effort, not a pause."""
    steep = make_segments([100.0, 300.0], paused=[True, True])
    out = build_features(steep, windows_km=(1,))
    assert out["running_cost"].iloc[1] > 1.5
    assert not out["pause"].iloc[1]


# --------------------------------------------------------------------------- #
# Drop category
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("drop_threshold", "expected_activity_1"),
    [
        (2.0, ["Flat", "Uphill", "Downhill", "Uphill", "Flat", "Downhill"]),
        # the threshold is strict: a 10 m drop is flat with a 10 m threshold
        (10.0, ["Flat", "Flat", "Flat", "Flat", "Flat", "Downhill"]),
    ],
)
def test_drop_category(dataset, drop_threshold, expected_activity_1):
    out = build_features(dataset, windows_km=(3,), drop_threshold=drop_threshold)
    actual = out.loc[out["activity_id"] == 1, "drop_category"]
    assert actual.tolist() == expected_activity_1


# --------------------------------------------------------------------------- #
# Heart-rate zone
# --------------------------------------------------------------------------- #
def test_heart_rate_zone(features):
    expected = {
        1: [
            "Récupération",
            "Aérobic (Endur.)",
            "Aérobic (Puiss.)",
            "Seuil",
            "Endurance anaérobie",
            "Anaérobie (Puiss.)",
        ],
        # 142 bpm is the lower (inclusive) bound of the 2nd zone; 220 is above all
        2: ["Aérobic (Endur.)", "Récupération", OUT_OF_ZONES, "Récupération"],
    }
    for activity_id, zones in expected.items():
        actual = features.loc[features["activity_id"] == activity_id, "heart_rate_zone"]
        assert actual.tolist() == zones


HR_EPSILON = 0.1  # bpm, distance to a bound for the "just below" case
OUT_OF_ZONES = OUT_OF_ZONES_LABEL  # label of a heart rate outside every zone
ZONE_BOUNDS = list(HEART_RATE_ZONES.values())


def _zones_of(make_segments, heart_rate: list[float]) -> list[str]:
    segments = make_segments([100.0] * len(heart_rate), heart_rate=heart_rate)
    return build_features(segments, windows_km=(1,))["heart_rate_zone"].tolist()


def test_heart_rate_zones_are_contiguous():
    """Each zone starts exactly where the previous one ends: no gap, no overlap."""
    upper_bounds = [high for _, high in ZONE_BOUNDS[:-1]]
    lower_bounds = [low for low, _ in ZONE_BOUNDS[1:]]
    assert upper_bounds == lower_bounds


@pytest.mark.parametrize(
    ("zone", "bounds"), HEART_RATE_ZONES.items(), ids=list(HEART_RATE_ZONES)
)
def test_heart_rate_zone_bounds(make_segments, zone, bounds):
    """Lower bound, midpoint and just below the upper bound fall in the zone."""
    low, high = bounds
    heart_rate = [float(low), (low + high) / 2, high - HR_EPSILON]
    assert _zones_of(make_segments, heart_rate) == [zone] * len(heart_rate)


def test_heart_rate_out_of_zones(make_segments):
    """Above the last zone, or missing, the heart rate is out of zones."""
    top = float(max(high for _, high in ZONE_BOUNDS))
    assert _zones_of(make_segments, [top, np.nan]) == [OUT_OF_ZONES] * 2


TWO_ZONES = {"low": (0, 150), "high": (150, 200)}


def test_assign_heart_rate_zones_matches_build_features(features):
    zones = assign_heart_rate_zones(features["heart_rate"], HEART_RATE_ZONES)

    pd.testing.assert_series_equal(zones, features["heart_rate_zone"])


def test_assign_heart_rate_zones_bounds_and_missing():
    heart_rate = pd.Series(
        [0.0, 149.9, 150.0, 199.9, 200.0, np.nan], index=list("abcdef")
    )

    zones = assign_heart_rate_zones(heart_rate, TWO_ZONES)

    expected = ["low", "low", "high", "high", OUT_OF_ZONES, OUT_OF_ZONES]
    assert zones.tolist() == expected
    assert zones.index.equals(heart_rate.index)  # labels aligned on the input


def test_assign_heart_rate_zones_empty():
    zones = assign_heart_rate_zones(pd.Series([], dtype="float64"), TWO_ZONES)

    assert zones.empty


def test_single_activity_drops_activity_id(dataset):
    single = dataset.loc[dataset["activity_id"] == 2].drop(columns="activity_id")
    out = build_features(single, windows_km=(3,))
    assert "activity_id" not in out.columns
    assert out["distance_total"].tolist() == [4000] * 4

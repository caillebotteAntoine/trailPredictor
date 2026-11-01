import numpy as np
import pandas as pd
import pytest

from TrailPredictor.dataset_preparation import build_features

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
        "altitude_drop_pos_remaining": [20, 20, 10, 10, 0, 0],
        "altitude_drop_pos_total": [20] * 6,
        "altitude_drop_neg_done": [0, 0, -5, -5, -5, -20],
        "altitude_drop_neg_remaining": [-20, -20, -20, -15, -15, -15],
        "altitude_drop_neg_total": [-20] * 6,
        "distance_done": [1000, 2000, 3000, 4000, 5000, 6000],
        "distance_remaining": [6000, 5000, 4000, 3000, 2000, 1000],
        "distance_total": [6000] * 6,
        # --- ratios (= done / total, remaining / total) ---
        "altitude_drop_pos_done_ratio": [0, 0.5, 0.5, 1, 1, 1],
        "altitude_drop_pos_remaining_ratio": [1, 1, 0.5, 0.5, 0, 0],
        "altitude_drop_neg_done_ratio": [0, 0, 0.25, 0.25, 0.25, 1],
        "altitude_drop_neg_remaining_ratio": [1, 1, 1, 0.75, 0.75, 0.75],
        "distance_done_ratio": [1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1],
        "distance_remaining_ratio": [1, 5 / 6, 4 / 6, 3 / 6, 2 / 6, 1 / 6],
        # --- fenêtres k = 3 (segment courant exclu) ---
        "altitude_drop_pos_back_3": [0, 0, 10, 10, 20, 10],
        "altitude_drop_pos_front_3": [20, 10, 10, 0, 0, 0],
        "altitude_drop_neg_back_3": [0, 0, 0, -5, -5, -5],
        "altitude_drop_neg_front_3": [-5, -5, -15, -15, -15, 0],
        "distance_back_3": [0, 1000, 2000, 3000, 3000, 3000],
        "distance_front_3": [3000, 3000, 3000, 2000, 1000, 0],
    },
    2: {
        "altitude_drop": [0, 0, 10, -5],
        "altitude_drop_pos": [0, 0, 10, 0],
        "altitude_drop_neg": [0, 0, 0, -5],
        "altitude_drop_pos_done": [0, 0, 10, 10],
        "altitude_drop_pos_remaining": [10, 10, 10, 0],
        "altitude_drop_pos_total": [10] * 4,
        "altitude_drop_neg_done": [0, 0, 0, -5],
        "altitude_drop_neg_remaining": [-5, -5, -5, -5],
        "altitude_drop_neg_total": [-5] * 4,
        "distance_done": [1000, 2000, 3000, 4000],
        "distance_remaining": [4000, 3000, 2000, 1000],
        "distance_total": [4000] * 4,
        "altitude_drop_pos_done_ratio": [0, 0, 1, 1],
        "altitude_drop_pos_remaining_ratio": [1, 1, 1, 0],
        "altitude_drop_neg_done_ratio": [0, 0, 0, 1],
        "altitude_drop_neg_remaining_ratio": [1, 1, 1, 1],
        "distance_done_ratio": [0.25, 0.5, 0.75, 1],
        "distance_remaining_ratio": [1, 0.75, 0.5, 0.25],
        "altitude_drop_pos_back_3": [0, 0, 0, 10],
        "altitude_drop_pos_front_3": [10, 10, 0, 0],
        "altitude_drop_neg_back_3": [0, 0, 0, 0],
        "altitude_drop_neg_front_3": [-5, -5, -5, 0],
        "distance_back_3": [0, 1000, 2000, 3000],
        "distance_front_3": [3000, 2000, 1000, 0],
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


@pytest.mark.parametrize(
    "feature", ["altitude_drop_pos", "altitude_drop_neg", "distance"]
)
def test_done_plus_remaining_minus_current_equals_total(features, feature):
    """Invariant : done + remaining - valeur courante == total."""
    lhs = (
        features[f"{feature}_done"]
        + features[f"{feature}_remaining"]
        - features[feature]
    )
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


# --------------------------------------------------------------------------- #
# Fenêtres back / front
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("col", WINDOW_COLUMNS)
def test_window_features(features, col):
    _assert_column(features, col)


def test_windows_do_not_leak_across_activities(features):
    first = features.groupby("activity_id").head(1)
    last = features.groupby("activity_id").tail(1)
    assert (first["distance_back_3"] == 0).all()
    assert (last["distance_front_3"] == 0).all()


def test_window_columns_follow_windows_km_argument(dataset):
    """windows_km=(1, 3) avec segments de 1 km -> k = 1 et k = 3."""
    out = build_features(dataset, windows_km=(1, 3))
    for k in (1, 3):
        assert f"distance_back_{k}" in out.columns
        assert f"distance_front_{k}" in out.columns
    # k = 1 : seulement le segment voisin
    np.testing.assert_allclose(
        out.loc[out["activity_id"] == 1, "distance_back_1"],
        [0, 1000, 1000, 1000, 1000, 1000],
    )

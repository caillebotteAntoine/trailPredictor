"""Tests for :func:`TrailPredictor.hashing.hash_dataframe`."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.hashing import FLOAT_DECIMALS, hash_dataframe
from TrailPredictor.modeling.config import KEY_LENGTH

SEED = 0
N_ROWS = 50
TINY_NOISE = 10.0 ** -(FLOAT_DECIMALS + 4)  # well below the rounding step


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    return pd.DataFrame(
        {
            "row_id": np.arange(N_ROWS),
            "value": rng.normal(size=N_ROWS),
            "label": rng.choice(["a", "b"], size=N_ROWS),
            "flag": rng.random(N_ROWS) > 0.5,
            "time": pd.Timestamp("2026-01-01")
            + pd.to_timedelta(np.arange(N_ROWS), unit="s"),
        }
    )


def test_key_is_short_hexadecimal(frame: pd.DataFrame) -> None:
    key = hash_dataframe(frame)

    assert len(key) == KEY_LENGTH
    int(key, 16)  # raises if not hexadecimal


def test_key_is_deterministic(frame: pd.DataFrame) -> None:
    assert hash_dataframe(frame) == hash_dataframe(frame.copy())


def test_input_is_not_modified(frame: pd.DataFrame) -> None:
    before = frame.copy()

    hash_dataframe(frame.sample(frac=1.0, random_state=SEED), sort_by=["row_id"])

    pd.testing.assert_frame_equal(frame, before)


def test_column_order_is_ignored(frame: pd.DataFrame) -> None:
    assert hash_dataframe(frame) == hash_dataframe(frame[frame.columns[::-1]])


def test_index_is_ignored(frame: pd.DataFrame) -> None:
    shifted = frame.set_axis(frame.index + 100)

    assert hash_dataframe(frame) == hash_dataframe(shifted)


def test_row_order_is_ignored_with_sort_by(frame: pd.DataFrame) -> None:
    shuffled = frame.sample(frac=1.0, random_state=SEED)

    assert hash_dataframe(frame, sort_by=["row_id"]) == hash_dataframe(
        shuffled, sort_by=["row_id"]
    )


def test_row_order_matters_without_sort_by(frame: pd.DataFrame) -> None:
    shuffled = frame.sample(frac=1.0, random_state=SEED)

    assert hash_dataframe(frame) != hash_dataframe(shuffled)


def test_missing_sort_column_raises(frame: pd.DataFrame) -> None:
    with pytest.raises(KeyError):
        hash_dataframe(frame, sort_by=["missing"])


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(lambda f: f.assign(value=f["value"] + 1e-6), id="value"),
        pytest.param(lambda f: f.rename(columns={"value": "other"}), id="name"),
        pytest.param(lambda f: f.drop(columns="label"), id="drop_column"),
        pytest.param(lambda f: f.iloc[:-1], id="drop_row"),
        pytest.param(lambda f: f.assign(flag=~f["flag"]), id="bool"),
    ],
)
def test_content_change_changes_key(frame: pd.DataFrame, change) -> None:
    assert hash_dataframe(frame) != hash_dataframe(change(frame))


def test_noise_below_rounding_is_ignored(frame: pd.DataFrame) -> None:
    # the values must not sit on a rounding boundary of FLOAT_DECIMALS
    rounded = frame.assign(value=frame["value"].round(FLOAT_DECIMALS - 1))
    noisy = rounded.assign(value=rounded["value"] + TINY_NOISE)

    assert hash_dataframe(rounded) == hash_dataframe(noisy)


def test_csv_round_trip_keeps_key(frame: pd.DataFrame, tmp_path: Path) -> None:
    path = tmp_path / "frame.csv"
    frame.to_csv(path, index=False)

    read_back = pd.read_csv(path, parse_dates=["time"])

    assert hash_dataframe(frame) == hash_dataframe(read_back)


@pytest.mark.parametrize(
    ("left", "right"),
    [
        pytest.param([0.0], [-0.0], id="signed_zero"),
        pytest.param([0.0], [-TINY_NOISE], id="rounds_to_negative_zero"),
        pytest.param([np.nan], [-np.nan], id="nan_bits"),
        pytest.param([1, 2], [1.0, 2.0], id="int_float"),
    ],
)
def test_equivalent_numbers_share_key(left: list[float], right: list[float]) -> None:
    assert hash_dataframe(pd.DataFrame({"x": left})) == hash_dataframe(
        pd.DataFrame({"x": right})
    )


def test_nullable_dtype_matches_numpy_dtype() -> None:
    nullable = pd.DataFrame({"x": pd.array([1.5, None], dtype="Float64")})
    numpy = pd.DataFrame({"x": [1.5, np.nan]})

    assert hash_dataframe(nullable) == hash_dataframe(numpy)


def test_categorical_matches_strings(frame: pd.DataFrame) -> None:
    categorical = frame.assign(label=frame["label"].astype("category"))

    assert hash_dataframe(frame) == hash_dataframe(categorical)


@pytest.mark.parametrize(
    "empty",
    [
        pytest.param(pd.DataFrame(), id="no_column"),
        pytest.param(pd.DataFrame({"x": pd.Series([], dtype=float)}), id="no_row"),
    ],
)
def test_empty_frame_has_key(empty: pd.DataFrame) -> None:
    assert len(hash_dataframe(empty)) == KEY_LENGTH


def test_empty_frames_with_other_columns_differ() -> None:
    assert hash_dataframe(pd.DataFrame(columns=["x"])) != hash_dataframe(
        pd.DataFrame(columns=["y"])
    )

"""Tests for :mod:`TrailPredictor.plot.track_3d`."""

import json

import numpy as np
import pandas as pd
import pytest

from TrailPredictor.plot.track_3d import (
    MISSING_VALUE_COLOR,
    TERRARIUM_DECODER,
    TEXTURE_ATTRIBUTIONS,
    TEXTURE_URLS,
    TRACK_ALTITUDE_OFFSET_M,
    build_track_map,
    color_bounds,
    km_marker_indices,
    values_to_rgb,
)

N_POINTS = 5
SEGMENT_WIDTH = 600.0  # m: 3 km in total, markers at 1 km and 2 km
START_LAT = 45.0
START_LONG = 6.0
LAT_LONG_STEP = 0.001


@pytest.fixture
def track() -> pd.DataFrame:
    """Return a straight five-point track with one pause."""
    steps = np.arange(N_POINTS, dtype=np.float64)
    return pd.DataFrame(
        {
            "position_lat": START_LAT + steps * LAT_LONG_STEP,
            "position_long": START_LONG + steps * LAT_LONG_STEP,
            "altitude": 1000.0 + 10.0 * steps,
            "distance": (steps + 1) * SEGMENT_WIDTH,
            "heart_rate": 140.0 + steps,
            "pace": 5.0 + steps / 10,
            "pause": [False, False, True, False, False],
        }
    )


def _layer(track_map, layer_id: str):
    (layer,) = [layer for layer in track_map.deck.layers if layer.id == layer_id]
    return layer


def _layer_ids(track_map) -> list[str]:
    return [layer.id for layer in track_map.deck.layers]


# --- color_bounds / values_to_rgb


def test_color_bounds_ignores_missing_and_infinite_values() -> None:
    values = [np.nan, np.inf, -np.inf, *np.arange(101.0)]

    assert color_bounds(values) == pytest.approx((5.0, 95.0))


@pytest.mark.parametrize("values", [[], [np.nan, np.inf]])
def test_color_bounds_without_finite_values(values: list[float]) -> None:
    assert np.isnan(color_bounds(values)).all()


def test_values_to_rgb_range_and_missing_values() -> None:
    rgb = values_to_rgb([-10.0, 0.0, 1.0, 10.0, np.nan], "viridis", 0.0, 1.0)

    assert rgb.shape == (5, 3)
    assert rgb.dtype == np.uint8
    # Values outside the bounds are clipped to the colour of the bound
    np.testing.assert_array_equal(rgb[0], rgb[1])
    np.testing.assert_array_equal(rgb[2], rgb[3])
    assert not np.array_equal(rgb[1], rgb[2])
    np.testing.assert_array_equal(rgb[4], MISSING_VALUE_COLOR)


def test_values_to_rgb_constant_scale_does_not_divide_by_zero() -> None:
    rgb = values_to_rgb([3.0, 3.0], "viridis", 3.0, 3.0)

    np.testing.assert_array_equal(rgb[0], rgb[1])


# --- km_marker_indices


@pytest.mark.parametrize(
    ("distance", "expected"),
    [
        ([600.0, 1200.0, 1800.0, 2400.0, 3000.0], [1, 2]),
        ([100.0, 900.0, 1500.0], [1]),
        ([100.0, 500.0, 900.0], []),  # shorter than one kilometre
        ([1500.0], []),
        ([], []),
        ([np.nan, np.nan], []),
    ],
)
def test_km_marker_indices(distance: list[float], expected: list[int]) -> None:
    np.testing.assert_array_equal(km_marker_indices(distance), expected)


# --- build_track_map


def test_build_track_map_layers(track: pd.DataFrame) -> None:
    track_map = build_track_map(track)

    assert _layer_ids(track_map) == [
        "terrain",
        "track",
        "pauses",
        "km-markers",
        "km-labels",
        "start-finish",
    ]
    assert len(_layer(track_map, "track").data) == N_POINTS - 1
    assert [row["text"] for row in _layer(track_map, "km-labels").data] == [
        "km 1",
        "km 2",
    ]


def test_build_track_map_is_json_serializable(track: pd.DataFrame) -> None:
    track = track.assign(heart_rate=np.nan)

    json.loads(build_track_map(track, "heart_rate").deck.to_json())


def test_build_track_map_without_optional_markers(track: pd.DataFrame) -> None:
    track_map = build_track_map(track.assign(pause=False), show_km_markers=False)

    assert _layer_ids(track_map) == ["terrain", "track", "start-finish"]


def test_build_track_map_track_positions(track: pd.DataFrame) -> None:
    track_map = build_track_map(track, vertical_exaggeration=2.0)

    first_path = _layer(track_map, "track").data[0]["path"]
    np.testing.assert_allclose(
        first_path,
        [
            [START_LONG, START_LAT, (1000.0 + TRACK_ALTITUDE_OFFSET_M) * 2],
            [
                START_LONG + LAT_LONG_STEP,
                START_LAT + LAT_LONG_STEP,
                (1010.0 + TRACK_ALTITUDE_OFFSET_M) * 2,
            ],
        ],
    )
    start, finish = (row["position"] for row in _layer(track_map, "start-finish").data)
    assert start[:2] == pytest.approx([START_LONG, START_LAT])
    assert finish[:2] == pytest.approx(
        [START_LONG + 4 * LAT_LONG_STEP, START_LAT + 4 * LAT_LONG_STEP]
    )


def test_build_track_map_terrain_exaggeration(track: pd.DataFrame) -> None:
    terrain = _layer(build_track_map(track, vertical_exaggeration=3.0), "terrain")

    decoder = terrain.elevation_decoder
    assert decoder == pytest.approx(
        {key: 3.0 * value for key, value in TERRARIUM_DECODER.items()}
    )


@pytest.mark.parametrize("texture", ["osm", "satellite"])
def test_build_track_map_texture(track: pd.DataFrame, texture: str) -> None:
    track_map = build_track_map(track, texture=texture)

    assert _layer(track_map, "terrain").texture == TEXTURE_URLS[texture]
    assert track_map.legend.attribution == TEXTURE_ATTRIBUTIONS[texture]


def test_build_track_map_skips_points_without_gps(track: pd.DataFrame) -> None:
    track = track.astype({"position_lat": "float64"})
    track.loc[0, "position_lat"] = np.nan

    track_map = build_track_map(track)

    assert len(_layer(track_map, "track").data) == N_POINTS - 2


def test_build_track_map_interpolates_missing_altitude(track: pd.DataFrame) -> None:
    track.loc[1, "altitude"] = np.nan

    first_path = _layer(build_track_map(track), "track").data[0]["path"]

    assert first_path[1][2] == pytest.approx(1010.0 + TRACK_ALTITUDE_OFFSET_M)


def test_build_track_map_legend(track: pd.DataFrame) -> None:
    legend = build_track_map(track, "pace").legend

    assert legend.label == "Pace (min/km)"
    assert legend.vmin_text.endswith("min/km")
    assert legend.css_gradient.startswith("linear-gradient(to right, rgb(")


def test_build_track_map_legend_without_values(track: pd.DataFrame) -> None:
    legend = build_track_map(track.assign(heart_rate=np.nan), "heart_rate").legend

    assert (legend.vmin_text, legend.vmax_text) == ("—", "—")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"color_by": "speed"},
        {"texture": "terrain"},
        {"vertical_exaggeration": 0.0},
    ],
)
def test_build_track_map_invalid_arguments(track: pd.DataFrame, kwargs: dict) -> None:
    with pytest.raises(ValueError, match="must be"):
        build_track_map(track, **kwargs)


@pytest.mark.parametrize(
    "track_update",
    [
        {"position_lat": np.nan},
        {"position_long": [np.nan] * 4 + [6.0]},
    ],
)
def test_build_track_map_requires_gps(track: pd.DataFrame, track_update: dict) -> None:
    with pytest.raises(ValueError, match="no valid GPS"):
        build_track_map(track.assign(**track_update))


def test_build_track_map_missing_column(track: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="Missing columns"):
        build_track_map(track.drop(columns="pause"))


def test_build_track_map_sizes_in_pixels(track: pd.DataFrame) -> None:
    deck = json.loads(build_track_map(track).deck.to_json())

    units = {
        layer["id"]: layer.get("widthUnits", layer.get("radiusUnits"))
        for layer in deck["layers"]
        if layer["@@type"] in ("PathLayer", "ScatterplotLayer")
    }
    assert units == dict.fromkeys(
        ["track", "pauses", "km-markers", "start-finish"], "pixels"
    )
    (labels,) = [layer for layer in deck["layers"] if layer["id"] == "km-labels"]
    assert labels["getAlignmentBaseline"] == "bottom"

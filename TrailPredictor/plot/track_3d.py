"""Pydeck 3D map of a GPS track drawn over the terrain.

The terrain is a deck.gl ``TerrainLayer`` fed with the free AWS Terrarium
elevation tiles and textured with OpenStreetMap or satellite imagery. The
tiles are fetched by the browser, so building the map needs no network.
"""

import logging
from dataclasses import dataclass
from typing import Literal, get_args

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd
import pydeck as pdk

from TrailPredictor.speed_columns import format_pace

logger = logging.getLogger(__name__)

ColorBy = Literal["altitude", "heart_rate", "pace"]
MapTexture = Literal["osm", "satellite"]

# --- Terrain tiles
TERRARIUM_ELEVATION_URL = (
    "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
)
TERRARIUM_MAX_ZOOM = 15
# Terrarium encoding: elevation (m) = R * 256 + G + B / 256 - 32768
TERRARIUM_DECODER = {
    "rScaler": 256.0,
    "gScaler": 1.0,
    "bScaler": 1.0 / 256.0,
    "offset": -32768.0,
}
TEXTURE_URLS: dict[MapTexture, str] = {
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
    "satellite": (
        "https://server.arcgisonline.com/ArcGIS/rest/services/"
        "World_Imagery/MapServer/tile/{z}/{y}/{x}"
    ),
}
TEXTURE_ATTRIBUTIONS: dict[MapTexture, str] = {
    "osm": "© OpenStreetMap contributors",
    "satellite": "Imagery © Esri, Maxar, Earthstar Geographics",
}

# --- View
DEFAULT_PITCH_DEG = 55.0
DEFAULT_BEARING_DEG = 0.0
MIN_ZOOM, MAX_ZOOM = 3.0, 15.0
REFERENCE_ZOOM = 11.0  # zoom showing a span of REFERENCE_SPAN_DEG
REFERENCE_SPAN_DEG = 0.05
MIN_SPAN_DEG = 1e-5

# --- Track and markers
# The watch altitude and the elevation model rarely agree exactly: lift the
# track so that it is not hidden under the terrain.
TRACK_ALTITUDE_OFFSET_M = 10.0
TRACK_WIDTH_PX = 4
COLOR_QUANTILES = (0.05, 0.95)
MISSING_VALUE_COLOR = (150, 150, 150)
START_COLOR = (0, 158, 115)
FINISH_COLOR = (213, 94, 0)
PAUSE_COLOR = (0, 78, 0)
KM_MARKER_COLOR = (51, 51, 51)
OUTLINE_COLOR = (255, 255, 255)
START_FINISH_RADIUS_PX = 8
MARKER_RADIUS_PX = 5
KM_STEP_M = 1000.0
KM_LABEL_SIZE_PX = 14
KM_LABEL_OFFSET_PX = (0, -14)
LEGEND_GRADIENT_STOPS = 11
RGB_MAX = 255
# pydeck turns plain strings into JavaScript accessor expressions ("@@=pixels"):
# literal string properties must be wrapped in ``pdk.types.String``.
PIXEL_UNITS = pdk.types.String("pixels")

REQUIRED_COLUMNS = ("position_lat", "position_long", "altitude", "distance", "pause")


@dataclass(frozen=True, slots=True)
class ColorOption:
    """Variable that can colour the track.

    Parameters
    ----------
    column : str
        Column of the activity data holding the values.
    label : str
        Name of the variable with its unit.
    colormap : str
        Name of a matplotlib colormap.
    value_format : str
        Format specification of a value, e.g. ``"{:.0f} m"``; ``"pace"``
        formats the value as ``m:ss min/km``.
    """

    column: str
    label: str
    colormap: str
    value_format: str

    def format_value(self, value: float) -> str:
        """Format one value for the tooltip and the legend.

        Parameters
        ----------
        value : float
            Value to format; missing values give ``"—"``.

        Returns
        -------
        str
            Formatted value with its unit.
        """
        if not np.isfinite(value):
            return "—"
        if self.value_format == "pace":
            return format_pace(value)
        return self.value_format.format(value)


COLOR_OPTIONS: dict[ColorBy, ColorOption] = {
    "altitude": ColorOption("altitude", "Altitude (m)", "viridis", "{:.0f} m"),
    "heart_rate": ColorOption(
        "heart_rate", "Heart Rate (bpm)", "viridis", "{:.0f} bpm"
    ),
    "pace": ColorOption("pace", "Pace (min/km)", "RdYlGn_r", "pace"),
}


@dataclass(frozen=True, slots=True)
class TrackLegend:
    """Colour legend of the track, displayed next to the map.

    pydeck has no colour bar, so the caller renders it from this object.

    Parameters
    ----------
    label : str
        Name of the colouring variable with its unit.
    vmin_text, vmax_text : str
        Formatted bounds of the colour scale.
    css_gradient : str
        CSS ``linear-gradient`` reproducing the colour scale.
    attribution : str
        Copyright notice of the map texture.
    """

    label: str
    vmin_text: str
    vmax_text: str
    css_gradient: str
    attribution: str


@dataclass(frozen=True, slots=True)
class TrackMap:
    """3D map of a track and its legend.

    Parameters
    ----------
    deck : pydeck.Deck
        Map to render with ``st.pydeck_chart``.
    legend : TrackLegend
        Colour legend of the track.
    """

    deck: pdk.Deck
    legend: TrackLegend


def color_bounds(values: npt.ArrayLike) -> tuple[float, float]:
    """Return the bounds of the colour scale.

    The bounds are the 5 % and 95 % quantiles of the finite values, so that a
    few outliers do not flatten the scale.

    Parameters
    ----------
    values : array-like of shape (n,)
        Values colouring the track; NaN and infinite values are ignored.

    Returns
    -------
    tuple of float
        Lower and upper bounds; NaN when no value is finite.
    """
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.nan, np.nan
    low, high = np.quantile(finite, COLOR_QUANTILES)
    return float(low), float(high)


def values_to_rgb(
    values: npt.ArrayLike, colormap: str, vmin: float, vmax: float
) -> npt.NDArray[np.uint8]:
    """Map values to RGB colours.

    Parameters
    ----------
    values : array-like of shape (n,)
        Values to colour; values outside ``[vmin, vmax]`` are clipped.
    colormap : str
        Name of a matplotlib colormap.
    vmin, vmax : float
        Bounds of the colour scale.

    Returns
    -------
    numpy.ndarray
        Array of shape (n, 3): colours as integers in ``[0, 255]``; missing or infinite values get
        ``MISSING_VALUE_COLOR``.
    """
    array = np.asarray(values, dtype=np.float64)
    assert array.ndim == 1
    span = vmax - vmin
    normalized = np.zeros_like(array)
    np.divide(array - vmin, span, out=normalized, where=np.isfinite(array) & (span > 0))
    rgba = mpl.colormaps[colormap](np.clip(normalized, 0.0, 1.0))
    rgb = np.round(rgba[:, :3] * RGB_MAX).astype(np.uint8)
    missing = ~np.isfinite(array)
    return np.where(
        missing[:, np.newaxis], np.asarray(MISSING_VALUE_COLOR), rgb
    ).astype(np.uint8)


def km_marker_indices(
    distance_m: npt.ArrayLike, step_m: float = KM_STEP_M
) -> npt.NDArray[np.intp]:
    """Return the points closest to each multiple of `step_m`.

    Parameters
    ----------
    distance_m : array-like of shape (n,)
        Cumulative distance of each point, in metres, sorted ascending.
    step_m : float, optional
        Distance between two markers, in metres. Default is 1000.0.

    Returns
    -------
    numpy.ndarray
        Array of shape (k,): positional index of the point closest to ``step_m``, ``2 * step_m``,
        ... up to the total distance (excluded).
    """
    distance = np.asarray(distance_m, dtype=np.float64)
    assert distance.ndim == 1
    if distance.size < 2 or not np.isfinite(distance).any():
        return np.empty(0, dtype=np.intp)
    targets = np.arange(step_m, np.nanmax(distance), step_m)
    right = np.clip(np.searchsorted(distance, targets), 1, distance.size - 1)
    left = right - 1
    closer_left = np.abs(distance[left] - targets) <= np.abs(distance[right] - targets)
    return np.where(closer_left, left, right).astype(np.intp)


def _map_zoom(latitude: pd.Series, longitude: pd.Series) -> float:
    span = max(
        float(latitude.max() - latitude.min()),
        float(longitude.max() - longitude.min()),
    )
    zoom = REFERENCE_ZOOM - np.log2(max(span, MIN_SPAN_DEG) / REFERENCE_SPAN_DEG)
    return float(np.clip(zoom, MIN_ZOOM, MAX_ZOOM))


def _css_gradient(colormap: str) -> str:
    stops = np.linspace(0.0, 1.0, LEGEND_GRADIENT_STOPS)
    rgb = values_to_rgb(stops, colormap, 0.0, 1.0)
    colors = ", ".join(
        f"rgb({r}, {g}, {b}) {stop:.0%}"
        for (r, g, b), stop in zip(rgb.tolist(), stops, strict=True)
    )
    return f"linear-gradient(to right, {colors})"


def _track_points(data: pd.DataFrame, vertical_exaggeration: float) -> pd.DataFrame:
    """Keep the points with GPS coordinates and compute their 3D position."""
    missing = set(REQUIRED_COLUMNS).difference(data.columns)
    if missing:
        raise ValueError(f"Missing columns for the track map: {sorted(missing)}")
    points = data.loc[
        data["position_lat"].notna() & data["position_long"].notna()
    ].reset_index(drop=True)
    if len(points) < 2:
        raise ValueError("The activity has no valid GPS coordinates.")
    altitude = (
        pd.to_numeric(points["altitude"], errors="coerce")
        .interpolate(limit_direction="both")
        .fillna(0.0)
    )
    elevation = (altitude + TRACK_ALTITUDE_OFFSET_M) * vertical_exaggeration
    positions = np.column_stack(
        (
            points["position_long"].to_numpy(dtype=np.float64),
            points["position_lat"].to_numpy(dtype=np.float64),
            elevation.to_numpy(dtype=np.float64),
        )
    )
    return points.assign(position=positions.tolist())


def _terrain_layer(texture: MapTexture, vertical_exaggeration: float) -> pdk.Layer:
    decoder = {
        key: value * vertical_exaggeration for key, value in TERRARIUM_DECODER.items()
    }
    return pdk.Layer(
        "TerrainLayer",
        id="terrain",
        elevation_data=TERRARIUM_ELEVATION_URL,
        elevation_decoder=decoder,
        texture=TEXTURE_URLS[texture],
        max_zoom=TERRARIUM_MAX_ZOOM,
    )


def _path_layer(
    points: pd.DataFrame, values: npt.NDArray[np.float64], option: ColorOption
) -> pdk.Layer:
    """Draw the track as one two-point path per pair of consecutive points.

    Each path is coloured by the mean value of its two ends.
    """
    positions = np.asarray(points["position"].tolist(), dtype=np.float64)
    segment_values = (values[:-1] + values[1:]) / 2
    vmin, vmax = color_bounds(values)
    distance_m = points["distance"].to_numpy(dtype=np.float64)[1:]
    paths = pd.DataFrame(
        {
            "path": np.stack((positions[:-1], positions[1:]), axis=1).tolist(),
            "color": values_to_rgb(
                segment_values, option.colormap, vmin, vmax
            ).tolist(),
            "tooltip": (
                "Distance : "
                + pd.Series(distance_m).map(lambda distance: f"{distance:.0f} m")
                + f"<br>{option.label} : "
                + pd.Series(segment_values).map(option.format_value)
            ),
        }
    )
    return pdk.Layer(
        "PathLayer",
        paths,
        id="track",
        get_path="path",
        get_color="color",
        get_width=TRACK_WIDTH_PX,
        width_units=PIXEL_UNITS,
        cap_rounded=True,
        joint_rounded=True,
        pickable=True,
    )


def _marker_layer(
    layer_id: str, markers: pd.DataFrame, radius_px: int, **kwargs: object
) -> pdk.Layer:
    return pdk.Layer(
        "ScatterplotLayer",
        markers,
        id=layer_id,
        get_position="position",
        get_fill_color="color",
        get_radius=radius_px,
        radius_units=PIXEL_UNITS,
        stroked=True,
        get_line_color=OUTLINE_COLOR,
        line_width_min_pixels=1,
        **kwargs,
    )


def _start_finish_layer(points: pd.DataFrame) -> pdk.Layer:
    ends = points.iloc[[0, -1]]
    markers = pd.DataFrame(
        {
            "position": ends["position"].tolist(),
            "color": [list(START_COLOR), list(FINISH_COLOR)],
            "tooltip": ["Départ", "Arrivée"],
        }
    )
    return _marker_layer("start-finish", markers, START_FINISH_RADIUS_PX, pickable=True)


def _pause_layer(points: pd.DataFrame) -> pdk.Layer | None:
    paused = points.loc[points["pause"].fillna(False).astype(bool)]
    if paused.empty:
        return None
    markers = pd.DataFrame(
        {
            "position": paused["position"].tolist(),
            "color": [list(PAUSE_COLOR)] * len(paused),
            "tooltip": "Pause",
        }
    )
    return _marker_layer("pauses", markers, MARKER_RADIUS_PX, pickable=True)


def _km_layers(points: pd.DataFrame) -> list[pdk.Layer]:
    indices = km_marker_indices(points["distance"].to_numpy(dtype=np.float64))
    if indices.size == 0:
        return []
    markers = pd.DataFrame(
        {
            "position": points["position"].iloc[indices].tolist(),
            "color": [list(KM_MARKER_COLOR)] * indices.size,
            "text": [f"km {km}" for km in range(1, indices.size + 1)],
        }
    )
    return [
        _marker_layer("km-markers", markers, MARKER_RADIUS_PX),
        pdk.Layer(
            "TextLayer",
            markers,
            id="km-labels",
            get_position="position",
            get_text="text",
            get_color=KM_MARKER_COLOR,
            get_size=KM_LABEL_SIZE_PX,
            get_pixel_offset=KM_LABEL_OFFSET_PX,
            get_alignment_baseline=pdk.types.String("bottom"),
            outline_width=2,
            outline_color=OUTLINE_COLOR,
            font_settings={"sdf": True},
        ),
    ]


def build_track_map(
    data: pd.DataFrame,
    color_by: ColorBy = "altitude",
    *,
    texture: MapTexture = "osm",
    show_km_markers: bool = True,
    vertical_exaggeration: float = 1.0,
) -> TrackMap:
    """Build the 3D map of a GPS track over the terrain.

    The track is coloured by `color_by` and drawn over a terrain built from
    the AWS Terrarium elevation tiles, textured with `texture`. Start, finish,
    pauses and optional kilometre markers are added on top.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per segment with the columns ``position_lat``,
        ``position_long`` (degrees), ``altitude``, ``distance`` (cumulative,
        metres), ``pause`` (bool) and the column of `color_by`. Rows without
        GPS coordinates are skipped.
    color_by : {"altitude", "heart_rate", "pace"}, optional
        Variable used to colour the track. Default is ``"altitude"``.
    texture : {"osm", "satellite"}, optional
        Imagery draped over the terrain. Default is ``"osm"``.
    show_km_markers : bool, optional
        Add a labelled marker at each kilometre. Default is True.
    vertical_exaggeration : float, optional
        Factor applied to the elevation of the terrain and of the track.
        Default is 1.0.

    Returns
    -------
    TrackMap
        The pydeck map and the colour legend of the track.

    Raises
    ------
    ValueError
        If `color_by`, `texture` or `vertical_exaggeration` is invalid, a
        required column is missing, or fewer than two points have GPS
        coordinates.

    Notes
    -----
    The track is lifted by ``TRACK_ALTITUDE_OFFSET_M`` because the altitude
    recorded by the watch and the elevation model may differ by a few metres.
    """
    if color_by not in COLOR_OPTIONS:
        raise ValueError(f"color_by must be one of {list(COLOR_OPTIONS)}")
    if texture not in get_args(MapTexture):
        raise ValueError(f"texture must be one of {list(get_args(MapTexture))}")
    if not np.isfinite(vertical_exaggeration) or vertical_exaggeration <= 0:
        raise ValueError("vertical_exaggeration must be positive")
    option = COLOR_OPTIONS[color_by]
    if option.column not in data.columns:
        raise ValueError(f"Missing column for the track colour: {option.column}")

    points = _track_points(data, vertical_exaggeration)
    values = (
        pd.to_numeric(points[option.column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .to_numpy(dtype=np.float64)
    )
    logger.debug("Building the 3D track map of %d points", len(points))

    layers = [_terrain_layer(texture, vertical_exaggeration)]
    layers.append(_path_layer(points, values, option))
    pause_layer = _pause_layer(points)
    if pause_layer is not None:
        layers.append(pause_layer)
    if show_km_markers:
        layers.extend(_km_layers(points))
    layers.append(_start_finish_layer(points))

    view_state = pdk.ViewState(
        latitude=float(points["position_lat"].mean()),
        longitude=float(points["position_long"].mean()),
        zoom=_map_zoom(points["position_lat"], points["position_long"]),
        pitch=DEFAULT_PITCH_DEG,
        bearing=DEFAULT_BEARING_DEG,
    )
    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        map_provider=None,
        map_style=None,
        tooltip={"html": "{tooltip}"},
    )
    vmin, vmax = color_bounds(values)
    legend = TrackLegend(
        label=option.label,
        vmin_text=option.format_value(vmin),
        vmax_text=option.format_value(vmax),
        css_gradient=_css_gradient(option.colormap),
        attribution=TEXTURE_ATTRIBUTIONS[texture],
    )
    return TrackMap(deck=deck, legend=legend)

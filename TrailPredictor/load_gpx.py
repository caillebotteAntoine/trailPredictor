"""Low-level reading of GPX files."""

import gzip
import logging
from pathlib import Path

import gpxpy
import gpxpy.geo
import gpxpy.gpx
import numpy as np
import numpy.typing as npt
import pandas as pd

from TrailPredictor.load_fit import ProgressCallback

logger = logging.getLogger(__name__)

GZIP_MAGIC = b"\x1f\x8b"
EARTH_RADIUS_M = gpxpy.geo.EARTH_RADIUS
TRACK_POINT_COLUMNS = ["distance", "altitude", "position_lat", "position_long"]


class EmptyGpxError(ValueError):
    """Raised when a GPX file holds no track point with an elevation."""


def load_gpx_file(
    file_path: Path,
    progress_cb: ProgressCallback | None = None,
) -> gpxpy.gpx.GPX:
    """Load a GPX file and optionally report parsing progress.

    Parameters
    ----------
    file_path : pathlib.Path
        Path to a ``.gpx`` or gzip-compressed ``.gpx.gz`` file.
    progress_cb : ProgressCallback, optional
        Called with ``(fraction, message)`` before and after parsing,
        ``fraction`` being in ``[0, 1]``.

    Returns
    -------
    gpxpy.gpx.GPX
        Parsed GPX object.

    Raises
    ------
    gpxpy.gpx.GPXXMLSyntaxException
        If the file is not valid XML.
    """
    if progress_cb:
        progress_cb(0.0, "Loading GPX file...")

    raw = file_path.read_bytes()
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)
    gpx = gpxpy.parse(raw.decode("utf-8"))
    logger.info("Loaded GPX file %s (%d tracks)", file_path, len(gpx.tracks))

    if progress_cb:
        progress_cb(1.0, "GPX file loaded")
    return gpx


def haversine_distance(
    lat_deg: npt.ArrayLike, long_deg: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """Horizontal distance between consecutive points on a spherical Earth.

    Parameters
    ----------
    lat_deg : array_like of shape (n,)
        Latitudes, in degrees.
    long_deg : array_like of shape (n,)
        Longitudes, in degrees.

    Returns
    -------
    distance_m : numpy.ndarray
        Distance from the previous point, of shape (n,), in metres; 0.0 for
        the first point.

    Notes
    -----
    The Earth radius is the one used by ``gpxpy`` (WGS84 equatorial radius).

    Examples
    --------
    >>> haversine_distance([0.0, 0.0], [0.0, 1.0]).round(1).tolist()
    [0.0, 111319.5]
    """
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    long = np.radians(np.asarray(long_deg, dtype=np.float64))
    assert lat.ndim == 1 and lat.shape == long.shape, "expected two 1-D arrays"

    half_chord = (
        np.sin(np.diff(lat) / 2) ** 2
        + np.cos(lat[:-1]) * np.cos(lat[1:]) * np.sin(np.diff(long) / 2) ** 2
    )
    step = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(half_chord, 0.0, 1.0)))
    return np.concatenate((np.zeros(min(lat.size, 1)), step))


def gpx_track_points(gpx: gpxpy.gpx.GPX) -> pd.DataFrame:
    """Extract the track points of a GPX object with their cumulative distance.

    The segments of every track are chained in file order, the gap between
    two segments being counted, and the distance is the horizontal (2D)
    distance from the first point, as for the ``distance`` field of a FIT
    file.

    Parameters
    ----------
    gpx : gpxpy.gpx.GPX
        Parsed GPX object.

    Returns
    -------
    pandas.DataFrame
        One row per track point with the ``float64`` columns ``distance``
        (metres from the start), ``altitude`` (metres), ``position_lat`` and
        ``position_long`` (degrees).

    Raises
    ------
    EmptyGpxError
        If no track point has an elevation.

    See Also
    --------
    haversine_distance : Distance between consecutive points.

    Notes
    -----
    Points without an elevation are dropped after the distance is computed,
    so they still count in the distance.
    """
    track_points = [
        point
        for track in gpx.tracks
        for segment in track.segments
        for point in segment.points
    ]
    points = pd.DataFrame(
        {
            "altitude": [point.elevation for point in track_points],
            "position_lat": [point.latitude for point in track_points],
            "position_long": [point.longitude for point in track_points],
        },
        dtype=np.float64,
    )
    points = (
        points.assign(
            distance=np.cumsum(
                haversine_distance(points["position_lat"], points["position_long"])
            )
        )[TRACK_POINT_COLUMNS]
        .dropna(subset=["altitude"])
        .reset_index(drop=True)
    )

    if points.empty:
        raise EmptyGpxError("The GPX file holds no track point with an elevation.")
    return points

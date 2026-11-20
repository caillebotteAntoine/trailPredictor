"""Single running activity aggregated into fixed-width distance segments."""

from pathlib import Path
from warnings import warn

import numpy as np
import pandas as pd

from TrailPredictor.load_fit import ProgressCallback, load_fit_file
from TrailPredictor.load_gpx import TRACK_POINT_COLUMNS, gpx_track_points, load_gpx_file
from TrailPredictor.speed_columns import add_speed_columns
from TrailPredictor.utils import COMPUTED_DIR, DATA_DIR


def _get_computed_file_path(file_path: str, segment_width: int) -> Path:
    """Get the path to the computed file for a given FIT file and segment width."""
    path = DATA_DIR / file_path
    file_name = path.with_name(path.with_name(path.stem).stem).name
    computed_path = COMPUTED_DIR / path.parent.name / f"{file_name}_{segment_width}.csv"
    return computed_path


class RunningActivities:
    """Running activity split into segments of constant distance.

    Parameters
    ----------
    data : pandas.DataFrame
        One row per segment, with at least a cumulative ``distance`` column
        in metres.
    prediction : pandas.Series, optional
        Predicted target for each segment. Its name must be the target name
        followed by ``"_pred"`` (e.g. ``"pace_pred"``).

    Raises
    ------
    ValueError
        If `prediction` is given without a name.

    See Also
    --------
    TrailPredictor.dataset.Dataset : Several activities stacked together.
    """

    def __init__(self, data: pd.DataFrame, prediction: pd.Series | None = None):
        self._data = data
        self._data["name"] = self._data.get("name", pd.NA)
        self._name = self._data["name"].iloc[0]

        self._segment_width = data["distance"].diff().max()

        self._prediction = prediction
        self._target_name: str | None = None
        if prediction is not None:
            if prediction.name is None:
                raise ValueError("Prediction series must have a name.")
            # remove suffix _pred
            self._target_name = str(prediction.name).removesuffix("_pred")

    @property
    def name(self) -> str:
        """Name of the activity, if any."""
        return self._name if self._name is not None else ""

    @name.setter
    def name(self, value: str):
        """Set the name of the activity.

        Parameters
        ----------
        value : str
            The new name for the activity.
        """
        assert isinstance(value, str), "Activity name must be a string."
        self._data["name"] = value
        self._name = value

    @property
    def target_name(self) -> str:
        """Name of the predicted target.

        Raises
        ------
        ValueError
            If the activity has no prediction.
        """
        if self._target_name is None:
            raise ValueError("No prediction available, target is None.")
        return self._target_name

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying data."""
        return self._data.copy()

    @property
    def segment_width(self) -> int:
        """Return the segment width in meters."""
        return self._segment_width

    @property
    def prediction(self) -> pd.Series:
        """Return a copy of the prediction series.

        Raises
        ------
        ValueError
            If the activity has no prediction.
        """
        if self._prediction is None:
            raise ValueError("No prediction available, prediction is None.")
        return self._prediction.copy()

    @property
    def target(self) -> pd.Series:
        """Return the target value."""
        return self._data[self.target_name].copy()

    @property
    def plot(self):
        """Return a plotting interface of the activity.

        Returns
        -------
        PlotActivities
            Plotting interface built from a copy of the activity.
        """
        # Lazy import: TrailPredictor.plot subclasses RunningActivities
        # (circular import).
        from TrailPredictor.plot import (  # pylint: disable=import-outside-toplevel
            PlotActivities,
        )

        return PlotActivities(self)

    @property
    def distance_km(self) -> pd.Series:
        """Return the distance in kilometers."""
        return self._data["distance"] / 1000

    # ==================================================================================== #
    #                                  CONSTRUCTOR METHODS
    # ==================================================================================== #
    @classmethod
    def from_fit_record(cls, record: pd.DataFrame, segment_width: int = 100):
        """Create an activity from the ``record`` messages of a FIT file.

        Missing values are filled, points are averaged into segments of
        `segment_width` metres and the speed / pace columns are added.

        Parameters
        ----------
        record : pandas.DataFrame
            Raw ``record`` messages, with at least ``timestamp``,
            ``distance``, ``altitude`` or ``enhanced_altitude``,
            ``position_lat`` and ``position_long``. It is not modified.
        segment_width : int, optional
            Segment length in metres. Default is 100.

        Returns
        -------
        RunningActivities
            The segmented activity.

        See Also
        --------
        from_fit_file : Read the records from a FIT file first.
        """
        x = record.copy()

        # fill missing values
        # ffill : propagating the last valid observation to next valid.
        # bfill : using the next valid observation to fill the gap.
        x = x.ffill().bfill()

        if "altitude" not in x.columns and "enhanced_altitude" in x.columns:
            warn("altitude column not found, using enhanced_altitude")
            x["altitude"] = x["enhanced_altitude"]

        if (x["enhanced_altitude"] == x["altitude"]).all():
            x = x.drop(columns=["enhanced_altitude"])
        else:
            warn("enhanced_altitude is not equal to altitude, using enhanced_altitude")
            x = x.drop(columns=["altitude"]).rename(
                columns={"enhanced_altitude": "altitude"}
            )

        # mandatory columns for the activity
        for col in ["heart_rate"]:
            if col not in x.columns:
                x[col] = pd.NA

        # point-wise elevation changes, summed per segment below; the first diff of
        # a segment is measured from the last point of the previous one, so no
        # elevation change is lost at segment boundaries
        altitude_change = x["altitude"].diff().fillna(0.0)
        x["drop_pos"] = altitude_change.clip(lower=0)
        x["drop_neg"] = altitude_change.clip(upper=0)

        # aggregate by segments of segment_width meters
        last_timestamp = x["timestamp"].max()
        x["segment"] = (x["distance"] // segment_width).astype(int)

        def segment_end(distance: pd.Series) -> float:
            """Upper bound of the segment containing the given distances.

            Parameters
            ----------
            distance : pandas.Series
                Cumulative distances of the records of one segment, in metres.

            Returns
            -------
            float
                End of the segment, in metres.
            """
            return (distance.max() // segment_width + 1) * segment_width

        x = (
            x.groupby("segment")
            .agg(
                {
                    # aggregate the features for each segment
                    "timestamp": "first",
                    "distance": segment_end,
                    # mean of the features for each segment
                    "heart_rate": "mean",
                    "altitude": "mean",
                    "position_lat": "mean",
                    "position_long": "mean",
                    # total positive / negative elevation change of the segment
                    "drop_pos": "sum",
                    "drop_neg": "sum",
                }
            )
            .reset_index(drop=False)
        )

        # convertion of semicircles to degrees for latitude and longitude
        x["position_lat"] = x["position_lat"] * 180 / 2**31
        x["position_long"] = x["position_long"] * 180 / 2**31

        # insert elapsed_time and pace features after timestamp column
        x.insert(1, "elapsed_time", x["timestamp"].diff(1).dt.total_seconds().shift(-1))
        x.loc[len(x) - 1, "elapsed_time"] = (
            last_timestamp - x["timestamp"].iloc[-1]
        ).total_seconds()
        x["elapsed_time"] = pd.to_timedelta(x["elapsed_time"], unit="s")

        x["speed"] = (segment_width / x["elapsed_time"].dt.total_seconds()).astype(
            "float64"
        )
        x = add_speed_columns(x)

        return cls(x)

    @classmethod
    def from_fit_file(
        cls,
        file_path: str,
        segment_width: int = 100,
        progress_cb: ProgressCallback | None = None,
    ):
        """Create an activity from a FIT file and cache it as CSV.

        Parameters
        ----------
        file_path : str
            Path of the FIT file, relative to ``DATA_DIR``.
        segment_width : int, optional
            Segment length in metres. Default is 100.
        progress_cb : ProgressCallback, optional
            Called with ``(fraction, message)`` while loading.

        Returns
        -------
        RunningActivities
            The segmented activity.

        See Also
        --------
        from_computed_file : Reuse the cached CSV when it exists.
        """
        path = DATA_DIR / file_path

        content, _ = load_fit_file(
            path,
            progress_cb=(
                (lambda fraction, text: progress_cb(fraction * 0.7, text))
                if progress_cb
                else None
            ),
        )
        if progress_cb:
            progress_cb(0.7, "Construction de l'activité...")
        activity = cls.from_fit_record(
            pd.DataFrame(content["record"]), segment_width=segment_width
        )

        path = _get_computed_file_path(file_path, segment_width=segment_width)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)

        activity._data.to_csv(path, index=False)

        if progress_cb:
            progress_cb(1.0, "Activité chargée")
        return activity

    @classmethod
    def from_track_points(
        cls, points: pd.DataFrame, segment_width: int = 100
    ) -> "RunningActivities":
        """Create a planned activity (a route, without time) from track points.

        The points are grouped into segments of `segment_width` metres as in
        ``from_fit_record``, but only the distance, the altitude, the position
        and the elevation changes are filled: the time, heart-rate, speed and
        pace columns are present but missing, ready to be predicted.

        Parameters
        ----------
        points : pandas.DataFrame
            Track points ordered along the route, with cumulative
            ``distance`` (metres), ``altitude`` (metres), ``position_lat``
            and ``position_long`` (degrees) columns. It is not modified.
        segment_width : int, optional
            Segment length in metres. Default is 100.

        Returns
        -------
        RunningActivities
            The segmented activity, with the same columns as
            ``from_fit_record``.

        Raises
        ------
        ValueError
            If `points` is empty or misses a column.

        See Also
        --------
        from_gpx_file : Read the track points from a GPX file first.
        """
        missing = set(TRACK_POINT_COLUMNS) - set(points.columns)
        if missing:
            raise ValueError(f"Track points miss the columns {sorted(missing)}.")
        if points.empty:
            raise ValueError("Cannot build an activity from no track point.")

        altitude_change = points["altitude"].diff().fillna(0.0)
        segments = (
            points.assign(
                drop_pos=altitude_change.clip(lower=0),
                drop_neg=altitude_change.clip(upper=0),
                segment=(points["distance"] // segment_width).astype(int),
            )
            .groupby("segment")
            .agg(
                altitude=("altitude", "mean"),
                position_lat=("position_lat", "mean"),
                position_long=("position_long", "mean"),
                drop_pos=("drop_pos", "sum"),
                drop_neg=("drop_neg", "sum"),
            )
            .reset_index(drop=False)
        )
        n_segments = len(segments)
        segments = segments.assign(
            # upper bound of each segment, as segment_end in from_fit_record
            distance=(segments["segment"] + 1) * segment_width,
            timestamp=pd.Series(pd.NaT, index=segments.index, dtype="datetime64[ns]"),
            elapsed_time=pd.to_timedelta(np.full(n_segments, np.nan), unit="s"),
            heart_rate=np.full(n_segments, np.nan),
            speed=np.full(n_segments, np.nan),
        )[
            [
                "segment",
                "elapsed_time",
                "timestamp",
                "distance",
                "heart_rate",
                "altitude",
                "position_lat",
                "position_long",
                "drop_pos",
                "drop_neg",
                "speed",
            ]
        ]
        return cls(add_speed_columns(segments))

    @classmethod
    def from_gpx_file(
        cls,
        file_path: str,
        segment_width: int = 100,
        progress_cb: ProgressCallback | None = None,
    ) -> "RunningActivities":
        """Create a planned activity (a route, without time) from a GPX file.

        Parameters
        ----------
        file_path : str
            Path of the GPX file, relative to ``DATA_DIR``.
        segment_width : int, optional
            Segment length in metres. Default is 100.
        progress_cb : ProgressCallback, optional
            Called with ``(fraction, message)`` while loading.

        Returns
        -------
        RunningActivities
            The segmented activity, named after the GPX metadata (or the file
            name), with only the distance, altitude, position and elevation
            changes filled.

        Raises
        ------
        TrailPredictor.load_gpx.EmptyGpxError
            If the file holds no track point with an elevation.

        See Also
        --------
        from_track_points : Segmentation of the track points.
        """
        path = DATA_DIR / file_path
        gpx = load_gpx_file(
            path,
            progress_cb=(
                (lambda fraction, text: progress_cb(fraction * 0.7, text))
                if progress_cb
                else None
            ),
        )
        if progress_cb:
            progress_cb(0.7, "Construction de l'activité...")

        activity = cls.from_track_points(
            gpx_track_points(gpx), segment_width=segment_width
        )
        activity.name = gpx.name or path.stem

        if progress_cb:
            progress_cb(1.0, "Activité chargée")
        return activity

    @classmethod
    def from_computed_file(
        cls,
        file_path: str,
        segment_width: int = 100,
        progress_cb: ProgressCallback | None = None,
    ):
        """Create an activity from its cached CSV, or from the FIT file.

        Parameters
        ----------
        file_path : str
            Path of the original FIT file, relative to ``DATA_DIR``.
        segment_width : int, optional
            Segment length in metres. Default is 100.
        progress_cb : ProgressCallback, optional
            Called with ``(fraction, message)`` while loading.

        Returns
        -------
        RunningActivities
            The segmented activity.

        See Also
        --------
        from_fit_file : Fallback used when no cached CSV exists.
        """
        path = DATA_DIR / file_path

        # try to load from the computed folder first
        path = _get_computed_file_path(file_path, segment_width=segment_width)
        if not path.exists():
            print(
                f"\t💾Computed file not found:\n\t{path},\n\t⌛trying to load from original file..."
            )
            return cls.from_fit_file(
                file_path, segment_width=segment_width, progress_cb=progress_cb
            )

        if progress_cb:
            progress_cb(0.0, "Loading CSV file...")

        chunk_size = 10**6  # 1 MB
        chunks = []
        with pd.read_csv(path, chunksize=chunk_size) as reader:
            for i, chunk in enumerate(reader):
                chunks.append(chunk)
                if progress_cb:
                    progress_cb(
                        (i + 1) * chunk_size / path.stat().st_size * 0.9,
                        "Loading CSV file...",
                    )
        x = pd.concat(chunks, ignore_index=True)

        # convert columns to appropriate types
        x["timestamp"] = pd.to_datetime(x["timestamp"])
        x["elapsed_time"] = pd.to_timedelta(x["elapsed_time"])
        x = add_speed_columns(x)
        activity = cls(x)

        if progress_cb:
            progress_cb(1.0, "Activity loaded")

        return activity

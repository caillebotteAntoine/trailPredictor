"""Single running activity aggregated into fixed-width distance segments."""

from pathlib import Path
from warnings import warn

import pandas as pd

from TrailPredictor.utils import DATA_DIR, COMPUTED_DIR
from TrailPredictor.load_fit import ProgressCallback, load_fit_file
from TrailPredictor.speed_columns import add_speed_columns


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
        self._segment_width = data["distance"].diff().max()

        self._prediction = prediction
        self._target_name: str | None = None
        if prediction is not None:
            if prediction.name is None:
                raise ValueError("Prediction series must have a name.")
            # remove suffix _pred
            self._target_name = str(prediction.name).removesuffix("_pred")

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
        from TrailPredictor.plot import PlotActivities

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
            x.drop(columns=["enhanced_altitude"], inplace=True)
        else:
            warn("enhanced_altitude is not equal to altitude, using enhanced_altitude")
            x = x.drop(columns=["altitude"]).rename(
                columns={"enhanced_altitude": "altitude"}
            )

        # mandatory columns for the activity
        for col in ["heart_rate"]:
            if col not in x.columns:
                x[col] = pd.NA

        # aggregate by segments of segment_width meters
        last_timestamp = x["timestamp"].max()
        x["segment"] = (x["distance"] // segment_width).astype(int)
        x = (
            x.groupby("segment")
            .agg(
                {
                    # aggregate the features for each segment
                    "timestamp": "first",
                    "distance": lambda x: (
                        (x.max() // segment_width + 1) * segment_width
                    ),
                    # mean of the features for each segment
                    "heart_rate": "mean",
                    "altitude": "mean",
                    "position_lat": "mean",
                    "position_long": "mean",
                    # TODO : sum of altitude drop for each segment
                    # "altitude_drop": "sum",
                    # "altitude_drop_pos": "sum",
                    # "altitude_drop_neg": "sum",
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
        if "speed" in x.columns:
            x = add_speed_columns(x)
        else:
            parts = (
                x["pace"]
                .astype("string")
                .str.extract(r"^\s*(\d+):(\d+(?:\.\d+)?)", expand=True)
            )
            pace = parts[0].astype("float64") + parts[1].astype("float64") / 60.0
            x = add_speed_columns(x)
        activity = cls(x)

        if progress_cb:
            progress_cb(1.0, "Activity loaded")

        return activity

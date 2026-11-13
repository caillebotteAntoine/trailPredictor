"""Dataset of segmented running activities used to train the models."""

from typing import cast

import numpy as np
import pandas as pd

from TrailPredictor.utils import COMPUTED_DIR
from TrailPredictor.features import create_dataset, split_test_train
from TrailPredictor.features import build_features
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.speed_columns import add_speed_columns, TARGETS, Target


# Never used as predictors: identifiers, raw time information (elapsed_time and
# timestamp give the pace directly), heart rate (consequence of the effort),
# GPS position (allows to memorise the races) and all the targets.
META_COLUMNS = [
    "activity_id",
    "name",
    "segment",
    "timestamp",
    "heart_rate",
    "elapsed_time",
    "position_lat",
    "position_long",
]
TARGET_COLUMNS = list(TARGETS)
# prediction columns (e.g. "pace_pred") must never leak into the predictors
PREDICTION_COLUMNS = [f"{t}_pred" for t in TARGETS]
EXCLUDED_COLUMNS = set(META_COLUMNS + TARGET_COLUMNS + PREDICTION_COLUMNS)


def infer_segment_width(data: pd.DataFrame) -> float:
    """Infer the segment width from the cumulative distance.

    The width is the median distance step computed inside each activity, so
    it is robust to the jump between two consecutive activities.

    Parameters
    ----------
    data : pandas.DataFrame
        Data with ``activity_id`` and cumulative ``distance`` (metres)
        columns.

    Returns
    -------
    float
        Segment width in metres.

    Raises
    ------
    ValueError
        If no finite, strictly positive width can be inferred.
    """
    width = data.groupby("activity_id")["distance"].diff().median()
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Cannot infer a positive segment width from 'distance'.")
    return float(width)


class Dataset:
    """Segments of several activities with their features and target.

    Parameters
    ----------
    dataset : pandas.DataFrame
        Raw segments of several activities, with at least ``activity_id``,
        ``distance``, ``altitude``, ``timestamp``, ``elapsed_time`` and
        ``speed`` columns. It is not modified.
    target : Target, optional
        Name of the column to predict. Default is ``"pace"``.

    Raises
    ------
    ValueError
        If the segment width cannot be inferred or a speed / pace column is
        missing after the preparation.

    See Also
    --------
    load : Build the dataset from the FIT files, with a CSV cache.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        target: Target = "pace",
    ):
        # No scaling here: scaling is done in get_preprocess() inside a Pipeline,
        # so that it is fitted on the training data only (no leakage, no double
        # scaling when the dataset is split).
        self._data = dataset.copy()
        self._segment_width = infer_segment_width(self._data)
        self._data = self._prepare_raw(self._data, self._segment_width)
        self._data = build_features(self._data)

        missing = {
            "speed",
            "pace",
            "log_speed",
            "log_pace",
            "accel_speed",
            "log_accel_speed",
        } - set(self._data.columns)
        if missing:
            raise ValueError(f"The dataset must contain the columns {sorted(missing)}.")

        self._target_name: Target = target
        self._features = [
            col for col in self._data.columns if col not in EXCLUDED_COLUMNS
        ]

    @property
    def target_name(self) -> Target:
        """Return the target name.

        Raises
        ------
        ValueError
            If the target is not one of ``TARGETS``.
        """
        if self._target_name not in TARGETS:
            raise ValueError("Invalid target")
        return self._target_name

    @property
    def segment_width(self) -> float:
        """Segment width in meters."""
        return self._segment_width

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying data."""
        return self._data.copy()

    def get_preprocess(self):
        """Build the preprocessing step that standardises the features.

        Every feature is scaled except the ``*_ratio`` features, which are
        already in ``[0, 1]``.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            Unfitted transformer with pandas output.

        Notes
        -----
        Put it in a ``Pipeline`` so that the mean and standard deviation are
        computed on the training set only.
        """
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler

        to_scale = [col for col in self._features if not col.endswith("_ratio")]

        return ColumnTransformer(
            transformers=[("scale", StandardScaler(), to_scale)],
            remainder="passthrough",  # do not transform other columns
            verbose_feature_names_out=False,  # do not add transformer name to output column names
        ).set_output(transform="pandas")

    @property
    def target(self) -> pd.Series:
        """Target column; setting it changes the target name.

        Raises
        ------
        ValueError
            If the new target is not one of ``TARGETS``.
        """
        return self._data[self._target_name].copy()

    @target.setter
    def target(self, target: Target):
        if target not in TARGETS:
            raise ValueError("Invalid target")
        self._target_name = target

    @property
    def prediction_label(self) -> str:
        """Return the prediction column names corresponding to the target."""
        return f"{self.target_name}_pred"

    @property
    def prediction(self) -> pd.Series:
        """Prediction column of the target.

        Raises
        ------
        ValueError
            If no prediction is set, or a new prediction does not have one
            value per segment.
        """
        if self.prediction_label not in self._data.columns:
            raise ValueError(
                f"Prediction column '{self.prediction_label}' not found in dataset."
                " Please set it first using the 'prediction' setter."
            )
        return self._data[self.prediction_label]

    @prediction.setter
    def prediction(self, predictions: pd.Series):
        """Set the prediction column of the target.

        Parameters
        ----------
        predictions : pandas.Series
            One value per segment, converted to ``float64`` (invalid values
            become NaN).

        Raises
        ------
        ValueError
            If `predictions` does not have one value per segment.
        """
        if len(predictions) != len(self._data):
            raise ValueError("Length of predictions does not match length of dataset.")

        self._data[self.prediction_label] = pd.Series(
            pd.to_numeric(predictions, errors="coerce"),
            index=self._data.index,
            dtype="float64",
        )

    @property
    def predictors(self) -> pd.DataFrame:
        """Return a copy of the feature columns."""
        return cast(pd.DataFrame, self._data[self._features].copy())

    @property
    def features(self) -> list[str]:
        """Return the candidate feature names."""
        return self._features.copy()

    @features.setter
    def features(self, features: list[str]):
        """Set the candidate feature names.

        Parameters
        ----------
        features : list of str
            Columns of the dataset to use as predictors.

        Raises
        ------
        ValueError
            If a feature is not a column, or is a target or prediction column.
        """
        for feature in features:
            if feature not in self._data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset.")
            if feature in TARGET_COLUMNS + PREDICTION_COLUMNS:
                raise ValueError(f"'{feature}' cannot be used as a predictor.")
        self._features = features.copy()

    @property
    def true_total_time(self) -> pd.Series:
        """Total time of each activity, as the sum of the segment durations."""
        return (
            self._data.groupby("activity_id")["elapsed_time"]
            .sum()
            .rename("true_total_time")
        )

    def smearing_factor(self) -> float:
        """Compute the correction factor of the total time for a log target.

        Returns
        -------
        float
            Duan's smearing factor ``mean(exp(residual))`` for ``log_pace``
            (``exp(-residual)`` for ``log_speed``), 1.0 for the non-log
            targets.

        See Also
        --------
        predict_total_time : Uses this factor through its `smearing` argument.

        Notes
        -----
        ``exp(prediction)`` is a median / geometric mean, which under-estimates
        the *expected* time per segment (stops and aid stations make the pace
        distribution heavy-tailed), so the total time is too small [1]_.

        Compute it on the training set (ideally with out-of-fold predictions,
        since in-sample residuals of a flexible model are too small), then
        pass it to ``predict_total_time(smearing=...)`` of the test dataset.

        References
        ----------
        .. [1] Duan, N., "Smearing Estimate: A Nonparametric Retransformation
           Method", Journal of the American Statistical Association, 1983.
        """
        if self._target_name not in ("log_pace", "log_speed"):
            return 1.0
        residual = (self.target - self.prediction).dropna()
        if self._target_name == "log_pace":  # time ∝ pace = exp(y)
            return float(np.exp(residual).mean())
        return float(np.exp(-residual).mean())  # time ∝ 1 / speed = exp(-y)

    def predict_target_to_speed(self) -> pd.Series:
        """Convert the predicted target to a speed.

        Returns
        -------
        pandas.Series
            Predicted speed in m/s, named ``"predicted_speed"``, clipped to a
            small positive value to avoid dividing by zero.

        Raises
        ------
        ValueError
            If no prediction is set, or its name does not match the target.
        """
        predicted = self.prediction

        if self.prediction_label != predicted.name:
            raise ValueError("Predictions must have the same name as the target")

        eps = 1e-6  # avoid dividing by 0 if a model predicts a null / negative value

        target_name = self._target_name
        if target_name.startswith("log_"):
            predicted = np.exp(predicted)
            target_name = target_name[4:]  # remove the "log_" prefix

        if target_name.startswith("accel_"):
            print("accel", predicted)
            elapsed_time = self._data["elapsed_time"].dt.total_seconds()
            predicted = predicted * elapsed_time
            print("accel to speed", predicted)
            target_name = target_name[6:]  # remove the "accel_" prefix

        if target_name == "pace":
            predicted_speed = 1000.0 / (
                60.0 * np.clip(predicted, a_min=eps, a_max=None)
            )
        else:  # speed
            predicted_speed = np.clip(predicted, a_min=eps, a_max=None)

        return pd.Series(
            predicted_speed, index=self._data.index, name="predicted_speed"
        )

    def predict_total_time(self, smearing: float = 1.0) -> pd.Series:
        """Predict the total time of each activity.

        Parameters
        ----------
        smearing : float, optional
            Multiplicative correction of each segment time. Default is 1.0.
            With a log target, pass ``train_dataset.smearing_factor()``.

        Returns
        -------
        pandas.Series
            Predicted total time (Timedelta), indexed by ``activity_id``.

        See Also
        --------
        smearing_factor : Correction factor for a log target.

        Notes
        -----
        ``pace`` is the natural target for a total time (time = pace x
        distance). With ``speed``, the sum of ``d / speed_pred`` is biased
        since E[1/v] != 1/E[v].
        """
        predicted_speed = self.predict_target_to_speed()
        segment_seconds = self._segment_width / predicted_speed * smearing
        return (
            pd.DataFrame(
                {
                    "activity_id": self._data["activity_id"],
                    "predict_total_time": pd.to_timedelta(segment_seconds, unit="s"),
                }
            )
            .groupby("activity_id")["predict_total_time"]
            .sum()
        )

    @property
    def plot(self):
        """Return a plotting interface of the dataset.

        Returns
        -------
        PlotDataset
            Plotting interface built from a copy of the dataset.
        """
        from TrailPredictor.plot import PlotDataset

        return PlotDataset(self)

    def get_activities(self, activity_id: int):
        """Extract one activity, with its prediction when available.

        Parameters
        ----------
        activity_id : int
            Identifier of the activity in the ``activity_id`` column.

        Returns
        -------
        RunningActivities
            The segments of the activity.

        Raises
        ------
        ValueError
            If no segment has this `activity_id`.
        """
        from TrailPredictor.activities import RunningActivities

        activity_data = cast(
            pd.DataFrame,
            self._data.loc[self._data["activity_id"] == activity_id],
        )
        if activity_data.empty:
            raise ValueError(f"No data found for activity_id {activity_id}")

        prediction = (
            self.prediction.loc[activity_data.index]
            if self.prediction_label in self._data
            else None
        )
        return RunningActivities(activity_data, prediction=prediction)

    def split_test_train(self):
        """Split the dataset into a training and a test set by activity.

        Returns
        -------
        train : Dataset
            Training activities, with the same target.
        test : Dataset
            Test activities, with the same target.

        See Also
        --------
        TrailPredictor.features.split_test_train : Underlying split.
        """
        train_data, test_data = split_test_train(self._data)
        return Dataset(train_data, self.target_name), Dataset(
            test_data, self.target_name
        )

    @staticmethod
    def _prepare_raw(raw: pd.DataFrame, segment_width: int) -> pd.DataFrame:
        """Convert the time columns and add the speed / pace columns.

        The same treatment is applied whether the raw data comes from the
        cache or from the FIT files.
        """
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw["elapsed_time"] = pd.to_timedelta(raw["elapsed_time"])
        raw = add_speed_columns(raw)
        return raw

    @classmethod
    def load(
        cls,
        min_distance=40,
        max_distance=60,
        segment_width=1000,
        progress_cb: ProgressCallback | None = None,
        export_features: bool = True,
    ):
        """Load the dataset of the activities within a distance range.

        Reading the FIT files is cached in ``COMPUTED_DIR``; the features are
        always rebuilt, so a change in ``build_features`` is never ignored.

        Parameters
        ----------
        min_distance : int, optional
            Minimum activity distance in km (exclusive). Default is 40.
        max_distance : int, optional
            Maximum activity distance in km (exclusive). Default is 60.
        segment_width : int, optional
            Segment length in metres. Default is 1000.
        progress_cb : ProgressCallback, optional
            Called with ``(fraction, message)`` while reading the FIT files.
        export_features : bool, optional
            Also write the features to a CSV in ``COMPUTED_DIR``. Default is
            True.

        Returns
        -------
        Dataset
            Dataset with ``"pace"`` as target.

        Raises
        ------
        ValueError
            If `min_distance` is not less than `max_distance`, or a speed /
            pace column is missing.
        """
        if min_distance >= max_distance:
            raise ValueError("min_distance must be less than max_distance")

        COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
        suffix = f"{min_distance}_{max_distance}_{segment_width}"
        raw_path = COMPUTED_DIR / f"raw_{suffix}.csv"

        # Only the slow part (reading the FIT files) is cached.
        # The features are always rebuilt, so a change in build_features is
        # never silently ignored by a stale cache.
        try:
            raw = pd.read_csv(raw_path)
        except FileNotFoundError:
            raw = create_dataset(
                min_distance=min_distance,
                max_distance=max_distance,
                segment_width=segment_width,
                progress_cb=progress_cb,
            )
            raw.to_csv(raw_path, index=False)

        raw = cls._prepare_raw(raw, segment_width)
        dataset = build_features(raw)

        missing = {
            "speed",
            "pace",
            "log_speed",
            "log_pace",
            "accel_speed",
            "log_accel_speed",
        } - set(dataset.columns)
        if missing:
            raise ValueError(f"The dataset must contain the columns {sorted(missing)}.")

        if export_features:  # handy to inspect / use from R, not read back
            dataset.to_csv(COMPUTED_DIR / f"dataset_{suffix}.csv", index=False)
        return cls(dataset)

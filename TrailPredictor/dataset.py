from typing import Literal, TypeAlias, cast

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
    """Segment width in meters, estimated inside each activity (robust to the
    jump between two activities)."""
    width = data.groupby("activity_id")["distance"].diff().median()
    if not np.isfinite(width) or width <= 0:
        raise ValueError("Cannot infer a positive segment width from 'distance'.")
    return float(width)


class Dataset:
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
        """Return the target name."""
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
        """Standardise every feature (except the 0-1 ratios). Put it in a
        Pipeline so that the mean / sd are computed on the training set only."""
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
        """Return the target column."""
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
        """Return the prediction column corresponding to the target."""
        if self.prediction_label not in self._data.columns:
            raise ValueError(
                f"Prediction column '{self.prediction_label}' not found in dataset."
                " Please set it first using the 'prediction' setter."
            )
        return self._data[self.prediction_label]

    @prediction.setter
    def prediction(self, predictions: pd.Series):
        """Set the prediction column corresponding to the target."""
        if len(predictions) != len(self._data):
            raise ValueError("Length of predictions does not match length of dataset.")

        self._data[self.prediction_label] = pd.Series(
            pd.to_numeric(predictions, errors="coerce"),
            index=self._data.index,
            dtype="float64",
        )

    @property
    def predictors(self) -> pd.DataFrame:
        return cast(pd.DataFrame, self._data[self._features].copy())

    @property
    def features(self) -> list[str]:
        """Return the candidate feature names."""
        return self._features.copy()

    @features.setter
    def features(self, features: list[str]):
        """Set the candidate feature names."""
        for feature in features:
            if feature not in self._data.columns:
                raise ValueError(f"Feature '{feature}' not found in dataset.")
            if feature in TARGET_COLUMNS + PREDICTION_COLUMNS:
                raise ValueError(f"'{feature}' cannot be used as a predictor.")
        self._features = features.copy()

    @property
    def true_total_time(self) -> pd.Series:
        """Total time of each activity (Timedelta): elapsed_time is the duration
        of each segment, so the sum over an activity is its total time."""
        return (
            self._data.groupby("activity_id")["elapsed_time"]
            .sum()
            .rename("true_total_time")
        )

    def smearing_factor(self) -> float:
        """Correction factor for the total time when the target is a log.

        exp(prediction) is a median / geometric mean, which under-estimates the
        *expected* time per segment (stops and aid stations make the pace
        distribution heavy-tailed), so the total time is too small.
        Duan's smearing multiplies the time by mean(exp(residual)).

        Compute it on the TRAINING set (ideally with out-of-fold predictions,
        in-sample residuals of a flexible model are too small), then pass it to
        `predict_total_time(smearing=...)` of the test dataset.
        Returns 1.0 for the non-log targets.
        """
        if self._target_name not in ("log_pace", "log_speed"):
            return 1.0
        residual = (self.target - self.prediction).dropna()
        if self._target_name == "log_pace":  # time ∝ pace = exp(y)
            return float(np.exp(residual).mean())
        return float(np.exp(-residual).mean())  # time ∝ 1 / speed = exp(-y)

    def predict_target_to_speed(self) -> pd.Series:
        """Convert the predicted target to a speed in m/s, for total time computation."""

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
        """Predicted total time (Timedelta) of each activity.

        `pace` is the natural target for a total time (time = pace * distance).
        With `speed` the sum of d / speed_pred is biased (E[1/v] != 1/E[v]).
        With a log target, pass `smearing=train_dataset.smearing_factor()`.
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
        from TrailPredictor.plot import PlotDataset

        return PlotDataset(self)

    def get_activities(self, activity_id: int):
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
        train_data, test_data = split_test_train(self._data)
        return Dataset(train_data, self.target_name), Dataset(
            test_data, self.target_name
        )

    @staticmethod
    def _prepare_raw(raw: pd.DataFrame, segment_width: int) -> pd.DataFrame:
        """Types + speed / pace columns. Same treatment whether the raw data
        comes from the cache or from the FIT files."""
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

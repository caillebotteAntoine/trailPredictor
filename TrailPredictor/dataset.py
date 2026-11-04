from typing import Literal, TypeAlias

import numpy as np
import pandas as pd

from TrailPredictor.utils import COMPUTED_DIR
from TrailPredictor.features import create_dataset, split_test_train
from TrailPredictor.features import build_features
from TrailPredictor.load_fit import ProgressCallback
from TrailPredictor.speed_columns import add_speed_columns


META_COLUMNS = [
    "activity_id",
    "name",
    "segment",
    "distance",
    "timestamp",
    "heart_rate",
    "elapsed_time",
    "speed",
    "position_lat",
    "position_long",
]


Target: TypeAlias = Literal["pace", "log_pace", "speed", "log_speed"]

TARGETS: tuple[Target, ...] = (
    "pace",
    "log_pace",
    "speed",
    "log_speed",
)


class Dataset:
    def __init__(
        self,
        dataset: pd.DataFrame,
        target: Target = "pace",
    ):
        self._data = dataset.copy()
        self._target: Target = target
        self._segment_width = self._data["distance"].diff().max()
        self._features = [
            col
            for col in self._data.columns
            if col not in META_COLUMNS + ["pace", "speed", "log_pace", "log_speed"]
        ]

    @property
    def segment_width(self) -> int:
        return self._segment_width

    @property
    def data(self) -> pd.DataFrame:
        """Return a copy of the underlying data."""
        return self._data.copy()

    @property
    def target(self):
        if self._target not in TARGETS:
            raise ValueError("Invalid target")
        return self._data[self._target].copy()

    @property
    def prediction_label(self) -> str:
        """Return the prediction column names corresponding to the target."""
        return f"{self._target}_pred"

    @property
    def prediction(self) -> pd.Series:
        """Return the prediction column corresponding to the target."""
        if self.prediction_label not in self._data.columns:
            raise ValueError(
                f"Prediction column '{self.prediction_label}' not found in dataset."
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
        return self._data[self._features].copy()

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
        self._features = features.copy()

    @property
    def true_total_time(self) -> pd.Series:
        """Compute the total time for each activity in hours."""
        return (
            self._data.groupby("activity_id")["elapsed_time"]
            .sum()
            .rename("true_total_time")
        )

    def predict_total_time(self) -> pd.Series:
        predictions = self.prediction

        if f"{self._target}_pred" != predictions.name:
            raise ValueError("Predictions must have the same name as the target")

        predicted = self.prediction.astype("float64")
        if self._target == "pace":
            predicted_speed = 1000.0 / (60.0 * predicted)
        elif self._target == "log_pace":
            predicted_speed = 1000.0 / (60.0 * np.exp(predicted))
        elif self._target == "speed":
            predicted_speed = predicted
        else:
            predicted_speed = np.exp(predicted)
        activity_id = self._data["activity_id"]
        return (
            pd.DataFrame(
                {
                    "activity_id": activity_id,
                    "speed": predicted_speed,
                    "target": self.target,
                }
            )
            .assign(
                predict_total_time=pd.to_timedelta(
                    self._segment_width / predicted_speed, unit="s"
                )
            )
            .groupby("activity_id")["predict_total_time"]
            .sum()
        )

    def change_target(self, target: Target):
        if target not in TARGETS:
            raise ValueError("Invalid target")
        self._target = target

    @property
    def plot(self):
        from TrailPredictor.plot import PlotDataset

        return PlotDataset(self)

    def get_activities(self, activity_id: int):
        from TrailPredictor.activities import RunningActivities

        activity_data = self._data[self._data["activity_id"] == activity_id]
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
        return Dataset(train_data, self._target), Dataset(test_data, self._target)

    @classmethod
    def load(
        cls,
        min_distance=40,
        max_distance=60,
        segment_width=1000,
        progress_cb=None,
    ):
        if min_distance >= max_distance:
            raise ValueError("min_distance must be less than max_distance")
        file_path = (
            COMPUTED_DIR / f"dataset_{min_distance}_{max_distance}_{segment_width}.csv"
        )
        try:
            dataset = pd.read_csv(file_path)

            # convert columns to appropriate types
            dataset["timestamp"] = pd.to_datetime(dataset["timestamp"])
            dataset["elapsed_time"] = pd.to_timedelta(dataset["elapsed_time"])
            if "speed" in dataset.columns:
                dataset = add_speed_columns(dataset, "speed", "speed")
            else:
                dataset = add_speed_columns(
                    dataset,
                    "elapsed_time",
                    "speed",
                )
                dataset["speed"] = (
                    segment_width / dataset["elapsed_time"].dt.total_seconds()
                )
                dataset = add_speed_columns(dataset, "speed", "speed")

        except FileNotFoundError:
            dataset = create_dataset(
                min_distance=min_distance,
                max_distance=max_distance,
                segment_width=segment_width,
                progress_cb=progress_cb,
            )

            dataset = build_features(dataset)

            COMPUTED_DIR.mkdir(parents=True, exist_ok=True)
            dataset.to_csv(file_path, index=False)

        if not {"speed", "pace", "log_speed", "log_pace"} <= set(dataset.columns):
            if "speed" in dataset.columns:
                dataset = add_speed_columns(dataset, "speed", "speed")
            else:
                raise ValueError("The dataset must contain a numeric 'speed' column.")
        return cls(dataset)

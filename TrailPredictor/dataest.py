from typing import Literal, TypeAlias

import numpy as np
import pandas as pd


META_COLUMNS = [
    "activity_id",
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
        self._features = [
            col
            for col in self._data.columns
            if col not in META_COLUMNS + ["pace", "speed"]
        ]

    @property
    def target(self):
        if self._target == "pace":
            return self._data["pace"].apply(lambda x: x.as_pace())

        if self._target == "log_pace":
            return np.log(self._data["pace"].apply(lambda x: x.as_pace()))

        if self._target == "speed":
            return self._data["speed"]

        if self._target == "log_speed":
            return np.log(self._data["speed"])

        raise ValueError("Invalid target")

    def change_target(self, target: Target):
        if target not in TARGETS:
            raise ValueError("Invalid target")
        self._target = target

    @property
    def plot(self):
        from TrailPredictor.plot import PlotDataset

        return PlotDataset(self)

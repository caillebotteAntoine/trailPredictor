"""Base class of the models: cross-validation and result persistence."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.pipeline import Pipeline

from TrailPredictor.dataset import Dataset
from TrailPredictor.features import split_test_train
from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.utils import MODELS_DIR

logger = logging.getLogger(__name__)

GROUP_COLUMN = "activity_id"  # cross-validation folds never split an activity
RMSE_SCORING = "neg_root_mean_squared_error"


class BaseModeling[ConfigT: ModelConfig](ABC):
    """Model trained on a dataset, with a validated configuration.

    The model holds a reference to the dataset (composition) and never
    modifies it. Its result is stored as a CSV file keyed by the model name,
    the target and the configuration hash, with the configuration dumped as
    JSON next to it.

    Parameters
    ----------
    dataset : Dataset
        Dataset the model is trained on. It is not modified.
    config : ModelConfig
        Validated configuration of the model.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.

    Attributes
    ----------
    result : pandas.DataFrame or None
        Result of the last run, or the saved result of the same
        configuration; None if the model has not been run yet.
    """

    name: ClassVar[str]

    def __init__(
        self, dataset: Dataset, config: ConfigT, models_dir: Path = MODELS_DIR
    ) -> None:
        self._dataset = dataset
        self._config = config
        self._models_dir = models_dir
        self.result: pd.DataFrame | None = self.load_result()

    @property
    def dataset(self) -> Dataset:
        """Return the dataset the model is trained on."""
        return self._dataset

    @property
    def config(self) -> ConfigT:
        """Return the (immutable) configuration of the model."""
        return self._config

    @property
    def result_path(self) -> Path:
        """Return the CSV file of the result of this configuration.

        Returns
        -------
        pathlib.Path
            ``<models_dir>/<name>/<target>/<config key>.csv``.
        """
        return (
            self._models_dir
            / self.name
            / self._dataset.target_name
            / f"{self._config.key}.csv"
        )

    def save_result(self) -> Path:
        """Save the result as CSV and the configuration as JSON next to it.

        Returns
        -------
        pathlib.Path
            Path of the CSV file.

        Raises
        ------
        ValueError
            If the model has no result.
        """
        if self.result is None:
            raise ValueError("The model has no result to save.")
        path = self.result_path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.result.to_csv(path, index=False)
        path.with_suffix(".json").write_text(self._config.model_dump_json(indent=2))
        logger.info("Result saved to %s", path)
        return path

    def load_result(self) -> pd.DataFrame | None:
        """Load the saved result of this configuration.

        Returns
        -------
        pandas.DataFrame or None
            Saved result, or None if this configuration has not been run.
        """
        path = self.result_path
        if not path.exists():
            logger.info("No saved result at %s: run the model first.", path)
            return None
        return pd.read_csv(path)

    def split_test_train(self) -> tuple[Dataset, Dataset]:
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
        train_data, test_data = split_test_train(
            self._dataset.data, random_state=self._config.seed
        )
        target_name = self._dataset.target_name
        return Dataset(train_data, target_name), Dataset(test_data, target_name)

    @abstractmethod
    def make_estimator(self) -> RegressorMixin:
        """Return the unfitted regressor placed after the preprocessing.

        Returns
        -------
        sklearn.base.RegressorMixin
            Unfitted regressor.
        """

    def build_pipeline(self, dataset: Dataset | None = None) -> Pipeline:
        """Build the preprocessing + regression pipeline.

        Parameters
        ----------
        dataset : Dataset or None, optional
            Dataset whose preprocessing is used. Default is None, meaning the
            dataset of the model.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline.
        """
        dataset = self._dataset if dataset is None else dataset
        return Pipeline(
            [
                ("preprocess", dataset.get_preprocess()),
                ("regression", self.make_estimator()),
            ]
        )

    def cross_validate(
        self, dataset: Dataset | None = None
    ) -> dict[str, npt.NDArray[np.float64]]:
        """Cross-validate the pipeline with folds grouped by activity.

        Parameters
        ----------
        dataset : Dataset or None, optional
            Dataset to cross-validate on, e.g. a variant of the dataset of the
            model with other features. Default is None, meaning the dataset of
            the model.

        Returns
        -------
        dict[str, numpy.ndarray]
            Scores of ``sklearn.model_selection.cross_validate``, each an
            array of shape ``(n_splits,)``; the ``"test_score"`` entry holds
            the negative RMSE of each fold.
        """
        dataset = self._dataset if dataset is None else dataset
        predictors = dataset.predictors
        target = dataset.target
        groups = dataset.data[GROUP_COLUMN]
        # folds precomputed from `groups`: no `groups=` nor metadata routing needed
        folds = list(
            GroupKFold(n_splits=self._config.n_splits).split(predictors, target, groups)
        )
        return sklearn_cross_validate(
            self.build_pipeline(dataset),
            predictors,
            target,
            scoring=RMSE_SCORING,
            cv=folds,
            error_score="raise",
        )

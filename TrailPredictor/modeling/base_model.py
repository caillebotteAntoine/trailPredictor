"""Base class of the models: cross-validation and result persistence."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
from sklearn import config_context
from sklearn.base import RegressorMixin, TransformerMixin
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_validate as sklearn_cross_validate
from sklearn.pipeline import Pipeline

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.config import ModelConfig
from TrailPredictor.utils import MODELS_DIR

logger = logging.getLogger(__name__)

GROUP_COLUMN = "activity_id"  # cross-validation folds never split an activity
RMSE_SCORING = "neg_root_mean_squared_error"


class BaseModeling[ConfigT: ModelConfig](ABC):
    """Model trained on a dataset, with a validated configuration.

    The model holds a reference to the dataset (composition) and never
    modifies it. Its result is stored as a CSV file keyed by the model name,
    the dataset hash, the target and the configuration hash, with the dataset
    hash and the configuration dumped as JSON next to it. A result is thus
    only reloaded for the very data it was computed on.

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
        """Return the CSV file of the result of this dataset and configuration.

        Returns
        -------
        pathlib.Path
            ``<models_dir>/<name>/<dataset key>/<target>/<config key>.csv``.

        See Also
        --------
        TrailPredictor.dataset.Dataset.key : Hash of the data of the dataset.
        """
        return (
            self._models_dir
            / self.name
            / self._dataset.key
            / self._dataset.target_name
            / f"{self._config.key}.csv"
        )

    def save_result(self) -> Path:
        """Save the result as CSV, and its metadata as JSON next to it.

        The JSON file holds the dataset key and the configuration, so a
        result can be traced back to the data it was computed on.

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
        metadata = {
            "dataset_key": self._dataset.key,
            "config": self._config.model_dump(mode="json"),
        }
        path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))
        logger.info("Result saved to %s (dataset %s)", path, metadata["dataset_key"])
        return path

    def load_result(self) -> pd.DataFrame | None:
        """Load the saved result of this dataset and configuration.

        Returns
        -------
        pandas.DataFrame or None
            Saved result, or None if this configuration has not been run on
            this dataset.
        """
        path = self.result_path
        if not path.exists():
            logger.info("No saved result at %s: run the model first.", path)
            return None
        return pd.read_csv(path)

    @abstractmethod
    def make_estimator(self) -> RegressorMixin:
        """Return the unfitted regressor placed after the preprocessing.

        Returns
        -------
        sklearn.base.RegressorMixin
            Unfitted regressor.
        """

    def make_pipeline_steps(self) -> list[tuple[str, TransformerMixin]]:
        """Return the unfitted steps placed between preprocessing and regression.

        Override it to add e.g. a feature selection step; the base pipeline
        has none.

        Returns
        -------
        list[tuple[str, sklearn.base.TransformerMixin]]
            Named steps, in pipeline order.
        """
        return []

    def build_pipeline(self, dataset: Dataset | None = None) -> Pipeline:
        """Build the preprocessing, intermediate steps and regression pipeline.

        Parameters
        ----------
        dataset : Dataset or None, optional
            Dataset whose preprocessing is used. Default is None, meaning the
            dataset of the model.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline.

        See Also
        --------
        make_pipeline_steps : Steps between preprocessing and regression.
        """
        dataset = self._dataset if dataset is None else dataset
        return Pipeline(
            [
                ("preprocess", dataset.get_preprocess()),
                *self.make_pipeline_steps(),
                ("regression", self.make_estimator()),
            ]
        )

    def cross_validate(
        self, dataset: Dataset | None = None, *, return_estimator: bool = False
    ) -> dict[str, Any]:
        """Cross-validate the pipeline with folds grouped by activity.

        The activity ids are routed as ``groups`` (sklearn metadata routing) to
        the outer folds and to every pipeline step requesting them, e.g. a
        ``LassoCV`` whose inner folds must also be grouped by activity.

        Parameters
        ----------
        dataset : Dataset or None, optional
            Dataset to cross-validate on, e.g. a variant of the dataset of the
            model with other features. Default is None, meaning the dataset of
            the model.
        return_estimator : bool, optional
            Also return the pipeline fitted on each fold, under the
            ``"estimator"`` key. Default is False.

        Returns
        -------
        dict[str, Any]
            Result of ``sklearn.model_selection.cross_validate``: arrays of
            shape ``(n_splits,)``, the ``"test_score"`` entry holding the
            negative RMSE of each fold, and the fitted pipelines if requested.
        """
        dataset = self._dataset if dataset is None else dataset
        # routing enabled only here, not globally: the rest of the code (and
        # the notebooks) keep the default sklearn behaviour
        with config_context(enable_metadata_routing=True):
            return sklearn_cross_validate(
                self.build_pipeline(dataset),
                dataset.predictors,
                dataset.target,
                scoring=RMSE_SCORING,
                cv=GroupKFold(n_splits=self._config.n_splits),
                params={"groups": dataset.data[GROUP_COLUMN]},
                return_estimator=return_estimator,
                error_score="raise",
            )

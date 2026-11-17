"""Entry point of the models trained on a dataset."""

from pathlib import Path

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.heart_rate import HeartRateConfig, ModelingHeartRate
from TrailPredictor.modeling.lasso import LassoConfig, ModelingLasso
from TrailPredictor.utils import MODELS_DIR


class Modeling:
    """Factory of the models trained on one dataset.

    Parameters
    ----------
    dataset : Dataset
        Dataset shared by the models. It is not modified.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.
    """

    def __init__(self, dataset: Dataset, models_dir: Path = MODELS_DIR) -> None:
        self._dataset = dataset
        self._models_dir = models_dir

    def heart_rate(self, config: HeartRateConfig | None = None) -> ModelingHeartRate:
        """Return the heart-rate zone optimisation of the dataset.

        Parameters
        ----------
        config : HeartRateConfig or None, optional
            Search space and budget. Default is None, meaning
            ``HeartRateConfig()``.

        Returns
        -------
        ModelingHeartRate
            Model with the saved result of `config`, if any.
        """
        return ModelingHeartRate(self._dataset, config, models_dir=self._models_dir)

    def lasso(self, config: LassoConfig | None = None) -> ModelingLasso:
        """Return the Lasso feature selection of the dataset.

        Parameters
        ----------
        config : LassoConfig or None, optional
            Configuration. Default is None, meaning ``LassoConfig()``.

        Returns
        -------
        ModelingLasso
            Model with the saved result of `config`, if any.
        """
        return ModelingLasso(self._dataset, config, models_dir=self._models_dir)

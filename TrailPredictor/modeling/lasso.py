"""Feature selection with a Lasso, followed by a linear regression."""

from pathlib import Path
from typing import ClassVar

import pandas as pd
from pydantic import Field, PositiveInt
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.base_model import BaseModeling
from TrailPredictor.modeling.config import MIN_SPLITS, ModelConfig
from TrailPredictor.utils import MODELS_DIR

SELECTION_COLUMNS = ["fold", "feature", "lasso_coef", "selected", "alpha", "rmse"]


class LassoConfig(ModelConfig):
    """Configuration of the Lasso feature selection.

    Parameters
    ----------
    seed : int, optional
        Seed of the Lasso coordinate descent. Default is 42.
    n_splits : int, optional
        Number of outer ``GroupKFold`` folds, measuring the RMSE, at least 2.
        Default is 5.
    inner_n_splits : int, optional
        Number of inner ``GroupKFold`` folds used by ``LassoCV`` to choose the
        penalty inside each outer training fold, at least 2. Default is 5.
    n_alphas : int, optional
        Number of penalties tried along the regularisation path. Default is
        100.
    selection_threshold : float, optional
        Features whose absolute Lasso coefficient (standardised scale) is
        below this value are dropped. Default is 1e-5.
    max_iter : int, optional
        Maximum number of coordinate descent iterations. Default is 10000.

    Raises
    ------
    pydantic.ValidationError
        If a field is unknown or invalid.
    """

    inner_n_splits: int = Field(default=5, ge=MIN_SPLITS)
    n_alphas: PositiveInt = 100
    selection_threshold: float = Field(default=1e-5, ge=0.0)
    max_iter: PositiveInt = 10_000


class ModelingLasso(BaseModeling[LassoConfig]):
    """Lasso feature selection followed by a linear regression (relaxed Lasso).

    The pipeline is preprocess -> standardisation of every column -> Lasso
    selection -> ordinary least squares on the selected features. Every
    column is standardised (ratios and one-hot columns included) so that the
    L1 penalty treats them alike: otherwise low-variance columns need larger
    coefficients and are dropped first, whatever their predictive power. The
    penalty is chosen by ``LassoCV`` with inner folds grouped by activity,
    inside each outer fold, so the RMSE is not biased by the selection.

    Parameters
    ----------
    dataset : Dataset
        Dataset the selection is run on. It is not modified.
    config : LassoConfig or None, optional
        Configuration. Default is None, meaning ``LassoConfig()``.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.

    Attributes
    ----------
    result : pandas.DataFrame or None
        One row per outer fold and feature, with columns ``SELECTION_COLUMNS``,
        or None before any run.
    """

    name: ClassVar[str] = "lasso"

    def __init__(
        self,
        dataset: Dataset,
        config: LassoConfig | None = None,
        models_dir: Path = MODELS_DIR,
    ) -> None:
        super().__init__(
            dataset, LassoConfig() if config is None else config, models_dir
        )

    def make_pipeline_steps(self) -> list[tuple[str, TransformerMixin]]:
        """Return the standardisation and Lasso selection steps.

        Returns
        -------
        list[tuple[str, sklearn.base.TransformerMixin]]
            ``("scale", StandardScaler)`` then
            ``("selection", SelectFromModel(LassoCV))``.
        """
        lasso = LassoCV(
            alphas=self._config.n_alphas,
            # receives the activity ids of the outer training fold through
            # metadata routing (see BaseModeling.cross_validate)
            cv=GroupKFold(n_splits=self._config.inner_n_splits),
            max_iter=self._config.max_iter,
            random_state=self._config.seed,
        )
        return [
            ("scale", StandardScaler().set_output(transform="pandas")),
            (
                "selection",
                SelectFromModel(lasso, threshold=self._config.selection_threshold),
            ),
        ]

    def make_estimator(self) -> LinearRegression:
        """Return the linear regression refitted on the selected features.

        Returns
        -------
        sklearn.linear_model.LinearRegression
            Unfitted regressor.
        """
        return LinearRegression()

    def evaluate_selection(self) -> pd.DataFrame:
        """Cross-validate the pipeline and record the selection of each fold.

        The result is stored in ``result`` and saved.

        Returns
        -------
        pandas.DataFrame
            One row per outer fold and feature: ``fold``, ``feature``,
            ``lasso_coef`` (standardised scale), ``selected``, ``alpha`` (penalty
            chosen in the fold) and ``rmse`` (test RMSE of the fold).
        """
        scores = self.cross_validate(return_estimator=True)
        self.result = pd.concat(
            [
                _fold_selection(fold, pipeline, -score)
                # one fitted pipeline per fold: loop over models, not data rows
                for fold, (pipeline, score) in enumerate(
                    zip(scores["estimator"], scores["test_score"], strict=True)
                )
            ],
            ignore_index=True,
        )
        self.save_result()
        return self.result

    @property
    def selection_frequency(self) -> pd.DataFrame:
        """Return how often each feature is selected across the folds.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``feature``, sorted by decreasing ``frequency`` (share
            of folds selecting it), with ``mean_abs_coef``, the mean absolute
            Lasso coefficient over the folds.

        Raises
        ------
        ValueError
            If the model has no result yet.
        """
        if self.result is None:
            raise ValueError("The model has no result: run evaluate_selection first.")
        return selection_frequency(self.result)


def _fold_selection(fold: int, pipeline: Pipeline, rmse: float) -> pd.DataFrame:
    """Return the selection of one fitted pipeline, one row per feature."""
    selector = pipeline.named_steps["selection"]
    return pd.DataFrame(
        {
            "fold": fold,
            "feature": selector.feature_names_in_,
            "lasso_coef": selector.estimator_.coef_,
            "selected": selector.get_support(),
            "alpha": selector.estimator_.alpha_,
            "rmse": rmse,
        },
        columns=SELECTION_COLUMNS,
    )


def selection_frequency(selection: pd.DataFrame) -> pd.DataFrame:
    """Summarise a per-fold selection by feature.

    Parameters
    ----------
    selection : pandas.DataFrame
        One row per fold and feature, with ``feature``, ``selected`` and
        ``lasso_coef`` columns, as ``ModelingLasso.result``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``feature``, sorted by decreasing ``frequency`` then
        ``mean_abs_coef``.

    Raises
    ------
    ValueError
        If a column is missing.

    Examples
    --------
    >>> selection = pd.DataFrame(
    ...     {
    ...         "feature": ["a", "b", "a", "b"],
    ...         "selected": [True, True, True, False],
    ...         "lasso_coef": [2.0, 0.5, 1.0, 0.0],
    ...     }
    ... )
    >>> selection_frequency(selection)  # doctest: +NORMALIZE_WHITESPACE
             frequency  mean_abs_coef
    feature
    a              1.0           1.50
    b              0.5           0.25
    """
    missing = {"feature", "selected", "lasso_coef"} - set(selection.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")
    return (
        selection.assign(abs_coef=selection["lasso_coef"].abs())
        .groupby("feature")
        .agg(
            frequency=("selected", "mean"),
            mean_abs_coef=("abs_coef", "mean"),
        )
        .sort_values(["frequency", "mean_abs_coef"], ascending=False)
    )

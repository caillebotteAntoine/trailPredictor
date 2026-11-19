"""Feature selection with a Lasso, followed by a linear regression."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import Field, PositiveInt, ValidationError
from sklearn import config_context
from sklearn.base import TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LinearRegression, lasso_path
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.base_model import GROUP_COLUMN, BaseModeling
from TrailPredictor.modeling.config import MIN_SPLITS, ModelConfig
from TrailPredictor.speed_columns import TARGETS, Target
from TrailPredictor.utils import MODELS_DIR

if TYPE_CHECKING:
    from TrailPredictor.plot.lasso import PlotLasso

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True, slots=True)
class LassoPath:
    """Regularisation path of a ``LassoCV`` and its cross-validation error.

    Parameters
    ----------
    coef_path : pandas.DataFrame
        Standardised coefficients along the path, indexed by decreasing α,
        one column per feature.
    mse_path : ndarray of shape (n_alphas, n_folds)
        Cross-validation MSE of each α (rows aligned with `coef_path`) and
        inner fold, in squared target units.
    alpha : float
        α selected by ``LassoCV`` (minimum mean CV error).
    threshold : float, optional
        A feature is active when its absolute coefficient is above this
        value. Default is 0.0.

    Raises
    ------
    ValueError
        If `mse_path` is not 2D or has not one row per α of `coef_path`.

    See Also
    --------
    ModelingLasso.fit_regularization_path : Build the path of a model.
    """

    coef_path: pd.DataFrame
    mse_path: npt.NDArray[np.float64]
    alpha: float
    threshold: float = 0.0

    def __post_init__(self) -> None:
        """Check that the CV errors are aligned with the path."""
        if self.mse_path.ndim != 2 or len(self.mse_path) != len(self.coef_path):
            raise ValueError(
                f"mse_path of shape {self.mse_path.shape} does not match "
                f"{len(self.coef_path)} alphas"
            )

    @property
    def alphas(self) -> npt.NDArray[np.float64]:
        """Return the decreasing α grid, of shape (n_alphas,)."""
        return self.coef_path.index.to_numpy(dtype=np.float64)

    @property
    def mse_mean(self) -> npt.NDArray[np.float64]:
        """Return the mean CV MSE over the folds, of shape (n_alphas,)."""
        return self.mse_path.mean(axis=1)

    @property
    def mse_se(self) -> npt.NDArray[np.float64]:
        """Return the standard error of the CV MSE, of shape (n_alphas,)."""
        return self.mse_path.std(axis=1) / np.sqrt(self.mse_path.shape[1])

    @property
    def mse_threshold(self) -> float:
        """Return the minimum mean CV MSE plus its standard error."""
        i_min = int(np.argmin(self.mse_mean))
        return float(self.mse_mean[i_min] + self.mse_se[i_min])

    @property
    def alpha_1se(self) -> float:
        """Return the largest α within one standard error of the minimum.

        Returns
        -------
        float
            α of the "one standard error" rule: the sparsest model whose
            mean CV MSE is at most ``mse_threshold``.
        """
        return float(self.alphas[self.mse_mean <= self.mse_threshold].max())

    @property
    def n_active(self) -> pd.Series:
        """Return the number of active features at each α of the path."""
        return (self.coef_path.abs() > self.threshold).sum(axis=1)

    def coefs_at(self, alpha: float) -> pd.Series:
        """Return the coefficients at the path point closest to `alpha`.

        Parameters
        ----------
        alpha : float
            Strictly positive α; closeness is measured on a log scale.

        Returns
        -------
        pandas.Series
            Standardised coefficients, indexed by feature name.
        """
        i_closest = int(np.argmin(np.abs(np.log(self.alphas) - np.log(alpha))))
        return self.coef_path.iloc[i_closest]


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

    @property
    def plot(self) -> "PlotLasso":
        """Return a plotting interface of the Lasso selection.

        Returns
        -------
        PlotLasso
            Plotting interface sharing the dataset and configuration.
        """
        # Lazy import: TrailPredictor.plot.lasso subclasses ModelingLasso.
        from TrailPredictor.plot.lasso import (  # pylint: disable=import-outside-toplevel
            PlotLasso,
        )

        return PlotLasso(self)

    def fit_regularization_path(self) -> LassoPath:
        """Fit the selection on the whole dataset and compute its Lasso path.

        The preprocessing, standardisation and ``LassoCV`` steps are fitted on
        every activity (inner folds grouped by activity), then the
        coefficients are recomputed by coordinate descent on the ``LassoCV``
        grid ``alphas_``, so they line up with its CV error ``mse_path_``.

        Returns
        -------
        LassoPath
            Standardised coefficients and CV error along the path.
        """
        predictors, target = self._dataset.predictors, self._dataset.target
        selection = Pipeline(
            [
                ("preprocess", self._dataset.get_preprocess()),
                *self.make_pipeline_steps(),
            ]
        )
        with config_context(enable_metadata_routing=True):
            selection.fit(predictors, target, groups=self._dataset.data[GROUP_COLUMN])
        lasso_cv = selection.named_steps["selection"].estimator_
        scaled = selection[:-1].transform(predictors)

        # centring X and y is what fit_intercept=True does inside LassoCV
        design = scaled.to_numpy(dtype=np.float64)
        response = target.to_numpy(dtype=np.float64)
        alphas, coefs, _ = lasso_path(
            design - design.mean(axis=0),
            response - response.mean(),
            alphas=lasso_cv.alphas_,
            max_iter=self._config.max_iter,
        )
        return LassoPath(
            coef_path=pd.DataFrame(coefs.T, index=alphas, columns=scaled.columns),
            mse_path=np.asarray(lasso_cv.mse_path_, dtype=np.float64),
            alpha=float(lasso_cv.alpha_),
            threshold=self._config.selection_threshold,
        )

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


def summarize_folds(selection: pd.DataFrame) -> pd.DataFrame:
    """Summarise a per-fold selection by outer fold.

    Parameters
    ----------
    selection : pandas.DataFrame
        One row per fold and feature, with the ``SELECTION_COLUMNS`` columns,
        as ``ModelingLasso.result``.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``fold``, with columns ``alpha`` (penalty chosen in the
        fold), ``rmse`` (its test RMSE) and ``n_selected`` (its number of
        selected features).

    Raises
    ------
    ValueError
        If a column is missing.

    Examples
    --------
    >>> selection = pd.DataFrame(
    ...     {
    ...         "fold": [0, 0, 1, 1],
    ...         "selected": [True, True, True, False],
    ...         "alpha": [0.1, 0.1, 0.2, 0.2],
    ...         "rmse": [1.0, 1.0, 2.0, 2.0],
    ...     }
    ... )
    >>> summarize_folds(selection)  # doctest: +NORMALIZE_WHITESPACE
          alpha  rmse  n_selected
    fold
    0       0.1   1.0           2
    1       0.2   2.0           1
    """
    missing = {"fold", "selected", "alpha", "rmse"} - set(selection.columns)
    if missing:
        raise ValueError(f"Missing selection columns: {sorted(missing)}")
    return selection.groupby("fold").agg(
        alpha=("alpha", "first"),
        rmse=("rmse", "first"),
        n_selected=("selected", "sum"),
    )


@dataclass(frozen=True, slots=True)
class SavedLassoResult:
    """Saved result of a Lasso feature selection.

    Parameters
    ----------
    target_name : Target
        Target the RMSE was computed on.
    config : LassoConfig
        Configuration of the selection, read from its JSON dump.
    result_path : pathlib.Path
        CSV file of the per-fold selection.
    dataset_key : str
        Hash of the dataset the selection was run on (see ``Dataset.key``).
    dataset_name : str or None, optional
        Name of that dataset, if it had one. Default is None.
    """

    target_name: Target
    config: LassoConfig
    result_path: Path
    dataset_key: str
    dataset_name: str | None = None


def list_saved_results(
    dataset_key: str, models_dir: Path = MODELS_DIR
) -> list[SavedLassoResult]:
    """List the saved Lasso selections of one dataset.

    A result is listed when its CSV file and its JSON metadata both exist
    under ``<models_dir>/lasso/<dataset_key>/<target>/``. Metadata that
    cannot be read are skipped and logged.

    Parameters
    ----------
    dataset_key : str
        Hash of the dataset (see ``Dataset.key``); only its results are
        listed, since the regularisation path needs the very same data.
    models_dir : pathlib.Path, optional
        Root directory of the saved results. Default is ``MODELS_DIR``.

    Returns
    -------
    list of SavedLassoResult
        Saved results, sorted by target then configuration key; empty when
        nothing is saved for this dataset.

    See Also
    --------
    BaseModeling.save_result : Writes the CSV and JSON files.
    """
    saved = []
    dataset_dir = models_dir / ModelingLasso.name / dataset_key
    # loop over the saved files, not over the data
    for metadata_path in sorted(dataset_dir.glob("*/*.json")):
        result_path = metadata_path.with_suffix(".csv")
        target_name = metadata_path.parent.name
        if not result_path.exists() or target_name not in TARGETS:
            logger.warning(
                "Skipping %s: no CSV result or unknown target", metadata_path
            )
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
            config = LassoConfig.model_validate(metadata["config"])
            dataset_name = metadata.get("dataset_name")
        # TypeError / KeyError: valid JSON, but not the metadata of save_result
        except (json.JSONDecodeError, KeyError, TypeError, ValidationError):
            logger.exception("Skipping %s: invalid metadata", metadata_path)
            continue
        saved.append(
            SavedLassoResult(
                cast(Target, target_name),
                config,
                result_path,
                dataset_key=dataset_key,
                dataset_name=dataset_name,
            )
        )
    return saved

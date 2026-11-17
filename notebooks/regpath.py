"""Regularisation path plot of an already fitted LassoCV or LassoLarsIC.

The bottom panel shows the selection criterion of α:

* LassoCV: cross-validation error ± 1 standard error (and the
  "one standard error" rule);
* LassoLarsIC: information criterion (AIC or BIC) along the LARS path.

Usage::

    from regpath import LassoPath, plot_lasso_path, summarize

    res = LassoPath.from_pipeline(lasso_full, X, y_model, lasso_step="lasso")
    fig, axes = plot_lasso_path(res, mse_label="MSE (log1p)")
    display_bullet(*summarize(res))

`y_model` must be the target *as seen by the Lasso* (e.g. ``np.log1p(y)``
when the target is transformed), otherwise the recomputed path does not match
the criterion. `from_pipeline` also accepts the estimator itself (without a
Pipeline); ``lasso_step=None`` selects the last step of the Pipeline.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import lars_path, lasso_path

GREY = "#9e9e9e"
MAIN = "#1f4e79"
ACCENT = "#d62728"
SECONDARY = "#2ca02c"
PALETTE = plt.get_cmap("tab10").colors
_LINESTYLES = ("-", "--", "-.", ":")


def _is_cv(obj) -> bool:
    return hasattr(obj, "mse_path_") and hasattr(obj, "alphas_")


def _is_ic(obj) -> bool:
    return hasattr(obj, "criterion_") and hasattr(obj, "alphas_")


def _unwrap_lasso(step, max_depth: int = 5):
    """Find the fitted LassoCV / LassoLarsIC, even when wrapped.

    Wrappers such as SelectFromModel, TransformedTargetRegressor or
    GridSearchCV are unwrapped up to `max_depth` levels.
    """
    obj = step
    for _ in range(max_depth):
        if _is_cv(obj):
            if np.ndim(obj.mse_path_) != 2:
                raise ValueError(
                    "mse_path_ doit être 2D (n_alphas, n_folds) : un seul l1_ratio attendu."
                )
            return obj
        if _is_ic(obj):
            return obj
        for attr in ("estimator_", "regressor_", "best_estimator_"):
            if hasattr(obj, attr):
                obj = getattr(obj, attr)
                break
        else:
            break
    raise TypeError(
        f"Aucun LassoCV ni LassoLarsIC ajusté trouvé dans {type(step).__name__}."
    )


_unwrap_lasso_cv = _unwrap_lasso  # compatibilité


def _positive_alphas(alphas: np.ndarray) -> np.ndarray:
    """Replace non-positive alphas by a tenth of the smallest positive one.

    The LARS path often ends at α = 0, which cannot be shown on a log scale.
    """
    alphas = np.asarray(alphas, dtype=float).copy()
    zero = alphas <= 0
    if zero.any():
        alphas[zero] = alphas[~zero].min() / 10
    return alphas


@dataclass
class LassoPath:
    """Regularisation path and selection criterion of α.

    Parameters
    ----------
    coef_path : pandas.DataFrame
        Coefficients along the path, indexed by α, one column per feature.
    crit_alphas : numpy.ndarray of shape (n_alphas,)
        Values of α where the criterion is evaluated.
    crit_path : numpy.ndarray of shape (n_alphas, n_folds) or (n_alphas,)
        Cross-validation MSE when `criterion` is ``"cv"``, information
        criterion when it is ``"aic"`` or ``"bic"``.
    alpha : float
        Selected α (minimum of the criterion).
    tol : float, optional
        A feature is active when ``|coef| > tol``. Default is 0.0.
    criterion : {"cv", "aic", "bic"}, optional
        Selection criterion. Default is ``"cv"``.
    alpha_zero : float, optional
        Value shown in place of α = 0 on the log scale.

    Attributes
    ----------
    crit_mean : numpy.ndarray of shape (n_alphas,)
        Mean criterion over the folds (the criterion itself for AIC / BIC).
    crit_se : numpy.ndarray of shape (n_alphas,) or None
        Standard error of the CV criterion, None for AIC / BIC.
    threshold : float or None
        Minimum mean CV error plus one standard error, None for AIC / BIC.
    alpha_1se : float or None
        Largest α whose mean CV error is below `threshold`, None for
        AIC / BIC.
    active : pandas.DataFrame
        Whether each coefficient is active along the path.
    n_active : pandas.Series
        Number of active features along the path.
    """

    coef_path: pd.DataFrame  # index = alphas, colonnes = variables
    crit_alphas: np.ndarray  # alphas où le critère est évalué
    crit_path: np.ndarray  # (n_alphas, n_folds) en CV, (n_alphas,) pour AIC/BIC
    alpha: float  # α retenu (minimum du critère)
    tol: float = 0.0  # |coef| > tol  ⇒ variable active
    criterion: str = "cv"
    alpha_zero: float | None = None  # valeur affichée à la place de α = 0 (échelle log)

    crit_mean: np.ndarray = field(init=False, repr=False)
    crit_se: np.ndarray | None = field(init=False, repr=False)
    threshold: float | None = field(init=False)
    alpha_1se: float | None = field(init=False)
    active: pd.DataFrame = field(init=False, repr=False)
    n_active: pd.Series = field(init=False, repr=False)

    def __post_init__(self):
        """Sort the path and compute the derived criterion statistics."""
        self.criterion = self.criterion.lower()
        self.coef_path = self.coef_path.sort_index(ascending=False)  # α décroissants
        self.crit_alphas = np.asarray(self.crit_alphas, dtype=float)
        self.crit_path = np.asarray(self.crit_path, dtype=float)
        if self.is_cv:
            self.crit_mean = self.crit_path.mean(axis=1)
            self.crit_se = self.crit_path.std(axis=1) / np.sqrt(self.crit_path.shape[1])
            i_min = self.crit_mean.argmin()
            self.threshold = float(self.crit_mean[i_min] + self.crit_se[i_min])
            self.alpha_1se = float(
                self.crit_alphas[self.crit_mean <= self.threshold].max()
            )
        else:
            self.crit_mean = self.crit_path.ravel()
            self.crit_se = None
            self.threshold = None
            self.alpha_1se = None
        self.active = self.coef_path.abs() > self.tol
        self.n_active = self.active.sum(axis=1)

    # --- propriétés utiles
    @property
    def is_cv(self) -> bool:
        """Whether α is selected by cross-validation."""
        return self.criterion == "cv"

    @property
    def alpha_ref(self) -> float:
        """Reference α for highlighting: 1-SE α in CV, selected α otherwise."""
        return self.alpha_1se if self.alpha_1se is not None else self.alpha

    @property
    def criterion_label(self) -> str:
        """Display name of the criterion: ``"CV"``, ``"AIC"`` or ``"BIC"``."""
        return "CV" if self.is_cv else self.criterion.upper()

    def fmt_alpha(self, alpha: float) -> str:
        """Format α, flagging the α = 0 (least squares) point of the LARS path.

        Parameters
        ----------
        alpha : float
            Value of α.

        Returns
        -------
        str
            α with 4 significant digits, or a least-squares label.
        """
        if self.alpha_zero is not None and np.isclose(alpha, self.alpha_zero):
            return "0 (moindres carrés)"
        return f"{alpha:.4g}"

    # --- alias de compatibilité (anciens noms, spécifiques à la CV)
    cv_alphas = property(lambda self: self.crit_alphas)
    mse_path = property(lambda self: self.crit_path)
    mse_mean = property(lambda self: self.crit_mean)
    mse_se = property(lambda self: self.crit_se)

    @classmethod
    def from_pipeline(
        cls,
        pipeline,
        X,
        y,
        lasso_step: str | None = "lasso",
        feature_names: Sequence[str] | None = None,
        max_iter: int = 50_000,
        tol: float = 0.0,
    ) -> "LassoPath":
        """Recompute the path of the LassoCV / LassoLarsIC in `pipeline`.

        Parameters
        ----------
        pipeline : sklearn.pipeline.Pipeline or estimator
            Fitted pipeline, or the fitted estimator itself (then without
            preprocessing). Every step before `lasso_step` is treated as
            preprocessing.
        X : array-like of shape (n_samples, n_features)
            Training data, before preprocessing.
        y : array-like of shape (n_samples,)
            Target as seen by the Lasso (after any target transformation).
        lasso_step : str or None, optional
            Name of the Lasso step; None selects the last step. Default is
            ``"lasso"``.
        feature_names : sequence of str, optional
            Feature names; inferred from the preprocessing or `X` when
            omitted.
        max_iter : int, optional
            Maximum number of coordinate descent iterations (LassoCV only).
            Default is 50 000.
        tol : float, optional
            A feature is active when ``|coef| > tol``. Default is 0.0.

        Returns
        -------
        LassoPath
            The recomputed path and its criterion.

        Raises
        ------
        TypeError
            If no fitted LassoCV or LassoLarsIC is found.

        Warns
        -----
        UserWarning
            If the recomputed LARS path does not match ``alphas_`` of the
            LassoLarsIC, which usually means `y` is not the target seen by
            the estimator.

        Notes
        -----
        LassoCV: coordinate descent path on the CV grid ``alphas_``.
        LassoLarsIC: exact LARS path (same breakpoints as ``alphas_``).
        """
        if hasattr(pipeline, "named_steps"):
            names = list(pipeline.named_steps)
            idx = len(names) - 1 if lasso_step is None else names.index(lasso_step)
            est = _unwrap_lasso(pipeline[idx])
            prep = pipeline[:idx] if idx > 0 else None
        else:
            est, prep = _unwrap_lasso(pipeline), None

        Z = prep.transform(X) if prep is not None else X
        Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z, dtype=float)

        if feature_names is None:
            try:
                feature_names = (
                    prep.get_feature_names_out() if prep is not None else X.columns
                )
            except AttributeError:
                feature_names = [f"x{i}" for i in range(Z.shape[1])]

        # Centrage de Z et y : équivalent à fit_intercept=True de l'estimateur.
        y = np.asarray(y, dtype=float).ravel()
        if getattr(est, "fit_intercept", True):
            Z, y = Z - Z.mean(axis=0), y - y.mean()
        positive = getattr(est, "positive", False)

        alpha_zero = None
        if _is_cv(est):
            alphas, coefs, _ = lasso_path(
                Z, y, alphas=est.alphas_, max_iter=max_iter, positive=positive
            )
            crit_alphas, crit_path, criterion = est.alphas_, est.mse_path_, "cv"
            alpha = float(est.alpha_)
        else:
            alphas, _, coefs = lars_path(
                Z,
                y,
                method="lasso",
                alpha_min=0.0,
                max_iter=est.max_iter,
                eps=est.eps,
                positive=positive,
            )
            if len(alphas) != len(est.alphas_) or not np.allclose(alphas, est.alphas_):
                warnings.warn(
                    "Le chemin LARS recalculé ne coïncide pas avec `alphas_` du "
                    "LassoLarsIC : vérifiez que `y` est la cible vue par l'estimateur."
                )
            criterion = est.criterion
            # α = 0 (fin du chemin LARS) n'est pas représentable en échelle log
            alphas = _positive_alphas(alphas)
            crit_alphas = _positive_alphas(est.alphas_)
            if np.any(np.asarray(est.alphas_) <= 0):
                alpha_zero = float(crit_alphas.min())
            crit_path = est.criterion_
            alpha = float(crit_alphas[np.argmin(est.criterion_)])

        coef_path = pd.DataFrame(coefs.T, index=alphas, columns=list(feature_names))
        return cls(coef_path, crit_alphas, crit_path, alpha, tol, criterion, alpha_zero)

    def coefs_at(self, alpha: float) -> pd.Series:
        """Return the coefficients at the path point closest to `alpha`.

        Parameters
        ----------
        alpha : float
            Strictly positive α; closeness is measured on a log scale.

        Returns
        -------
        pandas.Series
            Coefficients, indexed by feature name.
        """
        i = np.argmin(np.abs(np.log(self.coef_path.index.values) - np.log(alpha)))
        return self.coef_path.iloc[i]

    def n_active_at(self, alpha: float) -> int:
        """Return the number of active features at the path point closest to `alpha`.

        Parameters
        ----------
        alpha : float
            Strictly positive α.

        Returns
        -------
        int
            Number of coefficients with ``|coef| > tol``.
        """
        return int((self.coefs_at(alpha).abs() > self.tol).sum())


def plot_lasso_path(
    res: LassoPath,
    n_highlight: int | None = None,
    *,
    palette: Sequence = PALETTE,
    grey: str = GREY,
    main: str = MAIN,
    accent: str = ACCENT,
    secondary: str = SECONDARY,
    title: str = "Chemin de régularisation du Lasso (variables standardisées)",
    cv_title: str | None = None,
    mse_label: str | None = None,
    n_top_ticks: int = 8,
    figsize: tuple[float, float] = (13, 9.5),
):
    """Plot the coefficient path (top) and the selection criterion of α (bottom).

    Parameters
    ----------
    res : LassoPath
        Regularisation path to plot.
    n_highlight : int, optional
        Number of coloured features, the largest in ``|coef|`` at the
        reference α (1-SE α in CV, selected α otherwise). Defaults to every
        feature active at that α.
    palette : sequence, optional
        Colours of the highlighted features. Default is ``PALETTE``.
    grey : str, optional
        Colour of the other features. Default is ``GREY``.
    main : str, optional
        Colour of the criterion curve. Default is ``MAIN``.
    accent : str, optional
        Colour of the selected α. Default is ``ACCENT``.
    secondary : str, optional
        Colour of the 1-SE α. Default is ``SECONDARY``.
    title : str, optional
        Title of the top panel.
    cv_title : str, optional
        Title of the bottom panel, adapted to the criterion by default.
    mse_label : str, optional
        y-axis label of the bottom panel, adapted to the criterion by
        default.
    n_top_ticks : int, optional
        Approximate number of ticks of the top axis, which shows the number
        of active features. Default is 8.
    figsize : tuple of float, optional
        Figure size in inches. Default is ``(13, 9.5)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure.
    axes : tuple of matplotlib.axes.Axes
        The path axis and the criterion axis.

    See Also
    --------
    summarize : Comment sentences for this figure.

    Notes
    -----
    The bottom panel shows the CV error ± 1 standard error for a LassoCV, or
    the AIC / BIC for a LassoLarsIC.
    """
    if cv_title is None:
        cv_title = (
            "Erreur de validation croisée interne"
            if res.is_cv
            else f"Critère d'information ({res.criterion_label}) le long du chemin LARS"
        )
    if mse_label is None:
        mse_label = "MSE" if res.is_cv else res.criterion_label

    coefs_ref = res.coefs_at(res.alpha_ref).abs()
    if n_highlight is None:
        n_highlight = int((coefs_ref > res.tol).sum())
    highlight = coefs_ref.nlargest(n_highlight).index
    cp = res.coef_path

    fig, (ax_path, ax_crit) = plt.subplots(
        2, 1, figsize=figsize, sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
    )

    # --- Chemin des coefficients
    for f in cp.columns.difference(highlight):
        ax_path.plot(cp.index, cp[f], color=grey, lw=0.8, alpha=0.4)
    for i, f in enumerate(highlight):  # palette épuisée → on change de style de trait
        ax_path.plot(
            cp.index,
            cp[f],
            lw=2,
            label=f,
            color=palette[i % len(palette)],
            ls=_LINESTYLES[(i // len(palette)) % len(_LINESTYLES)],
        )
    ax_path.axhline(0, color="0.3", lw=0.8)
    ax_path.set(xscale="log", title=title, ylabel="coefficient")

    step = max(len(cp) // n_top_ticks, 1)
    top_ax = ax_path.secondary_xaxis("top")
    top_ax.set_xticks(cp.index[::step], res.n_active.iloc[::step].astype(str))
    top_ax.set_xlabel("variables actives", fontsize=9)

    # --- Critère de sélection
    if res.is_cv:
        ax_crit.plot(
            res.crit_alphas, res.crit_mean, color=main, lw=2, label="erreur CV moyenne"
        )
        ax_crit.fill_between(
            res.crit_alphas,
            res.crit_mean - res.crit_se,
            res.crit_mean + res.crit_se,
            color=main,
            alpha=0.2,
            label="± 1 erreur-type",
        )
        # Seuil « MSE min + 1 erreur-type », entre l'α retenu et l'α 1-SE
        ax_crit.hlines(
            res.threshold, res.alpha, res.alpha_1se, color="0.3", ls=":", lw=2
        )
    else:
        ax_crit.plot(
            res.crit_alphas,
            res.crit_mean,
            color=main,
            lw=2,
            marker="o",
            ms=3,
            label=res.criterion_label,
        )
    ax_crit.set(xlabel="α (échelle log)", ylabel=mse_label, title=cv_title)

    for ax in (ax_path, ax_crit):
        ax.axvline(
            res.alpha,
            color=accent,
            ls="--",
            lw=1.2,
            label=f"α retenu ({res.criterion_label}) = {res.fmt_alpha(res.alpha)}",
        )
        if res.alpha_1se is not None:
            ax.axvline(
                res.alpha_1se,
                color=secondary,
                ls=":",
                lw=1.5,
                label=f"α 1 erreur-type = {res.alpha_1se:.4g}",
            )
        ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))

    fig.tight_layout()
    return fig, (ax_path, ax_crit)


def summarize(res: LassoPath, n_first: int = 6) -> list[str]:
    """Write comment sentences for the regularisation path figure.

    Parameters
    ----------
    res : LassoPath
        Regularisation path.
    n_first : int, optional
        Number of first features entering the model to list. Default is 6.

    Returns
    -------
    list of str
        Sentences (in French) to pass to ``print``, ``display_bullet``, etc.
    """
    entered = res.active.any()
    # α décroissants → le premier True est le plus grand α où la variable est non nulle
    entry_alpha = res.active.idxmax()[entered].sort_values(ascending=False)
    lines = [
        "Premières variables à entrer dans le modèle quand α diminue : "
        + ", ".join(f"`{f}`" for f in entry_alpha.index[:n_first])
        + "."
    ]
    if res.is_cv:
        lines.append(
            f"À l'α retenu ({res.alpha:.4g}), {res.n_active_at(res.alpha)} variables sont "
            f"actives ; avec la règle « 1 erreur-type » (α = {res.alpha_1se:.4g}), "
            f"{res.n_active_at(res.alpha_1se)} variables suffiraient pour une erreur CV "
            "équivalente à une erreur-type près."
        )
    else:
        lines.append(
            f"{'Le' if res.criterion == 'bic' else "L'"}{' ' if res.criterion == 'bic' else ''}"
            f"{res.criterion_label} est minimal en α = {res.fmt_alpha(res.alpha)}, où "
            f"{res.n_active_at(res.alpha)} variables sont actives "
            f"(sur {res.coef_path.shape[1]}). "
            + (
                "Le BIC pénalise davantage la complexité que l'AIC et tend à retenir "
                "un modèle plus parcimonieux."
                if res.criterion == "bic"
                else "L'AIC pénalise moins la complexité que le BIC et tend à retenir "
                "davantage de variables."
            )
        )
    lines.append(
        "Les courbes qui se croisent ou changent de signe le long du chemin signalent des "
        "variables corrélées qui se substituent l'une à l'autre."
    )
    return lines


if __name__ == "__main__":  # démo rapide : python regpath.py
    from sklearn.datasets import make_regression
    from sklearn.feature_selection import SelectFromModel
    from sklearn.linear_model import LassoCV, LassoLarsIC
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X, y = make_regression(
        n_samples=300, n_features=25, n_informative=8, noise=20, random_state=0
    )
    X = pd.DataFrame(X, columns=[f"var_{i}" for i in range(X.shape[1])])

    for lasso in (LassoCV(cv=5, random_state=0), LassoLarsIC(criterion="bic")):
        pipe = Pipeline(
            [("scaler", StandardScaler()), ("lasso", SelectFromModel(lasso))]
        ).fit(X, y)
        res = LassoPath.from_pipeline(pipe, X, y)
        fig, _ = plot_lasso_path(res)
        print(f"--- {type(lasso).__name__}")
        print("\n".join(summarize(res)))
    plt.show()

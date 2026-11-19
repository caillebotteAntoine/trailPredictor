"""Streamlit page running and showing the Lasso feature selection.

Two modes: load a saved selection, or run a new one. Both work on the dataset
loaded in the "Dataset / Features" page (shared through ``st.session_state``):
the results are keyed by its hash, so only those of this very dataset are
offered.
"""

# Streamlit orders pages by their numeric file-name prefix.
# pylint: disable=invalid-name
# `_render_chart` and the dataset helpers mirror pages/3_heart_rate_zone.py:
# page scripts cannot import each other; a shared UI module would remove the
# duplicate.
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.lasso import (
    LassoConfig,
    ModelingLasso,
    SavedLassoResult,
    list_saved_results,
    selection_frequency,
    summarize_folds,
)
from TrailPredictor.plot import PlotActivities, PlotLasso
from TrailPredictor.speed_columns import TARGETS, Target

logger = logging.getLogger(__name__)

# Dataset loaded by the "Dataset / Features" page; keep in sync with
# pages/2_dataset_features.py.
DATASET_KEY = "app_dataset"
DATASET_PARAMS_KEY = "app_dataset_params"
# Last selection run (or reloaded) from the form of this page.
RUN_KEY = "lasso_last_run"

MODE_LOAD = "Charger un modèle sauvegardé"
MODE_RUN = "Nouvelle sélection"
DEFAULT_CONFIG = LassoConfig()
PATH_CHART_HEIGHT = 800
ACTIVITY_CHART_HEIGHT = 500
PREDICTED_ACTIVITY_ID = 0  # activity shown with its out-of-fold prediction


# --------------------------------------------------------------------------- #
#                                 CHARGEMENT
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)
def st_list_saved_results(dataset_key: str) -> list[SavedLassoResult]:
    """List the saved Lasso selections of a dataset (cached, refreshed every minute).

    Parameters
    ----------
    dataset_key : str
        Hash of the dataset loaded in the application.

    Returns
    -------
    list of SavedLassoResult
        Saved results of this dataset found in ``MODELS_DIR``.
    """
    return list_saved_results(dataset_key)


@st.cache_data
def load_selection(result_path: Path) -> pd.DataFrame:
    """Read the per-fold selection of a saved result (cached).

    Parameters
    ----------
    result_path : pathlib.Path
        CSV file of the selection.

    Returns
    -------
    pandas.DataFrame
        One row per outer fold and feature.
    """
    return pd.read_csv(result_path)


# The dataset is not hashed (slow): its key identifies it in the cache.
@st.cache_resource(
    show_spinner="Calcul du chemin de régularisation...",
    hash_funcs={Dataset: lambda _dataset: None},
)
def fit_plot(
    dataset: Dataset, dataset_key: str, target_name: Target, config_json: str
) -> PlotLasso:
    """Build the plotting interface and fit its regularisation path (cached).

    The path is fitted here, inside the cache, so the cached object is not
    modified afterwards.

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application; not hashed by the cache, which
        relies on `dataset_key` instead.
    dataset_key : str
        Hash of `dataset`, identifying it in the cache.
    target_name : Target
        Target of the selection.
    config_json : str
        JSON dump of the `LassoConfig` (hashable cache key).

    Returns
    -------
    PlotLasso
        Plotting interface whose ``path`` is already fitted.
    """
    del dataset_key  # cache key only
    config = LassoConfig.model_validate_json(config_json)
    plot = ModelingLasso(dataset.with_target(target_name), config).plot
    _ = plot.path  # fit once, inside the cache
    return plot


# The dataset is not hashed (slow): its key identifies it in the cache.
@st.cache_data(
    show_spinner="Prédiction hors-pli des activités...",
    hash_funcs={Dataset: lambda _dataset: None},
)
def out_of_fold_prediction(
    dataset: Dataset, dataset_key: str, target_name: Target, config_json: str
) -> pd.Series:
    """Predict every segment with the pipeline fitted on the other folds (cached).

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application; not hashed by the cache, which
        relies on `dataset_key` instead.
    dataset_key : str
        Hash of `dataset`, identifying it in the cache.
    target_name : Target
        Target of the selection.
    config_json : str
        JSON dump of the `LassoConfig` (hashable cache key).

    Returns
    -------
    pandas.Series
        Out-of-fold prediction of `target_name`, indexed like the dataset.
    """
    del dataset_key  # cache key only
    config = LassoConfig.model_validate_json(config_json)
    return ModelingLasso(dataset.with_target(target_name), config).cross_val_predict()


def run_selection(
    dataset: Dataset, dataset_key: str, target_name: Target, config: LassoConfig
) -> SavedLassoResult:
    """Run the cross-validated Lasso selection, unless already saved.

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application. It is not modified.
    dataset_key : str
        Hash of `dataset`.
    target_name : Target
        Target the RMSE is computed on.
    config : LassoConfig
        Configuration of the selection.

    Returns
    -------
    SavedLassoResult
        Saved selection of `config`.
    """
    model = ModelingLasso(dataset.with_target(target_name), config)
    if model.result is not None:
        st.info("Résultat déjà sauvegardé pour cette configuration : rechargé.")
    else:
        with st.spinner("Sélection LASSO en validation croisée..."):
            model.evaluate_selection()
        st_list_saved_results.clear()
        st.success("Sélection terminée et sauvegardée.")
    return SavedLassoResult(
        target_name,
        config,
        model.result_path,
        dataset_key=dataset_key,
        dataset_name=dataset.name,
    )


# --------------------------------------------------------------------------- #
#                                   SIDEBAR
# --------------------------------------------------------------------------- #
def format_label(saved: SavedLassoResult) -> str:
    """Format a saved selection as a selector label.

    Parameters
    ----------
    saved : SavedLassoResult
        Saved selection.

    Returns
    -------
    str
        Label ``"<target> — <n_splits>×<inner_n_splits> plis —
        <n_alphas> α — <key>"``.
    """
    config = saved.config
    return (
        f"{saved.target_name} — {config.n_splits}×{config.inner_n_splits} plis — "
        f"{config.n_alphas} α — {config.key}"
    )


def _format_dataset(name: str | None, key: str) -> str:
    return key if name is None else f"{name} ({key})"


def render_dataset_info(
    dataset: Dataset | None,
    dataset_key: str | None,
    params: tuple[int, int, int] | None,
) -> None:
    """Show which dataset of the application is used, with its hash.

    Parameters
    ----------
    dataset : Dataset or None
        Dataset loaded in the application, or None.
    dataset_key : str or None
        Hash of `dataset`, or None when no dataset is loaded.
    params : tuple of (int, int, int) or None
        Minimum and maximum distance, in km, and segment width, in metres.
    """
    st.sidebar.subheader("Dataset")
    if dataset is None or dataset_key is None or params is None:
        st.sidebar.warning(
            "Aucun dataset chargé : ouvrez la page « Dataset / Features » pour "
            "charger ou lancer une sélection LASSO."
        )
        return
    min_distance, max_distance, segment_width = params
    st.sidebar.caption(
        f"{_format_dataset(dataset.name, dataset_key)} · "
        f"{min_distance}–{max_distance} km · segments de {segment_width} m · "
        f"{len(dataset.data)} lignes · {len(dataset.features)} variables "
        "(page « Dataset / Features »)"
    )


def render_load_sidebar(
    saved_results: list[SavedLassoResult],
) -> SavedLassoResult | None:
    """Render the selector of the saved selections of the dataset.

    Parameters
    ----------
    saved_results : list of SavedLassoResult
        Saved selections of the dataset to choose from.

    Returns
    -------
    SavedLassoResult or None
        Selected result, or None when nothing is saved.
    """
    if not saved_results:
        st.sidebar.info(
            "Aucun modèle sauvegardé pour ce dataset : lancez une sélection."
        )
        return None
    selected = st.sidebar.selectbox(
        "Modèle sauvegardé", saved_results, format_func=format_label, index=0
    )
    with st.sidebar.expander("Configuration"):
        st.json(selected.config.model_dump())
    return selected


def render_run_form(dataset: Dataset, dataset_key: str) -> SavedLassoResult | None:
    """Render the selection form and run it when submitted.

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application.
    dataset_key : str
        Hash of `dataset`.

    Returns
    -------
    SavedLassoResult or None
        Last selection run from this form on this dataset, or None.
    """
    with st.sidebar.form("lasso_run_form"):
        target_name = st.selectbox(
            "Cible", TARGETS, index=TARGETS.index(dataset.target_name)
        )
        n_splits = st.number_input(
            "Plis GroupKFold externes", min_value=2, value=DEFAULT_CONFIG.n_splits
        )
        inner_n_splits = st.number_input(
            "Plis GroupKFold internes (LassoCV)",
            min_value=2,
            value=DEFAULT_CONFIG.inner_n_splits,
        )
        n_alphas = st.number_input(
            "Nombre de α", min_value=1, value=DEFAULT_CONFIG.n_alphas
        )
        selection_threshold = st.number_input(
            "Seuil de sélection |coef|",
            min_value=0.0,
            value=DEFAULT_CONFIG.selection_threshold,
            format="%.1e",
        )
        max_iter = st.number_input(
            "Itérations max", min_value=1, value=DEFAULT_CONFIG.max_iter
        )
        seed = st.number_input("Graine", value=DEFAULT_CONFIG.seed, step=1)
        submitted = st.form_submit_button("Lancer la sélection")

    if submitted:
        try:
            config = LassoConfig(
                n_splits=int(n_splits),
                inner_n_splits=int(inner_n_splits),
                n_alphas=int(n_alphas),
                selection_threshold=float(selection_threshold),
                max_iter=int(max_iter),
                seed=int(seed),
            )
        except ValidationError as error:
            st.error(f"Configuration invalide : {error}")
        else:
            try:
                st.session_state[RUN_KEY] = run_selection(
                    dataset, dataset_key, target_name, config
                )
            # UI boundary: any selection failure is shown instead of crashing the page.
            except Exception as error:  # pylint: disable=broad-exception-caught
                logger.exception("Lasso selection failed")
                st.error(f"Impossible de lancer la sélection : {error}")
    last_run: SavedLassoResult | None = st.session_state[RUN_KEY]
    # a run on another dataset (other parameters or zones) does not apply here
    if last_run is not None and last_run.dataset_key != dataset_key:
        return None
    return last_run


# --------------------------------------------------------------------------- #
#                                  RÉSULTATS
# --------------------------------------------------------------------------- #
def render_summary(selection: pd.DataFrame) -> None:
    """Render the key metrics, the selection frequency and the folds.

    Parameters
    ----------
    selection : pandas.DataFrame
        One row per outer fold and feature, as ``ModelingLasso.result``.
    """
    folds = summarize_folds(selection)
    frequencies = selection_frequency(selection)
    n_folds = len(folds)
    rmse_se = folds["rmse"].std(ddof=0) / n_folds**0.5
    metrics = [
        ("RMSE CV", f"{folds['rmse'].mean():.4f} ± {rmse_se:.4f}"),
        ("Variables sélectionnées (moy.)", f"{folds['n_selected'].mean():.1f}"),
        (
            "Toujours sélectionnées",
            f"{int((frequencies['frequency'] == 1).sum())} / {len(frequencies)}",
        ),
        ("α médian", f"{folds['alpha'].median():.3g}"),
    ]
    for column, (label, value) in zip(st.columns(len(metrics)), metrics):
        column.metric(label, value)

    frequency_column, folds_column = st.columns([3, 2], gap="large")
    with frequency_column:
        st.subheader("Fréquence de sélection")
        st.dataframe(
            frequencies,
            width="stretch",
            column_config={
                "frequency": st.column_config.ProgressColumn(
                    "Fréquence", min_value=0.0, max_value=1.0, format="percent"
                ),
                "mean_abs_coef": st.column_config.NumberColumn(
                    "|coef| moyen", format="%.4f"
                ),
            },
        )
    with folds_column:
        st.subheader("Plis externes")
        st.dataframe(
            folds,
            width="stretch",
            column_config={
                "alpha": st.column_config.NumberColumn("α", format="%.3g"),
                "rmse": st.column_config.NumberColumn("RMSE", format="%.4f"),
                "n_selected": st.column_config.NumberColumn("Sélectionnées"),
            },
        )


def _render_chart(
    name: str,
    method: Callable[..., Any],
    key: str,
    display_height: int | None = None,
    **kwargs: Any,
) -> None:
    try:
        figure = method(**kwargs)
        if display_height is not None:
            figure.update_layout(height=display_height)
        st.plotly_chart(figure, width="stretch", key=key)
    # UI boundary: any plotting failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.warning(f"{name} : {error}")


def render_path(plot: PlotLasso) -> None:
    """Render the regularisation path, with the number of highlighted features.

    Parameters
    ----------
    plot : PlotLasso
        Plotting interface whose path is fitted.
    """
    st.subheader("Chemin de régularisation")
    path = plot.path
    n_features = len(path.coef_path.columns)
    n_active_1se = int((path.coefs_at(path.alpha_1se).abs() > path.threshold).sum())
    n_highlight = st.slider(
        "Variables en couleur (plus grands |coef| au α 1 erreur-type)",
        min_value=0,
        max_value=n_features,
        value=n_active_1se,
        key="lasso_n_highlight",
    )
    _render_chart(
        "Chemin de régularisation",
        plot.regularization_path,
        f"lasso-path-{plot.config.key}-{n_highlight}",
        display_height=PATH_CHART_HEIGHT,
        n_highlight=n_highlight,
    )


def render_activity_prediction(selected: SavedLassoResult, dataset: Dataset) -> None:
    """Render an activity with the out-of-fold prediction of the selection.

    Parameters
    ----------
    selected : SavedLassoResult
        Selection whose pipeline predicts the activity.
    dataset : Dataset
        Dataset loaded in the application. It is not modified.
    """
    st.subheader(f"Activité {PREDICTED_ACTIVITY_ID} : cible et prédiction")
    st.caption(
        "Prédiction hors-pli : l'activité est prédite par le pipeline "
        "(LASSO puis régression linéaire) ajusté sur les plis qui ne la "
        "contiennent pas."
    )
    target_dataset = dataset.with_target(selected.target_name)
    try:
        prediction = out_of_fold_prediction(
            dataset,
            selected.dataset_key,
            selected.target_name,
            selected.config.model_dump_json(),
        )
        activity = PlotActivities(
            target_dataset.get_activities(PREDICTED_ACTIVITY_ID, prediction)
        )
    # UI boundary: any prediction failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("Predicting activity %d failed", PREDICTED_ACTIVITY_ID)
        st.error(f"Impossible de prédire l'activité : {error}")
        return
    _render_chart(
        "Vitesse et altitude",
        activity.speed,
        f"lasso-activity-{selected.config.key}-{selected.target_name}",
        display_height=ACTIVITY_CHART_HEIGHT,
    )


# --------------------------------------------------------------------------- #
#                                    PAGE
# --------------------------------------------------------------------------- #
def render_results(selected: SavedLassoResult, dataset: Dataset) -> None:
    """Render the metrics, tables and regularisation path of a selection.

    Parameters
    ----------
    selected : SavedLassoResult
        Selection to show, computed on `dataset`.
    dataset : Dataset
        Dataset loaded in the application.
    """
    try:
        selection = load_selection(selected.result_path)
    # UI boundary: any loading failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("Loading %s failed", selected.result_path)
        st.error(f"Impossible de charger le modèle : {error}")
        return
    if selection.empty:
        st.warning("Ce modèle ne contient aucune sélection.")
        return

    st.caption(
        f"Modèle : {_format_dataset(selected.dataset_name, selected.dataset_key)}"
        f" — {format_label(selected)}"
    )
    render_summary(selection)
    try:
        plot = fit_plot(
            dataset,
            selected.dataset_key,
            selected.target_name,
            selected.config.model_dump_json(),
        )
    # UI boundary: any fitting failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("Fitting the Lasso path failed")
        st.error(f"Impossible de calculer le chemin de régularisation : {error}")
        return
    render_path(plot)
    render_activity_prediction(selected, dataset)


def main() -> None:
    """Render the Lasso feature selection page."""
    st.title("Sélection LASSO")
    if RUN_KEY not in st.session_state:
        st.session_state[RUN_KEY] = None

    dataset: Dataset | None = st.session_state.get(DATASET_KEY)
    dataset_key = None if dataset is None else dataset.key  # hashed once per run
    render_dataset_info(dataset, dataset_key, st.session_state.get(DATASET_PARAMS_KEY))
    if dataset is None or dataset_key is None:
        st.info("Chargez d'abord un dataset depuis la page « Dataset / Features ».")
        return

    mode = st.sidebar.radio("Mode", [MODE_LOAD, MODE_RUN])
    if mode == MODE_LOAD:
        selected = render_load_sidebar(st_list_saved_results(dataset_key))
    else:
        selected = render_run_form(dataset, dataset_key)

    if selected is None:
        st.info(
            "Choisissez un modèle sauvegardé ou lancez une sélection "
            "depuis la barre latérale."
        )
        return
    render_results(selected, dataset)


if __name__ == "__main__":
    main()

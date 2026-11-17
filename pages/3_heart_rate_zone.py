"""Streamlit page optimising and showing the heart-rate zones.

Two modes: load a saved optimisation, or run a new one on the dataset loaded
in the "Dataset / Features" page (shared through ``st.session_state``).
"""

# Streamlit orders pages by their numeric file-name prefix.
# pylint: disable=invalid-name
# `_render_chart` mirrors the helper of pages/1_activities.py: page scripts
# cannot import each other; a shared UI module would remove the duplicate.
# pylint: disable=duplicate-code

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import ValidationError

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.heart_rate import (
    HR_CEIL_BPM,
    HR_FLOOR_BPM,
    MIN_ZONES,
    REFERENCE_CUTS,
    HeartRateConfig,
    ModelingHeartRate,
    SavedHeartRateResult,
    cuts_to_zones,
    list_saved_results,
    select_n_zones_one_se,
    summarize_zone_counts,
)
from TrailPredictor.plot import PlotHeartRate
from TrailPredictor.plot.heart_rate import ZONE_COLORS
from TrailPredictor.speed_columns import TARGETS, Target

logger = logging.getLogger(__name__)

# Dataset loaded by the "Dataset / Features" page; keep in sync with
# pages/2_dataset_features.py.
DATASET_KEY = "app_dataset"
DATASET_PARAMS_KEY = "app_dataset_params"
# Optimised zones added to the dataset ({label: cuts}) and the zones chosen for
# it (None: typed cuts); keep in sync with pages/2_dataset_features.py.
ZONES_KEY = "app_heart_rate_zones"
ZONES_CHOICE_KEY = "app_heart_rate_zones_choice"
# Last optimisation run (or reloaded) from the form of this page.
RUN_KEY = "heart_rate_last_run"

MODE_LOAD = "Charger un modèle sauvegardé"
MODE_OPTIMIZE = "Nouvelle optimisation"
MAX_ZONE_COUNT = len(ZONE_COLORS)  # one colour per zone in the figures
DEFAULT_CONFIG = HeartRateConfig()


# --------------------------------------------------------------------------- #
#                                 CHARGEMENT
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)
def st_list_saved_results() -> list[SavedHeartRateResult]:
    """List the saved heart-rate optimisations (cached, refreshed every minute).

    Returns
    -------
    list of SavedHeartRateResult
        Saved results found in ``MODELS_DIR``.
    """
    return list_saved_results()


@st.cache_data
def load_trials(result_path: Path) -> pd.DataFrame:
    """Read the trials of a saved optimisation (cached).

    Parameters
    ----------
    result_path : pathlib.Path
        CSV file of the trials.

    Returns
    -------
    pandas.DataFrame
        One row per Optuna trial.
    """
    return pd.read_csv(result_path)


# The dataset is not hashed (slow): `dataset_params` identifies it in the cache.
@st.cache_data(
    show_spinner="Calcul du RMSE de référence...",
    hash_funcs={Dataset: lambda _dataset: None},
)
def reference_score(
    dataset: Dataset,
    dataset_params: tuple[int, int, int],
    target_name: Target,
    config_json: str,
) -> tuple[float, float]:
    """Cross-validate the reference zones on the shared dataset (cached).

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application; not hashed by the cache, which
        relies on `dataset_params` instead.
    dataset_params : tuple of (int, int, int)
        Minimum and maximum activity distance, in km, and segment width, in
        metres, identifying `dataset` in the cache.
    target_name : Target
        Target of the optimisation.
    config_json : str
        JSON dump of the `HeartRateConfig` (hashable cache key).

    Returns
    -------
    rmse : float
        Mean RMSE of the ``REFERENCE_CUTS`` zones over the folds.
    rmse_se : float
        Standard error of the mean RMSE.
    """
    del dataset_params  # cache key only
    config = HeartRateConfig.model_validate_json(config_json)
    model = ModelingHeartRate(dataset.with_target(target_name), config)
    return model.evaluate_cuts_zones(REFERENCE_CUTS)


def run_optimization(
    dataset: Dataset, target_name: Target, config: HeartRateConfig
) -> SavedHeartRateResult:
    """Run the zone optimisation with a progress bar, unless already saved.

    Parameters
    ----------
    dataset : Dataset
        Dataset loaded in the application. It is not modified.
    target_name : Target
        Target the RMSE is computed on.
    config : HeartRateConfig
        Search space and budget.

    Returns
    -------
    SavedHeartRateResult
        Saved optimisation of `config`.
    """
    model = ModelingHeartRate(dataset.with_target(target_name), config)
    if model.result is not None:
        st.info("Résultat déjà sauvegardé pour cette configuration : rechargé.")
    else:
        progress_bar = st.progress(0.0, text="Démarrage...")

        def update_progress(fraction: float, text: str) -> None:
            progress_bar.progress(min(max(fraction, 0.0), 1.0), text=text)

        try:
            model.optimize_zones(progress_cb=update_progress)
        finally:
            progress_bar.empty()
        st_list_saved_results.clear()
        st.success("Optimisation terminée et sauvegardée.")
    return SavedHeartRateResult(
        target_name,
        config,
        model.result_path,
        dataset_key=model.dataset.key,
        dataset_name=model.dataset.name,
    )


# --------------------------------------------------------------------------- #
#                                   SIDEBAR
# --------------------------------------------------------------------------- #
def format_label(saved: SavedHeartRateResult) -> str:
    """Format a saved optimisation as a selector label.

    Parameters
    ----------
    saved : SavedHeartRateResult
        Saved optimisation.

    Returns
    -------
    str
        Label ``"<dataset> — <target> — N=<zone counts> — <n_trials> essais —
        <key>"``, the dataset being its name and hash (or its hash alone).
    """
    config = saved.config
    zone_counts = ", ".join(str(n) for n in config.zone_counts)
    return (
        f"{_format_dataset(saved)} — {saved.target_name} — N={zone_counts} — "
        f"{config.n_trials} essais — {config.key}"
    )


def _format_dataset(saved: SavedHeartRateResult) -> str:
    if saved.dataset_name is None:
        return saved.dataset_key
    return f"{saved.dataset_name} ({saved.dataset_key})"


def render_dataset_info(params: tuple[int, int, int] | None, n_rows: int) -> None:
    """Show which dataset of the application is used.

    Parameters
    ----------
    params : tuple of (int, int, int) or None
        Minimum and maximum distance, in km, and segment width, in metres;
        None when no dataset is loaded.
    n_rows : int
        Number of segments of the dataset.
    """
    st.sidebar.subheader("Dataset")
    if params is None:
        st.sidebar.warning(
            "Aucun dataset chargé : ouvrez la page « Dataset / Features » pour "
            "lancer une optimisation ou calculer la référence."
        )
        return
    min_distance, max_distance, segment_width = params
    st.sidebar.caption(
        f"{min_distance}–{max_distance} km · segments de {segment_width} m · "
        f"{n_rows} lignes (page « Dataset / Features »)"
    )


def render_load_sidebar(
    saved_results: list[SavedHeartRateResult],
) -> SavedHeartRateResult | None:
    """Render the selector of the saved optimisations.

    Parameters
    ----------
    saved_results : list of SavedHeartRateResult
        Saved optimisations to choose from.

    Returns
    -------
    SavedHeartRateResult or None
        Selected optimisation, or None when nothing is saved.
    """
    if not saved_results:
        st.sidebar.info("Aucun modèle sauvegardé : lancez une optimisation.")
        return None
    selected = st.sidebar.selectbox(
        "Modèle sauvegardé", saved_results, format_func=format_label, index=0
    )
    with st.sidebar.expander("Configuration"):
        st.json(selected.config.model_dump())
    return selected


def render_optimize_form(dataset: Dataset | None) -> SavedHeartRateResult | None:
    """Render the optimisation form and run it when submitted.

    Parameters
    ----------
    dataset : Dataset or None
        Dataset loaded in the application; the form is disabled without it.

    Returns
    -------
    SavedHeartRateResult or None
        Last optimisation run from this form, or None if none was run.
    """
    with st.sidebar.form("heart_rate_optimize_form"):
        target_name = st.selectbox(
            "Cible",
            TARGETS,
            index=0 if dataset is None else TARGETS.index(dataset.target_name),
        )
        zone_counts = st.multiselect(
            "Nombres de zones N",
            list(range(MIN_ZONES, MAX_ZONE_COUNT + 1)),
            default=list(DEFAULT_CONFIG.zone_counts),
        )
        first_cut_range = st.slider(
            "Premier seuil (bpm)",
            HR_FLOOR_BPM + 1,
            HR_CEIL_BPM - 1,
            DEFAULT_CONFIG.first_cut_range,
        )
        gap_range = st.slider(
            "Écart entre seuils (bpm)", 1, 100, DEFAULT_CONFIG.gap_range
        )
        max_last_cut = st.number_input(
            "Dernier seuil maximal (bpm)",
            min_value=HR_FLOOR_BPM + 1,
            max_value=HR_CEIL_BPM - 1,
            value=DEFAULT_CONFIG.max_last_cut,
        )
        n_trials = st.number_input(
            "Essais complétés par N", min_value=1, value=DEFAULT_CONFIG.n_trials
        )
        max_trials_factor = st.number_input(
            "Facteur max d'essais (élagués inclus)",
            min_value=1,
            value=DEFAULT_CONFIG.max_trials_factor,
        )
        n_splits = st.number_input(
            "Plis GroupKFold", min_value=2, value=DEFAULT_CONFIG.n_splits
        )
        seed = st.number_input("Graine", value=DEFAULT_CONFIG.seed, step=1)
        submitted = st.form_submit_button(
            "Lancer l'optimisation", disabled=dataset is None
        )

    if submitted and dataset is not None:
        try:
            config = HeartRateConfig(
                zone_counts=tuple(sorted(zone_counts)),
                first_cut_range=first_cut_range,
                gap_range=gap_range,
                max_last_cut=int(max_last_cut),
                n_trials=int(n_trials),
                max_trials_factor=int(max_trials_factor),
                n_splits=int(n_splits),
                seed=int(seed),
            )
        except ValidationError as error:
            st.error(f"Configuration invalide : {error}")
            return st.session_state[RUN_KEY]
        try:
            st.session_state[RUN_KEY] = run_optimization(dataset, target_name, config)
        # UI boundary: any optimisation failure is shown instead of crashing the page.
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("Heart-rate zone optimisation failed")
            st.error(f"Impossible de lancer l'optimisation : {error}")
    return st.session_state[RUN_KEY]


# --------------------------------------------------------------------------- #
#                                  RÉSULTATS
# --------------------------------------------------------------------------- #
def _format_rmse(rmse: float, rmse_se: float) -> str:
    return f"{rmse:.4f} ± {rmse_se:.4f}"


def render_summary(
    trials: pd.DataFrame,
    plot: PlotHeartRate,
    reference: tuple[float, float] | None,
) -> list[int]:
    """Render the key metrics and the summary table of each zone count.

    Parameters
    ----------
    trials : pandas.DataFrame
        Every trial of the optimisation.
    plot : PlotHeartRate
        Plotting interface of the trials.
    reference : tuple of (float, float) or None
        RMSE and standard error of the reference zones, if computed.

    Returns
    -------
    list of int
        Cuts, in bpm, of the best trial of the number of zones chosen by the
        one-standard-error rule.
    """
    reference_rmse = None if reference is None else reference[0]
    summary = summarize_zone_counts(trials, plot.best_trials, reference_rmse)
    selection = select_n_zones_one_se(summary)
    best = summary.loc[selection.best_n_zones]
    chosen = summary.loc[selection.n_zones]

    metrics = [
        ("N retenu (1-SE)", str(selection.n_zones), None),
        ("Meilleur N", str(selection.best_n_zones), None),
        ("Meilleur RMSE", _format_rmse(best["rmse"], best["rmse_se"]), None),
        ("Seuil 1-SE", f"{selection.threshold:.4f}", None),
    ]
    if reference is not None:
        metrics.append(
            (
                "RMSE référence",
                _format_rmse(*reference),
                f"{chosen['rmse'] - reference[0]:+.4f} (N retenu)",
            )
        )
    for column, (label, value, delta) in zip(st.columns(len(metrics)), metrics):
        column.metric(label, value, delta=delta, delta_color="inverse")
    st.markdown(f"**Zones retenues :** {_format_zones(chosen['cuts'])}")

    table = summary.assign(
        **{
            "RMSE ± SE": summary["rmse"].map(lambda value: f"{value:.4f}")
            + " ± "
            + summary["rmse_se"].map(lambda value: f"{value:.4f}"),
            "retenu (1-SE)": summary.index == selection.n_zones,
        }
    )
    columns = ["trial_number", "cuts", "RMSE ± SE", "n_complete", "n_pruned"]
    if reference is not None:
        columns.append("delta_vs_reference")
    st.dataframe(
        table[[*columns, "retenu (1-SE)"]],
        width="stretch",
        column_config={
            "trial_number": st.column_config.NumberColumn("Essai"),
            "cuts": st.column_config.ListColumn("Seuils (bpm)"),
            "n_complete": st.column_config.NumberColumn("Complétés"),
            "n_pruned": st.column_config.NumberColumn("Élagués"),
            "delta_vs_reference": st.column_config.NumberColumn(
                "Δ référence", format="%+.4f"
            ),
        },
    )
    return [int(cut) for cut in chosen["cuts"]]


def _add_zones(label: str, cuts: list[int]) -> None:
    st.session_state[ZONES_KEY][label] = cuts
    st.session_state[ZONES_CHOICE_KEY] = label


def render_add_zones_button(selected: SavedHeartRateResult, cuts: list[int]) -> None:
    """Render the button adding the chosen zones to the dataset of the app.

    The zones are offered in the "Dataset / Features" page and chosen there
    for the dataset.

    Parameters
    ----------
    selected : SavedHeartRateResult
        Optimisation the zones come from.
    cuts : list of int
        Cuts of the zones, in bpm.
    """
    label = (
        f"N={len(cuts) + 1} · {selected.target_name} · "
        f"{_format_dataset(selected)} · {selected.config.key}"
    )
    st.button(
        "Ajouter ces zones dans le dataset",
        on_click=_add_zones,
        args=(label, cuts),
        key=f"heart-rate-add-zones-{label}",
    )
    if st.session_state[ZONES_CHOICE_KEY] == label:
        st.success(
            "Zones utilisées par le dataset de la page « Dataset / Features » "
            f"({label})."
        )


def _format_zones(cuts: list[int]) -> str:
    return " · ".join(
        f"{zone} {low}–{high}" for zone, (low, high) in cuts_to_zones(cuts).items()
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


# --------------------------------------------------------------------------- #
#                                    PAGE
# --------------------------------------------------------------------------- #
def render_results(
    selected: SavedHeartRateResult,
    dataset: Dataset | None,
    dataset_params: tuple[int, int, int] | None,
) -> None:
    """Render the metrics, table and charts of an optimisation.

    Parameters
    ----------
    selected : SavedHeartRateResult
        Optimisation to show.
    dataset : Dataset or None
        Dataset loaded in the application, used for the reference RMSE.
    dataset_params : tuple of (int, int, int) or None
        Parameters identifying `dataset`.
    """
    try:
        trials = load_trials(selected.result_path)
        reference = (
            None
            if dataset is None or dataset_params is None
            else reference_score(
                dataset,
                dataset_params,
                selected.target_name,
                selected.config.model_dump_json(),
            )
        )
        plot = PlotHeartRate(
            trials,
            selected.target_name,
            reference_rmse=None if reference is None else reference[0],
        )
    # UI boundary: any loading failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("Loading %s failed", selected.result_path)
        st.error(f"Impossible de charger le modèle : {error}")
        return
    if plot.completed_trials.empty:
        st.warning("Ce modèle ne contient aucun essai complété.")
        return

    st.caption(f"Modèle : {format_label(selected)}")
    cuts = render_summary(trials, plot, reference)
    render_add_zones_button(selected, cuts)
    _render_chart(
        "Meilleur essai par N",
        plot.best_zones_per_n,
        "heart-rate-best-zones",
        display_height=700,
    )

    zone_counts = np.sort(plot.completed_trials["n_zones"].unique()).tolist()
    n_zones = st.radio("Nombre de zones N", zone_counts, horizontal=True)
    _render_chart(
        f"Essais pour N={n_zones}",
        plot.trials_zones,
        f"heart-rate-trials-{n_zones}",
        display_height=700,
        n_zones=n_zones,
    )


def main() -> None:
    """Render the heart-rate zone page."""
    st.title("Zones cardiaques")
    if RUN_KEY not in st.session_state:
        st.session_state[RUN_KEY] = None
    if ZONES_KEY not in st.session_state:
        st.session_state[ZONES_KEY] = {}
    if ZONES_CHOICE_KEY not in st.session_state:
        st.session_state[ZONES_CHOICE_KEY] = None

    dataset: Dataset | None = st.session_state.get(DATASET_KEY)
    dataset_params = st.session_state.get(DATASET_PARAMS_KEY)
    render_dataset_info(
        None if dataset is None else dataset_params,
        0 if dataset is None else len(dataset.data),
    )

    mode = st.sidebar.radio("Mode", [MODE_LOAD, MODE_OPTIMIZE])
    if mode == MODE_LOAD:
        selected = render_load_sidebar(st_list_saved_results())
    else:
        selected = render_optimize_form(dataset)

    if selected is None:
        st.info(
            "Choisissez un modèle sauvegardé ou lancez une optimisation "
            "depuis la barre latérale."
        )
        return
    render_results(selected, dataset, dataset_params)


if __name__ == "__main__":
    main()

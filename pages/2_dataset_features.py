"""Streamlit page exploring the dataset and its features.

Layout: a selection panel on the left (filters, presets, clickable table with
correlation and missing values) and the charts on the right, always visible.
Set ``SELECTION_IN_SIDEBAR = True`` to put the panel in the sidebar instead.
"""

# Streamlit orders pages by their numeric file-name prefix.
# pylint: disable=invalid-name

from __future__ import annotations

import hashlib
import logging
from typing import Any

import pandas as pd
import streamlit as st

from TrailPredictor.dataset import Dataset
from TrailPredictor.feature_selection import (
    CORRELATION_COLUMNS,
    FeatureGroup,
    Preset,
    apply_filters,
    apply_preset,
    merge_edits,
)
from TrailPredictor.feature_selection import (
    correlation_table as compute_correlation_table,
)
from TrailPredictor.modeling.heart_rate import (
    REFERENCE_CUTS,
    SavedHeartRateResult,
    cuts_to_zones,
    list_saved_results,
    parse_cuts,
    saved_one_se_cuts,
)
from TrailPredictor.plot import PlotDataset
from TrailPredictor.speed_columns import TARGETS, Target

logger = logging.getLogger(__name__)

SELECTION_IN_SIDEBAR = False

# Dataset loaded on this page, reused by the other pages (e.g. heart-rate
# zones); keep in sync with pages/3_heart_rate_zone.py.
DATASET_KEY = "app_dataset"
DATASET_PARAMS_KEY = "app_dataset_params"
# Zones chosen for the dataset: CSV path of a saved heart-rate optimisation, or
# None for the typed cuts; also set by the button of pages/3_heart_rate_zone.py
# (keep in sync).
ZONES_CHOICE_KEY = "app_heart_rate_zones_choice"
TYPED_ZONES = "Saisie libre"

# Streamlit récent : remplacer par {"width": "stretch"} si l'avertissement de
# dépréciation de `use_container_width` gêne.
STRETCH: dict[str, Any] = {"use_container_width": True}

SELECTION_KEY = "dataset_selected_features"  # liste ordonnée (état, pas un widget)
REVISION_KEY = (
    "dataset_selection_revision"  # incrémenté si la sélection change "de l'extérieur"
)
FILTER_KEYS = {
    "dataset_filter_search": "",
    "dataset_filter_groups": [],
    "dataset_filter_min_abs": 0.0,
    "dataset_filter_max_missing": 100,
}

GROUP_LABELS = {
    FeatureGroup.PAST: "Passé (back)",
    FeatureGroup.UPCOMING: "À venir (front)",
    FeatureGroup.CUMULATIVE: "Cumuls",
    FeatureGroup.RATIO: "Ratios",
    FeatureGroup.OTHER: "Autres",
}
PRESET_LABELS = {
    Preset.TOP_5: "Top 5 (|ρ|)",
    Preset.TOP_10: "Top 10 (|ρ|)",
    Preset.TOP_20: "Top 20 (|ρ|)",
    Preset.POSITIVE: "Corrélées positivement",
    Preset.NEGATIVE: "Corrélées négativement",
    Preset.ALL: "Toutes les features filtrées",
}


# --------------------------------------------------------------------------- #
#                                 CHARGEMENT
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset(min_distance: int, max_distance: int, segment_width: int) -> Dataset:
    """Load the dataset with the sidebar parameters (cached).

    Parameters
    ----------
    min_distance : int
        Minimum activity distance in km.
    max_distance : int
        Maximum activity distance in km.
    segment_width : int
        Segment length in metres.

    Returns
    -------
    Dataset
        The loaded dataset.

    See Also
    --------
    TrailPredictor.dataset.Dataset.load : Underlying loader.
    """
    return Dataset.load(
        min_distance=min_distance,
        max_distance=max_distance,
        segment_width=segment_width,
    )


@st.cache_data
def correlation_table(
    min_distance: int, max_distance: int, segment_width: int, target_name: Target
) -> pd.DataFrame:
    """Compute the Spearman correlation of each feature with the target (cached).

    Parameters
    ----------
    min_distance : int
        Minimum activity distance in km.
    max_distance : int
        Maximum activity distance in km.
    segment_width : int
        Segment length in metres.
    target_name : Target
        Target the features are correlated with.

    Returns
    -------
    pandas.DataFrame
        Table of `TrailPredictor.feature_selection.correlation_table`.
    """
    dataset = load_dataset(min_distance, max_distance, segment_width)
    return compute_correlation_table(dataset.with_target(target_name))


# --------------------------------------------------------------------------- #
#                              PANNEAU DE SÉLECTION
# --------------------------------------------------------------------------- #
def _set_selection(names: list[str]) -> None:
    """Set the selection from outside the table and reset the table."""
    st.session_state[SELECTION_KEY] = list(dict.fromkeys(names))
    st.session_state[REVISION_KEY] += 1
    st.rerun()


def _reset_filters() -> None:
    for key, default in FILTER_KEYS.items():
        st.session_state[key] = default


def _table_key(pool: pd.DataFrame) -> str:
    """Build the table key, which changes with the shown rows or external selection.

    It does NOT change when the user ticks a box, otherwise successive ticks
    would be lost.
    """
    digest = hashlib.md5("|".join(pool["name"]).encode()).hexdigest()[:8]
    return f"dataset_table_{st.session_state[REVISION_KEY]}_{digest}"


def selection_table(pool: pd.DataFrame, selected: list[str]) -> list[str]:
    """Render the clickable feature table, with a "✓" box per row.

    Parameters
    ----------
    pool : pandas.DataFrame
        Features to show, as returned by `apply_filters`.
    selected : list of str
        Currently selected features.

    Returns
    -------
    list of str
        Selection after the user's ticks.
    """
    frame = pool[CORRELATION_COLUMNS].assign(
        group=pool["group"].map(lambda group: GROUP_LABELS[FeatureGroup(group)])
    )
    frame.insert(0, "✓", frame["name"].isin(selected))
    edited = st.data_editor(
        frame,
        key=_table_key(pool),
        hide_index=True,
        height=min(38 + 35 * len(frame), 460),
        disabled=CORRELATION_COLUMNS,
        column_config={
            "✓": st.column_config.CheckboxColumn("✓", width="small"),
            "name": st.column_config.TextColumn("Feature", width="medium"),
            "spearman": st.column_config.NumberColumn(
                "ρ",
                format="%+.2f",
                width="small",
                help="Corrélation de Spearman avec la vitesse (+ : plus rapide).",
            ),
            "missing_pct": st.column_config.NumberColumn(
                "NaN %", format="%.1f", width="small"
            ),
            "group": st.column_config.TextColumn("Groupe", width="small"),
        },
        **STRETCH,
    )
    flags = dict(zip(edited["name"], edited["✓"].astype(bool)))
    return merge_edits(selected, flags)


def render_selection_panel(plot: PlotDataset, correlations: pd.DataFrame) -> None:
    """Render the filters, presets and clickable feature table.

    Parameters
    ----------
    plot : PlotDataset
        Dataset being explored.
    correlations : pandas.DataFrame
        Table returned by `correlation_table`.
    """
    ranked = correlations["name"].tolist()  # numériques, triées par |rho|
    if not ranked:
        st.warning("Aucune feature numérique disponible dans ce dataset.")
        return

    st.session_state.setdefault(SELECTION_KEY, ranked[:1])
    st.session_state.setdefault(REVISION_KEY, 0)
    # Ne garder que les features qui existent encore (changement de paramètres)
    st.session_state[SELECTION_KEY] = [
        name for name in st.session_state[SELECTION_KEY] if name in ranked
    ]

    for key, default in FILTER_KEYS.items():
        st.session_state.setdefault(key, default)

    st.subheader("Features")

    # --- Filtres -----------------------------------------------------------
    st.text_input("Rechercher", key="dataset_filter_search")
    st.multiselect(
        "Groupes",
        options=sorted(correlations["group"].unique()),
        format_func=lambda group: GROUP_LABELS[FeatureGroup(group)],
        key="dataset_filter_groups",
        placeholder="Tous les groupes",
    )
    left, right = st.columns(2)
    left.slider("|ρ| minimum", 0.0, 1.0, step=0.01, key="dataset_filter_min_abs")
    right.slider("Manquantes max (%)", 0, 100, key="dataset_filter_max_missing")
    st.button("Réinitialiser les filtres", on_click=_reset_filters)

    pool = apply_filters(
        correlations,
        st.session_state["dataset_filter_search"],
        st.session_state["dataset_filter_groups"],
        st.session_state["dataset_filter_min_abs"],
        st.session_state["dataset_filter_max_missing"],
    )

    # --- Presets (ils piochent dans les features filtrées) -----------------
    preset = st.selectbox(
        "Preset", list(Preset), format_func=PRESET_LABELS.get, key="dataset_preset"
    )
    replace_col, add_col, clear_col = st.columns(3)
    if replace_col.button("Remplacer", key="dataset_preset_replace"):
        _set_selection(apply_preset(preset, pool))
    if add_col.button("Ajouter", key="dataset_preset_add"):
        _set_selection(st.session_state[SELECTION_KEY] + apply_preset(preset, pool))
    if clear_col.button("Vider", key="dataset_preset_clear"):
        _set_selection([])

    # --- Tableau cliquable --------------------------------------------------
    st.caption(
        f"{len(pool)} feature(s) affichée(s) sur {len(correlations)} · "
        "cochez pour afficher le graphique ; clic sur un en-tête pour trier."
    )
    if pool.empty:
        st.info("Aucune feature ne correspond aux filtres.")
    else:
        st.session_state[SELECTION_KEY] = selection_table(
            pool, st.session_state[SELECTION_KEY]
        )

    selected = st.session_state[SELECTION_KEY]
    with st.expander(f"Sélection : {len(selected)} feature(s)"):
        st.write(" · ".join(selected) if selected else "Aucune feature sélectionnée.")

    non_numeric = [name for name in plot.features if name not in ranked]
    if non_numeric:
        with st.expander(f"{len(non_numeric)} feature(s) non numérique(s) ignorée(s)"):
            st.write(
                "Un nuage de points nécessite une abscisse numérique : "
                + ", ".join(non_numeric)
            )


# --------------------------------------------------------------------------- #
#                                  GRAPHIQUES
# --------------------------------------------------------------------------- #
def _render_chart(
    plot: PlotDataset, title: str, key: str, method: Any, **kwargs: Any
) -> None:
    try:
        figure = plot.merge_figure(method, title=title, **kwargs)
        figure.update_layout(height=700)
        st.plotly_chart(figure, key=key, **STRETCH)
    # UI boundary: any plotting failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.warning(f"{title} : {error}")


def render_charts(plot: PlotDataset) -> None:
    """Render the target histogram, then the charts of the selected features.

    Parameters
    ----------
    plot : PlotDataset
        Dataset being explored.
    """
    selected = st.session_state.get(SELECTION_KEY, [])
    st.subheader(f"Graphiques ({len(selected)} feature(s))")
    if len(selected) > 12:
        st.warning("Un grand nombre de graphiques peut ralentir l'affichage.")
    columns_per_row = st.radio(
        "Colonnes par ligne",
        [1, 2],
        index=0,
        horizontal=True,
        key="dataset_chart_columns",
    )

    charts = [
        (
            "Distribution de la cible",
            "dataset-target-histogram",
            plot.target_histogram,
            {},
        )
    ]
    charts.extend(
        (
            f"Cible vs {feature}",
            f"dataset-feature-{feature}",
            plot.target_vs_feat,
            {"features": feature},
        )
        for feature in selected
    )
    for index in range(0, len(charts), columns_per_row):
        chart_row = charts[index : index + columns_per_row]
        columns = st.columns(columns_per_row)
        for column, (title, key, method, kwargs) in zip(columns, chart_row):
            with column:
                _render_chart(plot, title, key, method, **kwargs)


# --------------------------------------------------------------------------- #
#                                ZONES CARDIAQUES
# --------------------------------------------------------------------------- #
@st.cache_data(ttl=60)
def saved_zone_options() -> dict[str, tuple[str, list[int]]]:
    """List the zones of the saved heart-rate optimisations (cached one minute).

    Results whose trials cannot be read, or hold no completed trial, are
    skipped and logged.

    Returns
    -------
    dict[str, tuple[str, list[int]]]
        CSV path of each saved optimisation mapped to its selector label and
        the cuts, in bpm, chosen by the one-standard-error rule.
    """
    saved_results = {saved.result_path: saved for saved in list_saved_results()}
    return {
        str(path): (_zones_label(saved_results[path], cuts), cuts)
        for path, cuts in saved_one_se_cuts(saved_results.values()).items()
    }


def _zones_label(saved: SavedHeartRateResult, cuts: list[int]) -> str:
    return (
        f"N={len(cuts) + 1} · {saved.target_name} · {saved.dataset_label} · "
        f"{saved.config.key}"
    )


def render_zones_sidebar() -> list[int]:
    """Render the choice of the heart-rate zones of the dataset.

    The cuts are typed (default ``REFERENCE_CUTS``) or taken from a saved
    heart-rate optimisation (cuts of the one-standard-error rule).

    Returns
    -------
    list of int
        Chosen cuts, in bpm.

    Raises
    ------
    ValueError
        If the typed cuts are invalid.
    """
    choice = st.session_state[ZONES_CHOICE_KEY]
    saved_zones = saved_zone_options()
    if choice is not None and choice not in saved_zones:
        saved_zone_options.clear()  # e.g. optimisation just run on the other page
        saved_zones = saved_zone_options()
    options = [TYPED_ZONES, *saved_zones]
    choice = st.sidebar.selectbox(
        "Zones cardiaques",
        options,
        index=options.index(choice) if choice in saved_zones else 0,
        format_func=lambda key: key if key == TYPED_ZONES else saved_zones[key][0],
        help="Saisie libre, ou zones (règle 1-SE) d'une optimisation sauvegardée "
        "depuis la page « Zones cardiaques ».",
    )
    st.session_state[ZONES_CHOICE_KEY] = None if choice == TYPED_ZONES else choice
    if choice == TYPED_ZONES:
        typed = st.sidebar.text_input(
            "Seuils des zones (bpm)",
            value=", ".join(map(str, REFERENCE_CUTS)),
            key="dataset_zone_cuts",
            help="Seuils croissants séparés par des virgules : N seuils → N+1 zones.",
        )
        return parse_cuts(typed)
    cuts = saved_zones[choice][1]
    st.sidebar.caption("Seuils : " + ", ".join(map(str, cuts)) + " bpm")
    return cuts


# --------------------------------------------------------------------------- #
#                                    PAGE
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the dataset exploration page."""
    st.title("Dataset / Features")
    if ZONES_CHOICE_KEY not in st.session_state:
        st.session_state[ZONES_CHOICE_KEY] = None
    st.sidebar.subheader("Paramètres du dataset")
    min_distance = st.sidebar.number_input(
        "Distance minimale (km)", min_value=1, max_value=200, value=40, step=1
    )
    max_distance = st.sidebar.number_input(
        "Distance maximale (km)", min_value=2, max_value=300, value=60, step=1
    )
    segment_width = st.sidebar.number_input(
        "Taille des segments (m)", min_value=100, max_value=5000, value=1000, step=100
    )
    target_name = st.sidebar.selectbox("Cible", TARGETS, index=0)
    try:
        cuts = render_zones_sidebar()
    except ValueError as error:
        st.sidebar.error(f"Seuils invalides : {error}")
        return
    params = (int(min_distance), int(max_distance), int(segment_width))
    try:
        dataset = (
            load_dataset(*params)
            .with_target(target_name)
            .with_heart_rate_zones(cuts_to_zones(cuts))
        )
        plot = dataset.plot
        correlations = correlation_table(*params, target_name)
    # UI boundary: any loading failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.error(f"Impossible de charger le dataset : {error}")
        return

    st.session_state[DATASET_KEY] = dataset
    st.session_state[DATASET_PARAMS_KEY] = params

    st.sidebar.caption(f"{len(dataset.data)} lignes")

    # Le panneau de sélection est rendu AVANT les graphiques : ils lisent la
    # sélection à jour dans le même run.
    if SELECTION_IN_SIDEBAR:
        with st.sidebar:
            st.divider()
            render_selection_panel(plot, correlations)
        render_charts(plot)
    else:
        panel, charts = st.columns([2, 3], gap="large")
        with panel:
            render_selection_panel(plot, correlations)
        with charts:
            render_charts(plot)


if __name__ == "__main__":
    main()

"""Streamlit page exploring the dataset and its features.

Layout: a selection panel on the left (filters, presets, clickable table with
correlation and missing values) and the charts on the right, always visible.
Set ``SELECTION_IN_SIDEBAR = True`` to put the panel in the sidebar instead.
"""

# Streamlit orders pages by their numeric file-name prefix.
# pylint: disable=invalid-name

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from TrailPredictor.dataset import Dataset
from TrailPredictor.modeling.heart_rate import REFERENCE_CUTS, cuts_to_zones, parse_cuts
from TrailPredictor.plot import PlotDataset
from TrailPredictor.speed_columns import TARGETS, Target

SELECTION_IN_SIDEBAR = False

# Dataset loaded on this page, reused by the other pages (e.g. heart-rate
# zones); keep in sync with pages/3_heart_rate_zone.py.
DATASET_KEY = "app_dataset"
DATASET_PARAMS_KEY = "app_dataset_params"
# Optimised zones added from pages/3_heart_rate_zone.py ({label: cuts}) and the
# zones chosen for the dataset (None: typed cuts); keep in sync with that page.
ZONES_KEY = "app_heart_rate_zones"
ZONES_CHOICE_KEY = "app_heart_rate_zones_choice"
TYPED_ZONES = "Saisie libre"

# Dataset loaded on this page, reused by the other pages (e.g. heart-rate
# zones); keep in sync with pages/3_heart_rate_zone.py.
DATASET_KEY = "app_dataset"
DATASET_PARAMS_KEY = "app_dataset_params"

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

# La corrélation de Spearman est calculée avec la cible choisie, puis son
# signe est inversé pour les cibles de type allure (« pace ») afin de toujours
# l'exprimer par rapport à la vitesse (+ : la feature va avec une allure plus
# rapide).
PACE_SIGN = -1.0
SPEED_SIGN = 1.0

TABLE_COLUMNS = ["Nom", "Spearman", "Manquantes (%)", "Groupe"]


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


def _feature_group(name: str) -> str:
    if "_back_" in name:
        return "Passé (back)"
    if "_front_" in name:
        return "À venir (front)"
    if name.endswith("_done") or name.endswith("_remaining") or name.endswith("_total"):
        return "Cumuls"
    if name.endswith("_ratio"):
        return "Ratios"
    return "Autres"


@st.cache_data
def correlation_table(
    min_distance: int, max_distance: int, segment_width: int, target_name: Target
) -> pd.DataFrame:
    """Compute the Spearman correlation of each numeric feature with the target.

    The sign is flipped for pace-like targets, so a positive correlation always
    means a faster running.

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
        Columns ``Nom``, ``Spearman``, ``Manquantes (%)`` and ``Groupe``,
        sorted by decreasing absolute correlation (NaN last).
    """
    dataset = load_dataset(min_distance, max_distance, segment_width)
    plot = dataset.with_target(target_name).plot

    target = pd.to_numeric(plot.target, errors="coerce")
    target = target.replace([np.inf, -np.inf], np.nan)

    data = dataset.data
    names = [
        name for name in plot.features if pd.api.types.is_numeric_dtype(data[name])
    ]
    frame = (
        data[names]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    frame["__target__"] = target
    sign = PACE_SIGN if target_name.endswith("pace") else SPEED_SIGN
    rho = frame.corr(method="spearman")["__target__"].drop("__target__") * sign

    table = pd.DataFrame(
        {
            "Nom": rho.index,
            "Spearman": rho.to_numpy(),
            "Manquantes (%)": (data[names].isna().mean() * 100).round(2).to_numpy(),
            "Groupe": [_feature_group(name) for name in rho.index],
        }
    )
    order = table["Spearman"].abs().sort_values(ascending=False, na_position="last")
    return table.loc[order.index].reset_index(drop=True)


# --------------------------------------------------------------------------- #
#                           FILTRES ET PRESETS (purs)
# --------------------------------------------------------------------------- #
def apply_filters(
    correlations: pd.DataFrame,
    search: str,
    groups: list[str],
    min_abs: float,
    max_missing: float,
) -> pd.DataFrame:
    """Keep the features that pass every filter, in the same order.

    Parameters
    ----------
    correlations : pandas.DataFrame
        Table returned by `correlation_table`.
    search : str
        Case-insensitive substring of the feature name.
    groups : list of str
        Allowed feature groups; all groups when empty.
    min_abs : float
        Minimum absolute Spearman correlation; no filter when 0.
    max_missing : float
        Maximum percentage of missing values.

    Returns
    -------
    pandas.DataFrame
        Filtered table with a fresh index.
    """
    mask = correlations["Nom"].str.contains(search, case=False, na=False, regex=False)
    if groups:
        mask &= correlations["Groupe"].isin(groups)
    mask &= correlations["Manquantes (%)"] <= max_missing
    if min_abs > 0:
        mask &= correlations["Spearman"].abs() >= min_abs
    return correlations[mask].reset_index(drop=True)


def _ranked(pool: pd.DataFrame) -> pd.DataFrame:
    return pool.dropna(subset=["Spearman"])


PRESETS: dict[str, Callable[[pd.DataFrame], list[str]]] = {
    "Top 5 (|ρ|)": lambda pool: _ranked(pool).head(5)["Nom"].tolist(),
    "Top 10 (|ρ|)": lambda pool: _ranked(pool).head(10)["Nom"].tolist(),
    "Top 20 (|ρ|)": lambda pool: _ranked(pool).head(20)["Nom"].tolist(),
    "Corrélées positivement": lambda pool: (
        _ranked(pool).loc[lambda d: d["Spearman"] > 0, "Nom"].tolist()
    ),
    "Corrélées négativement": lambda pool: (
        _ranked(pool).loc[lambda d: d["Spearman"] < 0, "Nom"].tolist()
    ),
    "Toutes les features filtrées": lambda pool: pool["Nom"].tolist(),
}


def merge_edits(current: list[str], flags: dict[str, bool]) -> list[str]:
    """Apply the ticked boxes of the table to the selection.

    Parameters
    ----------
    current : list of str
        Currently selected features, in selection order.
    flags : dict[str, bool]
        Tick state of each feature shown in the table.

    Returns
    -------
    list of str
        Updated selection. Features missing from `flags` (hidden by a
        filter) are unchanged, the order is kept and new features are
        appended.
    """
    updated = [name for name in current if flags.get(name, True)]
    updated += [name for name, on in flags.items() if on and name not in updated]
    return updated


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
    digest = hashlib.md5("|".join(pool["Nom"]).encode()).hexdigest()[:8]
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
    frame = pool[TABLE_COLUMNS].copy()
    frame.insert(0, "✓", frame["Nom"].isin(selected))
    edited = st.data_editor(
        frame,
        key=_table_key(pool),
        hide_index=True,
        height=min(38 + 35 * len(frame), 460),
        disabled=TABLE_COLUMNS,
        column_config={
            "✓": st.column_config.CheckboxColumn("✓", width="small"),
            "Nom": st.column_config.TextColumn("Feature", width="medium"),
            "Spearman": st.column_config.NumberColumn(
                "ρ",
                format="%+.2f",
                width="small",
                help="Corrélation de Spearman avec la vitesse (+ : plus rapide).",
            ),
            "Manquantes (%)": st.column_config.NumberColumn(
                "NaN %", format="%.1f", width="small"
            ),
            "Groupe": st.column_config.TextColumn("Groupe", width="small"),
        },
        **STRETCH,
    )
    flags = dict(zip(edited["Nom"], edited["✓"].astype(bool)))
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
    ranked = correlations["Nom"].tolist()  # numériques, triées par |rho|
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
        options=sorted(correlations["Groupe"].unique()),
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
    preset = st.selectbox("Preset", list(PRESETS), key="dataset_preset")
    replace_col, add_col, clear_col = st.columns(3)
    if replace_col.button("Remplacer", key="dataset_preset_replace"):
        _set_selection(PRESETS[preset](pool))
    if add_col.button("Ajouter", key="dataset_preset_add"):
        _set_selection(st.session_state[SELECTION_KEY] + PRESETS[preset](pool))
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
def render_zones_sidebar() -> list[int]:
    """Render the choice of the heart-rate zones of the dataset.

    The cuts are typed (default ``REFERENCE_CUTS``) or picked among the
    optimised zones added from the "Zones cardiaques" page.

    Returns
    -------
    list of int
        Chosen cuts, in bpm.

    Raises
    ------
    ValueError
        If the typed cuts are invalid.
    """
    saved_zones: dict[str, list[int]] = st.session_state[ZONES_KEY]
    options = [TYPED_ZONES, *saved_zones]
    choice = st.session_state[ZONES_CHOICE_KEY]
    choice = st.sidebar.selectbox(
        "Zones cardiaques",
        options,
        index=options.index(choice) if choice in saved_zones else 0,
        help="Zones optimisées : ajoutez-les depuis la page « Zones cardiaques ».",
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
    cuts = saved_zones[choice]
    st.sidebar.caption("Seuils : " + ", ".join(map(str, cuts)) + " bpm")
    return cuts


# --------------------------------------------------------------------------- #
#                                    PAGE
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the dataset exploration page."""
    st.title("Dataset / Features")
    if ZONES_KEY not in st.session_state:
        st.session_state[ZONES_KEY] = {}
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

    st.sidebar.write(
        f"Cible courante : `{plot.target_name}` · {len(dataset.data)} lignes"
    )

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

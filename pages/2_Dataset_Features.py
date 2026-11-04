"""Page Streamlit d'exploration du dataset et de ses features.

Disposition : un panneau de sélection à gauche (filtres, presets, tableau
cliquable avec corrélation et valeurs manquantes) et les graphiques à droite,
toujours visibles. Mettre SELECTION_IN_SIDEBAR = True pour placer le panneau
dans la sidebar à la place.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from TrailPredictor.dataset import Dataset
from TrailPredictor.plot import PlotDataset


SELECTION_IN_SIDEBAR = False

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

# Spearman est invariant par transformation monotone : le classement des
# features est identique pour pace, log_pace, speed et log_speed, seul le signe
# change. On calcule sur "pace" et on inverse le signe pour l'exprimer par
# rapport à la vitesse (+ : la feature va avec une allure plus rapide).
SPEED_SIGN = -1.0

TABLE_COLUMNS = ["Nom", "Spearman", "Manquantes (%)", "Groupe"]


# --------------------------------------------------------------------------- #
#                                 CHARGEMENT
# --------------------------------------------------------------------------- #
@st.cache_data
def load_dataset(min_distance: int, max_distance: int, segment_width: int) -> Dataset:
    """Charge le dataset via Dataset.load avec les paramètres de la sidebar."""
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
    min_distance: int, max_distance: int, segment_width: int
) -> pd.DataFrame:
    """Corrélation de Spearman de chaque feature numérique avec la cible.

    Retourne un tableau trié par |corrélation| décroissante (NaN en dernier),
    avec les colonnes : Nom, Spearman, Manquantes (%), Groupe.
    """
    dataset = load_dataset(min_distance, max_distance, segment_width)
    plot = dataset.plot
    plot.change_target("pace")

    target = pd.to_numeric(plot.target, errors="coerce")
    target = target.replace([np.inf, -np.inf], np.nan)

    names = [
        name
        for name in plot.features
        if pd.api.types.is_numeric_dtype(dataset._data[name])
    ]
    frame = (
        dataset._data[names]
        .apply(pd.to_numeric, errors="coerce")
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
    )
    frame["__target__"] = target
    rho = frame.corr(method="spearman")["__target__"].drop("__target__") * SPEED_SIGN

    table = pd.DataFrame(
        {
            "Nom": rho.index,
            "Spearman": rho.to_numpy(),
            "Manquantes (%)": (dataset._data[names].isna().mean() * 100)
            .round(2)
            .to_numpy(),
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
    """Réduit le tableau aux features qui passent tous les filtres (ordre conservé)."""
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
    """Applique les cases cochées du tableau à la sélection.

    Les features absentes du tableau (masquées par un filtre) ne changent pas ;
    l'ordre de sélection est conservé et les nouvelles features sont ajoutées à la fin.
    """
    updated = [name for name in current if flags.get(name, True)]
    updated += [name for name, on in flags.items() if on and name not in updated]
    return updated


# --------------------------------------------------------------------------- #
#                              PANNEAU DE SÉLECTION
# --------------------------------------------------------------------------- #
def _set_selection(names: list[str]) -> None:
    """Change la sélection depuis l'extérieur du tableau et réinitialise celui-ci."""
    st.session_state[SELECTION_KEY] = list(dict.fromkeys(names))
    st.session_state[REVISION_KEY] += 1
    st.rerun()


def _reset_filters() -> None:
    for key, default in FILTER_KEYS.items():
        st.session_state[key] = default


def _table_key(pool: pd.DataFrame) -> str:
    """Clé du tableau : change si le contenu affiché ou la sélection externe change.

    Elle ne change PAS quand l'utilisateur coche une case, sinon les cases
    cochées successivement seraient perdues.
    """
    digest = hashlib.md5("|".join(pool["Nom"]).encode()).hexdigest()[:8]
    return f"dataset_table_{st.session_state[REVISION_KEY]}_{digest}"


def selection_table(pool: pd.DataFrame, selected: list[str]) -> list[str]:
    """Tableau cliquable (case « ✓ » par ligne) ; retourne la nouvelle sélection."""
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
    """Filtres, presets et tableau cliquable."""
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
    except Exception as error:
        st.warning(f"{title} : {error}")


def render_charts(plot: PlotDataset) -> None:
    """Affiche l'histogramme puis les graphiques des features sélectionnées."""
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
#                                    PAGE
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("Dataset / Features")
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
    params = (int(min_distance), int(max_distance), int(segment_width))
    try:
        dataset = load_dataset(*params)
        plot = dataset.plot
        correlations = correlation_table(*params)
    except Exception as error:
        st.error(f"Impossible de charger le dataset : {error}")
        return

    st.sidebar.write(f"Cible courante : `{plot._target}` · {len(dataset._data)} lignes")

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

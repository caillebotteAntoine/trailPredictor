"""Streamlit page analysing a single running activity."""

# Streamlit orders pages by their numeric file-name prefix.
# pylint: disable=invalid-name

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import streamlit as st

from TrailPredictor.activities import RunningActivities
from TrailPredictor.activity_metrics import compute_activity_metrics
from TrailPredictor.loading import (
    activity_file_name,
    load_activities,
    parse_strava_date,
    running_activities,
)
from TrailPredictor.plot import PlotActivities

TRACK_MAP_HEIGHT = 500  # px


@st.cache_data
def st_load_activities() -> pd.DataFrame:
    """Load the running activities from the CSV summary (cached).

    Returns
    -------
    pandas.DataFrame
        Summary rows of the ``run`` activities.
    """
    return running_activities(load_activities())


def format_label(activity: pd.Series) -> str:
    """Format an activity as a selector label.

    Parameters
    ----------
    activity : pandas.Series
        Row of the activity summary, with ``date``, ``distance_m`` and
        ``name``.

    Returns
    -------
    str
        Label ``"<distance> km — <YYYY-MM-DD> — <name>"``.
    """
    date = parse_strava_date(activity["date"])
    date_text = date.strftime("%Y-%m-%d") if not pd.isna(date) else "date inconnue"
    distance = activity["distance_m"] / 1000
    distance_text = f"{distance:.1f}" if pd.notna(distance) else "—"
    return f"{distance_text} km — {date_text} — {activity['name']}"


def load_activity(
    activity_id: int,
    segment_width: int,
    _progress_cb: Callable[[float, str], None] | None = None,
) -> RunningActivities:
    """Load the FIT activity of the summary row `activity_id`.

    Parameters
    ----------
    activity_id : int
        Strava identifier of the activity.
    segment_width : int
        Segment length in metres.

    Returns
    -------
    RunningActivities
        The segmented activity.

    Raises
    ------
    UnknownActivityError
        If `activity_id` is not in the summary.
    """
    file_name = activity_file_name(st_load_activities(), activity_id)
    return RunningActivities.from_computed_file(
        file_name, segment_width=segment_width, progress_cb=_progress_cb
    )


def render_sidebar(activities: pd.DataFrame) -> tuple[pd.Series, dict[str, Any]]:
    """Render the sort order, activity selector and display options.

    Parameters
    ----------
    activities : pandas.DataFrame
        Running activities to choose from.

    Returns
    -------
    selected : pandas.Series
        Summary row of the selected activity.
    options : dict[str, Any]
        Display options chosen in the sidebar.
    """
    order = st.sidebar.radio(
        "Ordre des activités", ["Décroissant", "Croissant"], horizontal=True
    )
    ordered = activities.sort_values(
        "distance_m", ascending=order == "Croissant", na_position="last"
    ).reset_index(drop=True)
    labels = [format_label(activity) for _, activity in ordered.iterrows()]
    selected_label = st.sidebar.selectbox("Activité", labels, index=0)
    selected = ordered.iloc[labels.index(selected_label)]

    st.sidebar.subheader("Options")
    options = {
        "segment_width": st.sidebar.slider(
            "Taille des segments (m)",
            5,
            1000,
            100,
            step=5,
            help="Granularité utilisée pour agréger les données FIT.",
        ),
        "drop_threshold": st.sidebar.slider(
            "Seuil de dénivelé (m)", 0.5, 10.0, 2.0, step=0.5
        ),
        "color_by": st.sidebar.selectbox(
            "Couleur de la carte", ["altitude", "heart_rate", "pace"], index=0
        ),
        "x_axis": st.sidebar.radio(
            "Axe de la FC",
            ["distance", "time"],
            format_func=lambda x: {"distance": "Distance", "time": "Temps"}[x],
        ),
        "texture": st.sidebar.radio(
            "Fond de la carte 3D",
            ["osm", "satellite"],
            format_func=lambda x: {"osm": "OpenStreetMap", "satellite": "Satellite"}[x],
            horizontal=True,
        ),
        "show_km_markers": st.sidebar.checkbox(
            "Afficher les repères kilométriques", True
        ),
    }
    return selected, options


def _metric_value(value: float, formatter: Callable[[float], str]) -> str | None:
    return formatter(value) if np.isfinite(value) else None


def render_metrics(plot: PlotActivities) -> None:
    """Render the summary metrics available in the FIT data.

    Parameters
    ----------
    plot : PlotActivities
        Activity to summarise.
    """
    metrics = compute_activity_metrics(plot.data)
    metric_values = [
        ("Distance", _metric_value(metrics.distance_km, lambda x: f"{x:.2f} km")),
        (
            "Durée",
            _metric_value(metrics.duration_s, lambda x: str(pd.Timedelta(seconds=x))),
        ),
        (
            "Dénivelé positif",
            _metric_value(metrics.elevation_gain_m, lambda x: f"{x:.0f} m"),
        ),
        (
            "Allure moyenne",
            _metric_value(metrics.average_pace_min_per_km, lambda x: f"{x:.2f} min/km"),
        ),
        (
            "FC moyenne",
            _metric_value(metrics.average_heart_rate_bpm, lambda x: f"{x:.0f} bpm"),
        ),
    ]
    available = [(label, value) for label, value in metric_values if value is not None]
    for column, (label, value) in zip(st.columns(len(available)), available):
        column.metric(label, value)


def _render_chart(
    name: str,
    method: Callable[[], Any],
    key: str,
    **kwargs: Any,
) -> None:
    try:
        figure = method(**kwargs)
        st.plotly_chart(figure, width="stretch", key=key)
    # UI boundary: any plotting failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.warning(f"{name} : {error}")


def render_track_map(plot: PlotActivities, options: dict[str, Any]) -> None:
    """Render the 3D map of the GPS track and its colour legend.

    Parameters
    ----------
    plot : PlotActivities
        Activity to map.
    options : dict[str, Any]
        Display options chosen in the sidebar.
    """
    try:
        track = plot.track_map(
            color_by=options["color_by"],
            show_km_markers=options["show_km_markers"],
            texture=options["texture"],
        )
    # UI boundary: any plotting failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.warning(f"Parcours GPS : {error}")
        return
    st.pydeck_chart(track.deck, height=TRACK_MAP_HEIGHT, key="track-map")
    legend = track.legend
    st.markdown(
        f"**{legend.label}** &nbsp; {legend.vmin_text} "
        f'<span style="display:inline-block;width:12rem;height:0.8rem;'
        f'vertical-align:middle;background:{legend.css_gradient}"></span> '
        f"{legend.vmax_text}",
        unsafe_allow_html=True,
    )
    st.caption(f"{legend.attribution} · Elevation: AWS Terrain Tiles")


def main() -> None:
    """Render the activity analysis page."""
    st.title("Analyse d'activité Strava")
    activities = st_load_activities()
    if activities.empty:
        st.warning("Aucune activité de course avec un fichier FIT disponible.")
        return
    selected, options = render_sidebar(activities)
    progress_bar = st.progress(0.0, text="Démarrage...")

    def update_progress(fraction: float, text: str) -> None:
        progress_bar.progress(min(max(fraction, 0.0), 1.0), text=text)

    try:
        running_activity = load_activity(
            int(selected["id"]), options["segment_width"], update_progress
        )
        plot = PlotActivities(running_activity, options["drop_threshold"])
    # UI boundary: any loading failure is shown instead of crashing the page.
    except Exception as error:  # pylint: disable=broad-exception-caught
        st.error(f"Impossible de charger l'activité : {error}")
        return
    finally:
        progress_bar.empty()

    render_metrics(plot)
    render_track_map(plot, options)
    altitude, heart_rate = st.columns(2)
    with altitude:
        _render_chart("Altitude", plot.altitude, "altitude")
    with heart_rate:
        _render_chart(
            "Fréquence cardiaque",
            plot.hr_overlay,
            "hr-overlay",
            x_axis=options["x_axis"],
        )
    pace, zones = st.columns(2)
    with pace:
        _render_chart("Allure par segment", plot.pace_by_segment, "pace-by-segment")
    with zones:
        _render_chart("Zones cardiaques", plot.time_in_hr_zones, "time-in-hr-zones")


if __name__ == "__main__":
    main()

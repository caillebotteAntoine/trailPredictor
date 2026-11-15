"""Layout helper and target constants shared by the plotting modules."""

import plotly.graph_objects as go


def _base_layout(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        title={"text": title, "subtitle": {"text": subtitle}},
        margin={"l": 50, "r": 25, "t": 75 if subtitle else 55, "b": 45},
        legend={"orientation": "h", "y": 1.02, "x": 0},
        hoverlabel={"namelength": -1},
    )
    return fig


TARGETS_UNITS = {
    "pace": "min/km",
    "log_pace": "log(min/km)",
    "speed": "m/s",
    "log_speed": "log(m/s)",
    "accel_pace": "min/km²",
    "log_accel_pace": "log(min/km²)",
    "accel_speed": "m/s²",
    "log_accel_speed": "log(m/s²)",
}
TARGETS_COLORS = {
    "pace": "#56B4E9",
    "log_pace": "#E95656",
    "speed": "#5BE956",
    "log_speed": "#E456E9",
    "accel_pace": "#E9E956",
    "log_accel_pace": "#56E9E9",
    "accel_speed": "#E9A956",
    "log_accel_speed": "#A956E9",
}

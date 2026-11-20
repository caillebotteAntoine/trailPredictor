"""Plotly figures sent to the front end, shared by the API pages."""

import json
import logging
from collections.abc import Callable
from typing import Any

import plotly.graph_objects as go
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChartView(BaseModel):
    """Plotly figure of a chart, or the error that prevented it.

    Attributes
    ----------
    figure : dict or None
        Plotly figure JSON (``data`` and ``layout``).
    error : str or None
        Error message when the figure could not be built.
    """

    figure: dict[str, Any] | None = None
    error: str | None = None


def build_chart(name: str, build: Callable[[], go.Figure]) -> ChartView:
    """Build a figure, reporting its error instead of raising.

    A failing chart must not hide the other charts of a page, so any error
    is logged and returned in the view.

    Parameters
    ----------
    name : str
        Name of the chart, for the logs.
    build : callable
        Returns the Plotly figure.

    Returns
    -------
    ChartView
        The figure JSON, or the error message.
    """
    try:
        return ChartView(figure=json.loads(build().to_json()))
    # UI boundary: a failing chart is reported, the others are still shown.
    except Exception as error:  # pylint: disable=broad-exception-caught
        logger.exception("Chart %s failed", name)
        return ChartView(error=str(error) or repr(error))

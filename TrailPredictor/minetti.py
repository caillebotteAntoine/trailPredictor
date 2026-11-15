"""Energy cost estimates for walking and running from Minetti et al. (2002)."""

from __future__ import annotations

import numpy as np
import pandas as pd

type Numeric = float | np.ndarray | pd.Series

_MIN_GRADE = -0.45
_MAX_GRADE = 0.45
_RUNNING_COEFFICIENTS = np.array((3.6, 19.5, 46.3, -43.3, -30.4, 155.4))
_WALKING_COEFFICIENTS = np.array((2.5, 19.6, 51.9, -76.8, -58.7, 280.5))


def _cost_of_locomotion(decimal_grade: Numeric, coefficients: np.ndarray) -> Numeric:
    """Evaluate a Minetti cost curve for one or more decimal grades."""
    grade = np.asarray(decimal_grade)
    clipped_grade = np.clip(grade, _MIN_GRADE, _MAX_GRADE)
    cost = coefficients[0] + sum(
        coefficient * clipped_grade**power
        for power, coefficient in enumerate(coefficients[1:], start=1)
    )

    if cost.ndim == 0:
        return float(cost)
    return cost


def cost_of_running(decimal_grade: Numeric) -> Numeric:
    """Return the running energy cost at the given decimal grade.

    Parameters
    ----------
    decimal_grade : float or numpy.ndarray
        Slope as a decimal fraction (e.g. ``0.1`` for 10 %).

    Returns
    -------
    float or numpy.ndarray
        Energy cost in J/kg/m, a float for a scalar input and an array of
        the same shape otherwise.

    See Also
    --------
    cost_of_walking : Minimum walking energy cost.

    Notes
    -----
    The curve is a fifth-degree polynomial valid for grades from -45 % to
    45 % [1]_. Values outside that range are clipped to the nearest valid
    endpoint.

    References
    ----------
    .. [1] Minetti, A. E. et al., "Energy cost of walking and running at
       extreme uphill and downhill slopes", Journal of Applied Physiology,
       2002.

    Examples
    --------
    >>> cost_of_running(0.0)
    3.6
    """
    return _cost_of_locomotion(decimal_grade, _RUNNING_COEFFICIENTS)


def cost_of_walking(decimal_grade: Numeric) -> Numeric:
    """Return the minimum walking energy cost at the given decimal grade.

    Parameters
    ----------
    decimal_grade : float or numpy.ndarray
        Slope as a decimal fraction (e.g. ``0.1`` for 10 %).

    Returns
    -------
    float or numpy.ndarray
        Energy cost in J/kg/m, a float for a scalar input and an array of
        the same shape otherwise.

    See Also
    --------
    cost_of_running : Running energy cost.

    Notes
    -----
    The curve is the minimum cost over walking speeds, valid for grades from
    -45 % to 45 % [1]_. Values outside that range are clipped to the nearest
    valid endpoint.

    References
    ----------
    .. [1] Minetti, A. E. et al., "Energy cost of walking and running at
       extreme uphill and downhill slopes", Journal of Applied Physiology,
       2002.

    Examples
    --------
    >>> cost_of_walking(0.0)
    2.5
    """
    return _cost_of_locomotion(decimal_grade, _WALKING_COEFFICIENTS)

"""Energy cost estimates for walking and running from Minetti et al. (2002)."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np


Numeric: TypeAlias = float | np.ndarray

_MIN_GRADE = -0.45
_MAX_GRADE = 0.45
_RUNNING_COEFFICIENTS = np.array((3.6, 19.5, 46.3, -43.3, -30.4, 155.4))
_WALKING_COEFFICIENTS = np.array((2.5, 19.6, 51.9, -76.8, -58.7, 280.5))


def _cost_of_locomotion(
    decimal_grade: Numeric, coefficients: np.ndarray
) -> Numeric:
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
    """Return running energy cost in J/kg/m at the given decimal grade.

    The Minetti et al. (2002) curve is valid for grades from -45% to 45%.
    Values outside that range are clipped to the nearest valid endpoint.
    Scalars and array-like inputs are supported.
    """
    return _cost_of_locomotion(decimal_grade, _RUNNING_COEFFICIENTS)


def cost_of_walking(decimal_grade: Numeric) -> Numeric:
    """Return minimum walking energy cost in J/kg/m at the given decimal grade.

    The Minetti et al. (2002) curve represents the minimum cost over walking
    speeds and is valid for grades from -45% to 45%. Values outside that range
    are clipped to the nearest valid endpoint. Scalars and array-like inputs
    are supported.
    """
    return _cost_of_locomotion(decimal_grade, _WALKING_COEFFICIENTS)


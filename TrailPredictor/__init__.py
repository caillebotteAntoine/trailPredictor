"""Trail running time prediction from Strava FIT activities."""

# The package keeps its historical PascalCase name used by every import.
# pylint: disable=invalid-name

from TrailPredictor.activities import RunningActivities
from TrailPredictor.minetti import cost_of_running, cost_of_walking
from TrailPredictor.utils import DATA_DIR, COMPUTED_DIR

__all__ = [
    "COMPUTED_DIR",
    "DATA_DIR",
    "RunningActivities",
    "cost_of_running",
    "cost_of_walking",
]

DELETE_COMPUTED = False

if DELETE_COMPUTED:
    for file in COMPUTED_DIR.glob("**/*.csv"):
        print(f"Deleting {file}")
        file.unlink()

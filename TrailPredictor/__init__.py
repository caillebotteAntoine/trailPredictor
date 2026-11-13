from TrailPredictor.activities import RunningActivities
from TrailPredictor.minetti import cost_of_running, cost_of_walking
from TrailPredictor.utils import DATA_DIR, COMPUTED_DIR

DELETE_COMPUTED = False

if DELETE_COMPUTED:
    for file in COMPUTED_DIR.glob("**/*.csv"):
        print(f"Deleting {file}")
        file.unlink()

"""Tests for :mod:`TrailPredictor.api.activity_routes` and the page assets."""

import json
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from TrailPredictor.activities import RunningActivities
from TrailPredictor.api import create_app
from TrailPredictor.api.activity_service import CHARTS, ActivityService
from TrailPredictor.api.jobs import JobManager
from TrailPredictor.load_fit import ProgressCallback

ALTITUDE = [100.0, 110.0, 105.0, 120.0, 118.0, 125.0]
NO_GPS_ID = 3


@pytest.fixture
def summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, NO_GPS_ID, 4],
            "type": ["run", "run", "run", "ride"],
            "name": ["Short", "Long", "Treadmill", "Ride"],
            "date": ["2 janv. 2026, 08:00:00"] * 4,
            "distance_m": [5000.0, 42195.0, 10000.0, 60000.0],
            "file_name": ["a/1.fit.gz", "a/2.fit.gz", "a/3.fit.gz", "a/4.fit"],
        }
    )


@pytest.fixture
def client(summary: pd.DataFrame, make_segments) -> Iterator[TestClient]:
    def load(file_name: str, _width: int, progress: ProgressCallback):
        progress(0.5, "Loading CSV file...")
        segments = make_segments(ALTITUDE)
        if file_name == "a/3.fit.gz":
            segments = segments.assign(position_lat=np.nan, position_long=np.nan)
        return RunningActivities(segments)

    jobs = JobManager()
    app = create_app(jobs, activity_service=ActivityService(lambda: summary, load))
    with TestClient(app) as test_client:
        yield test_client
    jobs.shutdown()


def load(client: TestClient, activity_id: int, segment_width: int = 100) -> dict:
    """Load an activity and return the last event of its job."""
    response = client.post(
        f"/api/activities/{activity_id}/load", params={"segment_width": segment_width}
    )
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():  # loop over the stream, not over data
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events[-1]


# --- list


def test_list_activities(client: TestClient) -> None:
    response = client.get("/api/activities")

    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [2, NO_GPS_ID, 1]
    assert response.json()[0]["date"] == "2026-01-02T08:00:00"


def test_list_activities_without_summary_is_503() -> None:
    def missing() -> pd.DataFrame:
        raise FileNotFoundError("data/activities.csv")

    app = create_app(activity_service=ActivityService(missing))
    with TestClient(app) as test_client:
        response = test_client.get("/api/activities")

    assert response.status_code == 503
    assert "data/activities.csv" in response.json()["detail"]


# --- load


def test_load_activity_job_succeeds(client: TestClient) -> None:
    assert load(client, 1)["status"] == "succeeded"


@pytest.mark.parametrize("activity_id", [4, 99], ids=["not_a_run", "missing"])
def test_load_unknown_activity_is_404(client: TestClient, activity_id: int) -> None:
    response = client.post(f"/api/activities/{activity_id}/load")

    assert response.status_code == 404


@pytest.mark.parametrize("segment_width", [0, 7, 1005])
def test_load_invalid_segment_width_is_422(
    client: TestClient, segment_width: int
) -> None:
    response = client.post(
        "/api/activities/1/load", params={"segment_width": segment_width}
    )

    assert response.status_code == 422


# --- charts


def test_charts_of_a_loaded_activity(client: TestClient) -> None:
    load(client, 1, segment_width=200)

    response = client.get(
        "/api/activities/1/charts",
        params={"segment_width": 200, "drop_threshold": 5.0, "x_axis": "time"},
    )

    assert response.status_code == 200
    body = response.json()
    assert set(body["charts"]) == set(CHARTS)
    assert all(chart["figure"]["data"] for chart in body["charts"].values())
    assert body["metrics"]["distance_km"] == pytest.approx(6.0)


def test_charts_before_loading_is_409(client: TestClient) -> None:
    load(client, 1, segment_width=100)

    response = client.get("/api/activities/1/charts", params={"segment_width": 200})

    assert response.status_code == 409


@pytest.mark.parametrize(
    "params",
    [{"x_axis": "speed"}, {"drop_threshold": 0.0}, {"drop_threshold": 11.0}],
)
def test_charts_invalid_options_are_422(client: TestClient, params: dict) -> None:
    load(client, 1)

    assert client.get("/api/activities/1/charts", params=params).status_code == 422


# --- track


def test_track_of_a_loaded_activity(client: TestClient) -> None:
    load(client, 1)

    response = client.get(
        "/api/activities/1/track",
        params={"color_by": "pace", "texture": "satellite", "show_km_markers": False},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["deck"]["layers"]
    assert body["tooltip"] == {"html": "{tooltip}"}
    assert body["legend"]["label"] == "Pace (min/km)"


def test_track_without_gps_is_422(client: TestClient) -> None:
    load(client, NO_GPS_ID)

    response = client.get(f"/api/activities/{NO_GPS_ID}/track")

    assert response.status_code == 422


def test_track_before_loading_is_409(client: TestClient) -> None:
    assert client.get("/api/activities/1/track").status_code == 409


@pytest.mark.parametrize("params", [{"color_by": "speed"}, {"texture": "mapbox"}])
def test_track_invalid_options_are_422(client: TestClient, params: dict) -> None:
    load(client, 1)

    assert client.get("/api/activities/1/track", params=params).status_code == 422


# --- page and scripts


@pytest.mark.parametrize("name", ["plotly.min.js", "deckgl.min.js"])
def test_package_scripts_are_served(client: TestClient, name: str) -> None:
    response = client.get(f"/lib/{name}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/javascript")
    assert len(response.content) > 100_000


def test_unknown_package_script_is_404(client: TestClient) -> None:
    assert client.get("/lib/jquery.js").status_code == 404


def test_activity_page_and_its_assets(client: TestClient) -> None:
    page = client.get("/activities.html")

    assert page.status_code == 200
    for asset in [
        "css/app.css",
        "lib/plotly.min.js",
        "lib/deckgl.min.js",
        "js/activities.js",
        "vendor/alpine-3.14.9.min.js",
    ]:
        assert f'"{asset}"' in page.text
        assert client.get(f"/{asset}").status_code == 200

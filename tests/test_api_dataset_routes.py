"""Tests for :mod:`TrailPredictor.api.dataset_routes` and the dataset page."""

import json
from collections.abc import Iterator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from TrailPredictor.api import create_app
from TrailPredictor.api.dataset_service import DatasetService
from TrailPredictor.api.jobs import JobManager
from TrailPredictor.dataset import Dataset
from TrailPredictor.load_fit import ProgressCallback

PARAMS = {"min_distance": 40, "max_distance": 60, "segment_width": 1000}
SELECTION = {**PARAMS, "target": "speed", "cuts": [150, 170]}


@pytest.fixture
def client(dataset: pd.DataFrame) -> Iterator[TestClient]:
    def load(_min: int, _max: int, _width: int, progress: ProgressCallback):
        progress(0.5, "Reading FIT files")
        return Dataset(dataset)

    jobs = JobManager()
    service = DatasetService(load, saved_results=list)
    with TestClient(create_app(jobs, dataset_service=service)) as test_client:
        yield test_client
    jobs.shutdown()


def load(client: TestClient, params: dict | None = None) -> dict:
    """Load a dataset and return the last event of its job."""
    response = client.post("/api/dataset/load", params=params or PARAMS)
    assert response.status_code == 202
    job_id = response.json()["job_id"]
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():  # loop over the stream, not over data
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events[-1]


@pytest.fixture
def loaded(client: TestClient) -> TestClient:
    assert load(client)["status"] == "succeeded"
    return client


# --- load


@pytest.mark.parametrize(
    "params",
    [
        {**PARAMS, "min_distance": 60},  # empty range
        {**PARAMS, "min_distance": 0},
        {**PARAMS, "segment_width": 150},  # not a multiple of 100
        {**PARAMS, "segment_width": 6000},
    ],
)
def test_load_invalid_params_are_422(client: TestClient, params: dict) -> None:
    assert client.post("/api/dataset/load", params=params).status_code == 422


# --- features and presets


def test_features(loaded: TestClient) -> None:
    response = loaded.get("/api/dataset/features", params={**PARAMS, "target": "speed"})

    assert response.status_code == 200
    body = response.json()
    assert body["n_rows"] > 0
    assert [row["name"] for row in body["rows"]] == body["ranked"]


def test_features_with_filters(loaded: TestClient) -> None:
    response = loaded.get(
        "/api/dataset/features",
        params={**PARAMS, "groups": ["past", "upcoming"], "search": "drop"},
    )

    rows = response.json()["rows"]
    assert rows
    assert {row["group"] for row in rows} <= {"past", "upcoming"}
    assert all("drop" in row["name"] for row in rows)


def test_features_before_loading_is_409(client: TestClient) -> None:
    assert client.get("/api/dataset/features", params=PARAMS).status_code == 409


@pytest.mark.parametrize(
    "params",
    [{"groups": ["unknown"]}, {"min_abs": 2}, {"max_missing": -1}, {"target": "x"}],
)
def test_features_invalid_filters_are_422(loaded: TestClient, params: dict) -> None:
    response = loaded.get("/api/dataset/features", params={**PARAMS, **params})

    assert response.status_code == 422


def test_preset(loaded: TestClient) -> None:
    response = loaded.get(
        "/api/dataset/preset", params={**PARAMS, "preset": "top_5", "search": "back"}
    )

    assert response.status_code == 200
    names = response.json()
    assert 0 < len(names) <= 5
    assert all("back" in name for name in names)


def test_unknown_preset_is_422(loaded: TestClient) -> None:
    response = loaded.get("/api/dataset/preset", params={**PARAMS, "preset": "top_3"})

    assert response.status_code == 422


# --- charts and cuts


def test_target_histogram_chart(loaded: TestClient) -> None:
    response = loaded.get("/api/dataset/chart", params={**PARAMS, "cuts": "150, 170"})

    assert response.status_code == 200
    assert response.json()["figure"]["data"]


def test_feature_chart(loaded: TestClient) -> None:
    feature = loaded.get("/api/dataset/features", params=PARAMS).json()["ranked"][0]

    response = loaded.get(
        "/api/dataset/chart", params={**PARAMS, "cuts": "150", "feature": feature}
    )

    assert response.json()["error"] is None


def test_unknown_feature_chart_reports_the_error(loaded: TestClient) -> None:
    response = loaded.get(
        "/api/dataset/chart", params={**PARAMS, "cuts": "150", "feature": "missing"}
    )

    assert response.status_code == 200
    assert "missing" in response.json()["error"]


def test_chart_before_loading_is_409(client: TestClient) -> None:
    response = client.get("/api/dataset/chart", params={**PARAMS, "cuts": "150"})

    assert response.status_code == 409


@pytest.mark.parametrize("cuts", ["170, 150", "abc", "", "0"])
def test_chart_invalid_cuts_are_422(loaded: TestClient, cuts: str) -> None:
    response = loaded.get("/api/dataset/chart", params={**PARAMS, "cuts": cuts})

    assert response.status_code == 422


def test_parse_cuts(client: TestClient) -> None:
    response = client.get("/api/dataset/cuts", params={"cuts": "142; 161 170,183"})

    assert response.json() == [142, 161, 170, 183]


@pytest.mark.parametrize("cuts", ["170, 150", "142,x"])
def test_parse_invalid_cuts_is_422(client: TestClient, cuts: str) -> None:
    response = client.get("/api/dataset/cuts", params={"cuts": cuts})

    assert response.status_code == 422
    assert response.json()["detail"].startswith("Invalid cuts")


def test_saved_zones_empty(client: TestClient) -> None:
    assert client.get("/api/dataset/saved-zones").json() == []


# --- current dataset


def test_current_dataset(loaded: TestClient) -> None:
    assert loaded.get("/api/dataset/current").json() is None

    response = loaded.put("/api/dataset/current", json=SELECTION)

    assert response.status_code == 200
    assert loaded.get("/api/dataset/current").json() == SELECTION


def test_select_a_dataset_not_loaded_is_409(client: TestClient) -> None:
    assert client.put("/api/dataset/current", json=SELECTION).status_code == 409


@pytest.mark.parametrize(
    "selection",
    [{**SELECTION, "cuts": [170, 150]}, {**SELECTION, "target": "x"}],
)
def test_select_invalid_dataset_is_422(loaded: TestClient, selection: dict) -> None:
    assert loaded.put("/api/dataset/current", json=selection).status_code == 422


# --- page


def test_dataset_page_and_its_assets(client: TestClient) -> None:
    page = client.get("/dataset.html")

    assert page.status_code == 200
    for asset in [
        "css/app.css",
        "lib/plotly.min.js",
        "js/dataset.js",
        "vendor/alpine-3.14.9.min.js",
    ]:
        assert f'"{asset}"' in page.text
        assert client.get(f"/{asset}").status_code == 200
    assert client.get("/js/ui.js").status_code == 200

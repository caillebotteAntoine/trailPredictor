"""Tests for :mod:`TrailPredictor.api.heart_rate_routes` and the zones page."""

import json
from collections.abc import Iterator
from pathlib import Path

import optuna
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from TrailPredictor.api import create_app
from TrailPredictor.api.dataset_service import DatasetService
from TrailPredictor.api.heart_rate_service import HeartRateService
from TrailPredictor.api.jobs import JobManager
from TrailPredictor.dataset import Dataset
from TrailPredictor.load_fit import ProgressCallback

PARAMS = {"min_distance": 40, "max_distance": 60, "segment_width": 1000}
SELECTION = {**PARAMS, "target": "speed", "cuts": [150]}
CONFIG = {"zone_counts": [2, 3], "n_trials": 2, "max_trials_factor": 5, "n_splits": 2}


@pytest.fixture(autouse=True)
def quiet_optuna() -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)


@pytest.fixture
def client(dataset: pd.DataFrame, tmp_path: Path) -> Iterator[TestClient]:
    def load(_min: int, _max: int, _width: int, _progress: ProgressCallback):
        return Dataset(dataset)

    jobs = JobManager()
    datasets = DatasetService(load, saved_results=list)
    app = create_app(
        jobs,
        dataset_service=datasets,
        heart_rate_service=HeartRateService(datasets, models_dir=tmp_path),
    )
    with TestClient(app) as test_client:
        yield test_client
    jobs.shutdown()


def follow(client: TestClient, job_id: str) -> dict:
    """Return the last event of a job."""
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as stream:
        for line in stream.iter_lines():  # loop over the stream, not over data
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
    return events[-1]


@pytest.fixture
def with_dataset(client: TestClient) -> TestClient:
    """Client whose dataset page chose a dataset."""
    job_id = client.post("/api/dataset/load", params=PARAMS).json()["job_id"]
    assert follow(client, job_id)["status"] == "succeeded"
    assert client.put("/api/dataset/current", json=SELECTION).status_code == 200
    return client


@pytest.fixture
def optimized(with_dataset: TestClient) -> tuple[TestClient, str]:
    """Client with one saved optimisation, and its key."""
    response = with_dataset.post(
        "/api/heart-rate/optimize", json={"target": "speed", "config": CONFIG}
    )
    assert response.status_code == 202
    body = response.json()
    assert not body["already_saved"]
    assert follow(with_dataset, body["job_id"])["status"] == "succeeded"
    return with_dataset, body["key"]


# --- context and saved list


def test_context_without_dataset(client: TestClient) -> None:
    body = client.get("/api/heart-rate/context").json()

    assert body["dataset"] is None
    assert body["defaults"]["zone_counts"] == [5]
    assert body["limits"]["min_zones"] == 2


def test_context_with_dataset(with_dataset: TestClient) -> None:
    body = with_dataset.get("/api/heart-rate/context").json()

    assert body["dataset"]["selection"] == SELECTION
    assert body["dataset"]["n_rows"] > 0


def test_saved_empty(client: TestClient) -> None:
    assert client.get("/api/heart-rate/saved").json() == []


# --- optimisation


def test_optimize_without_dataset_is_409(client: TestClient) -> None:
    response = client.post(
        "/api/heart-rate/optimize", json={"target": "speed", "config": CONFIG}
    )

    assert response.status_code == 409


@pytest.mark.parametrize(
    "body",
    [
        {"target": "speed", "config": {**CONFIG, "zone_counts": [1]}},
        {"target": "speed", "config": {**CONFIG, "unknown": 1}},
        {"target": "speed", "config": {**CONFIG, "first_cut_range": [170, 120]}},
        {"target": "x", "config": CONFIG},
    ],
)
def test_optimize_invalid_body_is_422(with_dataset: TestClient, body: dict) -> None:
    assert with_dataset.post("/api/heart-rate/optimize", json=body).status_code == 422


def test_optimize_saves_the_result(optimized: tuple[TestClient, str]) -> None:
    client, key = optimized

    saved = client.get("/api/heart-rate/saved").json()

    assert [result["key"] for result in saved] == [key]
    assert saved[0]["target_name"] == "speed"
    assert saved[0]["config"]["zone_counts"] == [2, 3]


def test_optimize_twice_reloads(optimized: tuple[TestClient, str]) -> None:
    client, key = optimized

    response = client.post(
        "/api/heart-rate/optimize", json={"target": "speed", "config": CONFIG}
    )

    assert response.json()["already_saved"]
    assert response.json()["key"] == key
    assert follow(client, response.json()["job_id"])["status"] == "succeeded"


# --- results


def test_result(optimized: tuple[TestClient, str]) -> None:
    client, key = optimized

    response = client.get("/api/heart-rate/result", params={"key": key})

    assert response.status_code == 200
    body = response.json()
    assert body["has_completed_trials"]
    assert body["zone_counts"] == [2, 3]
    assert body["reference"] is not None  # computed on the current dataset
    assert body["best_zones_chart"]["figure"]["data"]


def test_trials_chart(optimized: tuple[TestClient, str]) -> None:
    client, key = optimized

    response = client.get(
        "/api/heart-rate/trials-chart", params={"key": key, "n_zones": 3}
    )

    assert response.json()["figure"]["data"]


@pytest.mark.parametrize("route", ["result", "trials-chart"])
def test_unknown_result_is_404(client: TestClient, route: str) -> None:
    response = client.get(
        f"/api/heart-rate/{route}", params={"key": "C:/Windows/win.ini", "n_zones": 2}
    )

    assert response.status_code == 404


def test_trials_chart_invalid_n_zones_is_422(
    optimized: tuple[TestClient, str],
) -> None:
    client, key = optimized

    response = client.get(
        "/api/heart-rate/trials-chart", params={"key": key, "n_zones": 1}
    )

    assert response.status_code == 422


# --- page


def test_heart_rate_page_and_its_assets(client: TestClient) -> None:
    page = client.get("/heart_rate.html")

    assert page.status_code == 200
    for asset in [
        "css/app.css",
        "lib/plotly.min.js",
        "js/heart_rate.js",
        "vendor/alpine-3.14.9.min.js",
    ]:
        assert f'"{asset}"' in page.text
        assert client.get(f"/{asset}").status_code == 200

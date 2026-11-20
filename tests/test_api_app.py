"""Tests for :mod:`TrailPredictor.api.app`."""

import asyncio
import json
import threading
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from TrailPredictor.api.app import job_events
from TrailPredictor.api import create_app
from TrailPredictor.api.jobs import JobManager
from TrailPredictor.load_fit import ProgressCallback

TIMEOUT_S = 5.0  # generous: only reached if a job hangs


@pytest.fixture
def manager() -> Iterator[JobManager]:
    jobs = JobManager()
    yield jobs
    jobs.shutdown()


@pytest.fixture
def client(manager: JobManager) -> Iterator[TestClient]:
    with TestClient(create_app(job_manager=manager)) as test_client:
        yield test_client


def read_events(client: TestClient, job_id: str, on_first=None) -> list[dict]:
    """Return the data events of a job stream; call `on_first` after the first."""
    events = []
    with client.stream("GET", f"/api/jobs/{job_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():  # loop over the stream, not over data
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
                if on_first is not None and len(events) == 1:
                    on_first()
    return events


# --- static front end


def test_health(client: TestClient) -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_is_served_at_root(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<title>Trail Predictor</title>" in response.text


@pytest.mark.parametrize(
    "path",
    ["/js/app.js", "/js/api.js", "/css/app.css", "/vendor/alpine-3.14.9.min.js"],
)
def test_assets_referenced_by_the_page_exist(client: TestClient, path: str) -> None:
    assert client.get(path).status_code == 200


def test_unknown_static_file_is_404(client: TestClient) -> None:
    assert client.get("/missing.js").status_code == 404


def test_custom_static_dir(manager: JobManager, tmp_path) -> None:
    (tmp_path / "index.html").write_text("<p>custom</p>", encoding="utf-8")

    with TestClient(create_app(manager, static_dir=tmp_path)) as custom:
        assert custom.get("/").text == "<p>custom</p>"


# --- jobs


@pytest.mark.parametrize("suffix", ["", "/events"])
def test_unknown_job_is_404(client: TestClient, suffix: str) -> None:
    response = client.get(f"/api/jobs/missing{suffix}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Unknown job"}


def test_job_status(client: TestClient, manager: JobManager) -> None:
    job_id = manager.submit(lambda _report: None)
    read_events(client, job_id)  # wait until finished

    response = client.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["status"] == "succeeded"
    assert body["progress"] == pytest.approx(1.0)
    assert body["error"] is None


def test_job_events_stream_progress_until_success(
    client: TestClient, manager: JobManager
) -> None:
    release = threading.Event()

    def task(report: ProgressCallback) -> None:
        release.wait(TIMEOUT_S)
        report(0.5, "halfway")

    job_id = manager.submit(task)
    events = read_events(client, job_id, on_first=release.set)

    versions = [event["version"] for event in events]
    assert versions == sorted(set(versions))
    assert events[0]["status"] in {"pending", "running"}
    assert any(event["message"] == "halfway" for event in events)
    assert events[-1]["status"] == "succeeded"


def test_job_events_end_on_failure(client: TestClient, manager: JobManager) -> None:
    def task(_report: ProgressCallback) -> None:
        raise ValueError("no FIT file")

    events = read_events(client, manager.submit(task))

    assert events[-1]["status"] == "failed"
    assert events[-1]["error"] == "no FIT file"


def test_job_events_send_keepalive_while_waiting(manager: JobManager) -> None:
    release = threading.Event()
    started = threading.Event()

    def task(_report: ProgressCallback) -> None:
        started.set()
        release.wait(TIMEOUT_S)

    async def collect(job_id: str) -> list[str]:
        events: list[str] = []
        async for event in job_events(manager, job_id, keepalive_s=0.05):
            events.append(event)
            if event.startswith(":"):
                release.set()
        return events

    job_id = manager.submit(task)
    assert started.wait(TIMEOUT_S)
    events = asyncio.run(collect(job_id))

    assert ": keepalive\n\n" in events
    assert events[0].startswith("data: ")
    assert '"status":"succeeded"' in events[-1]

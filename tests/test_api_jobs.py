"""Tests for :mod:`TrailPredictor.api.jobs`."""

import threading
import time
from collections.abc import Iterator

import pytest

from TrailPredictor.api.jobs import (
    JobManager,
    JobNotSucceededError,
    JobSnapshot,
    JobStatus,
    UnknownJobError,
)
from TrailPredictor.load_fit import ProgressCallback

TIMEOUT_S = 5.0  # generous: only reached if a job hangs


@pytest.fixture
def manager() -> Iterator[JobManager]:
    jobs = JobManager()
    yield jobs
    jobs.shutdown()


def wait_until_finished(manager: JobManager, job_id: str) -> JobSnapshot:
    deadline = time.monotonic() + TIMEOUT_S
    snapshot = manager.snapshot(job_id)
    while not snapshot.finished:  # waits on the job, not over data
        assert time.monotonic() < deadline, "job did not finish in time"
        snapshot = manager.wait_for_change(job_id, snapshot.version, TIMEOUT_S)
    return snapshot


def test_succeeded_job(manager: JobManager) -> None:
    job_id = manager.submit(lambda report: 42)

    snapshot = wait_until_finished(manager, job_id)

    assert snapshot.status is JobStatus.SUCCEEDED
    assert snapshot.progress == pytest.approx(1.0)
    assert snapshot.error is None
    assert manager.result(job_id) == 42


def test_progress_is_recorded_and_clamped(manager: JobManager) -> None:
    release = threading.Event()
    reported = threading.Event()

    def task(report: ProgressCallback) -> None:
        report(1.5, "beyond the end")
        report(-0.5, "before the start")
        report(0.25, "a quarter")
        reported.set()
        release.wait(TIMEOUT_S)

    job_id = manager.submit(task)
    assert reported.wait(TIMEOUT_S)
    running = manager.snapshot(job_id)
    release.set()
    finished = wait_until_finished(manager, job_id)

    assert running.status is JobStatus.RUNNING
    assert running.progress == pytest.approx(0.25)
    assert running.message == "a quarter"
    assert finished.version > running.version


def test_clamping_bounds(manager: JobManager) -> None:
    seen: list[float] = []
    job_ids: list[str] = []
    submitted = threading.Event()

    def task(report: ProgressCallback) -> None:
        submitted.wait(TIMEOUT_S)  # job_ids is filled by the test thread
        report(1.5, "")
        seen.append(manager.snapshot(job_ids[0]).progress)
        report(-0.5, "")
        seen.append(manager.snapshot(job_ids[0]).progress)

    job_ids.append(manager.submit(task))
    submitted.set()
    wait_until_finished(manager, job_ids[0])

    assert seen == [pytest.approx(1.0), pytest.approx(0.0)]


def test_failed_job(manager: JobManager) -> None:
    def task(_report: ProgressCallback) -> None:
        raise ValueError("bad activity")

    job_id = manager.submit(task)
    snapshot = wait_until_finished(manager, job_id)

    assert snapshot.status is JobStatus.FAILED
    assert snapshot.error == "bad activity"
    with pytest.raises(JobNotSucceededError, match="failed"):
        manager.result(job_id)


def test_failed_job_without_message_reports_the_exception(
    manager: JobManager,
) -> None:
    def task(_report: ProgressCallback) -> None:
        raise KeyError

    snapshot = wait_until_finished(manager, manager.submit(task))

    assert snapshot.error == "KeyError()"


def test_result_of_running_job_raises(manager: JobManager) -> None:
    release = threading.Event()
    job_id = manager.submit(lambda _report: release.wait(TIMEOUT_S))
    try:
        with pytest.raises(JobNotSucceededError):
            manager.result(job_id)
    finally:
        release.set()


def test_wait_for_change_times_out_with_the_same_version(
    manager: JobManager,
) -> None:
    release = threading.Event()
    started = threading.Event()

    def task(_report: ProgressCallback) -> None:
        started.set()
        release.wait(TIMEOUT_S)

    job_id = manager.submit(task)
    assert started.wait(TIMEOUT_S)
    current = manager.snapshot(job_id)
    try:
        waited = manager.wait_for_change(job_id, current.version, timeout=0.05)
    finally:
        release.set()

    assert waited == current


def test_wait_for_change_returns_at_once_for_another_version(
    manager: JobManager,
) -> None:
    job_id = manager.submit(lambda _report: None)
    finished = wait_until_finished(manager, job_id)

    assert manager.wait_for_change(job_id, -1, timeout=TIMEOUT_S) == finished


def test_jobs_have_distinct_ids(manager: JobManager) -> None:
    first = manager.submit(lambda _report: 1)
    second = manager.submit(lambda _report: 2)

    assert first != second
    assert wait_until_finished(manager, first).finished
    assert wait_until_finished(manager, second).finished
    assert (manager.result(first), manager.result(second)) == (1, 2)


@pytest.mark.parametrize(
    "call",
    [
        lambda jobs: jobs.snapshot("missing"),
        lambda jobs: jobs.result("missing"),
        lambda jobs: jobs.wait_for_change("missing", 0, 0.0),
    ],
    ids=["snapshot", "result", "wait_for_change"],
)
def test_unknown_job_raises(manager: JobManager, call) -> None:
    with pytest.raises(UnknownJobError):
        call(manager)


@pytest.mark.parametrize(
    ("status", "finished"),
    [
        (JobStatus.PENDING, False),
        (JobStatus.RUNNING, False),
        (JobStatus.SUCCEEDED, True),
        (JobStatus.FAILED, True),
    ],
)
def test_snapshot_finished(status: JobStatus, finished: bool) -> None:
    assert JobSnapshot("id", status, 0.0, "", None, 0).finished is finished

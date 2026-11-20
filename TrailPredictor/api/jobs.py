"""Background jobs with progress reporting, shared by the API routes.

Long computations (loading an activity, optimising the heart-rate zones) run
in a thread pool; their progress is stored in an immutable `JobSnapshot` that
the API streams to the browser.
"""

import logging
import threading
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from enum import StrEnum

from pydantic import BaseModel

from TrailPredictor.load_fit import ProgressCallback

logger = logging.getLogger(__name__)

DEFAULT_MAX_WORKERS = 2  # a desktop app runs few heavy jobs at once
MIN_PROGRESS, MAX_PROGRESS = 0.0, 1.0

Task = Callable[[ProgressCallback], object]


class JobStatus(StrEnum):
    """Lifecycle of a job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


FINISHED_STATUSES = frozenset({JobStatus.SUCCEEDED, JobStatus.FAILED})


class JobCreated(BaseModel):
    """Response of a route that starts a job.

    Attributes
    ----------
    job_id : str
        Identifier to follow at ``/api/jobs/{job_id}/events``.
    """

    job_id: str


class UnknownJobError(KeyError):
    """Raised when a job identifier was never submitted."""


class JobNotSucceededError(RuntimeError):
    """Raised when the result of an unfinished or failed job is requested."""


@dataclass(frozen=True, slots=True)
class JobSnapshot:
    """State of a job at one point in time.

    Parameters
    ----------
    job_id : str
        Identifier returned by `JobManager.submit`.
    status : JobStatus
        Current status.
    progress : float
        Completed fraction, between 0 and 1.
    message : str
        Last progress message.
    error : str or None
        Error message of a failed job, None otherwise.
    version : int
        Incremented at every change, to detect updates.
    """

    job_id: str
    status: JobStatus
    progress: float
    message: str
    error: str | None
    version: int

    @property
    def finished(self) -> bool:
        """Return True once the job succeeded or failed.

        Returns
        -------
        bool
            Whether the status is final.
        """
        return self.status in FINISHED_STATUSES


class _Job:
    """Mutable holder of a job: its latest snapshot and its result."""

    def __init__(self, job_id: str) -> None:
        self._condition = threading.Condition()
        self._snapshot = JobSnapshot(
            job_id, JobStatus.PENDING, MIN_PROGRESS, "", None, 0
        )
        self.result: object = None

    def snapshot(self) -> JobSnapshot:
        """Return the latest state of the job.

        Returns
        -------
        JobSnapshot
            Current snapshot.
        """
        with self._condition:
            return self._snapshot

    def update(
        self,
        *,
        status: JobStatus | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        """Replace the given fields, bump the version and wake the waiters.

        Parameters
        ----------
        status : JobStatus or None, optional
            New status; unchanged if None.
        progress : float or None, optional
            New completed fraction; unchanged if None.
        message : str or None, optional
            New progress message; unchanged if None.
        error : str or None, optional
            Error message; unchanged if None.
        """
        with self._condition:
            current = self._snapshot
            self._snapshot = replace(
                current,
                status=current.status if status is None else status,
                progress=current.progress if progress is None else progress,
                message=current.message if message is None else message,
                error=current.error if error is None else error,
                version=current.version + 1,
            )
            self._condition.notify_all()

    def wait_for_change(self, version: int, timeout: float) -> JobSnapshot:
        """Block until the version differs from `version`, or until `timeout`.

        Parameters
        ----------
        version : int
            Version already known by the caller.
        timeout : float
            Maximum waiting time, in seconds.

        Returns
        -------
        JobSnapshot
            Latest snapshot.
        """
        with self._condition:
            self._condition.wait_for(
                lambda: self._snapshot.version != version, timeout=timeout
            )
            return self._snapshot


class JobManager:
    """Run tasks in background threads and track their progress.

    Parameters
    ----------
    max_workers : int, optional
        Number of jobs running at the same time. Default is
        ``DEFAULT_MAX_WORKERS``.

    Notes
    -----
    The jobs are kept in memory for the lifetime of the manager, so their
    result can be fetched after they finish.
    """

    def __init__(self, max_workers: int = DEFAULT_MAX_WORKERS) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="job"
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()

    def submit(self, task: Task) -> str:
        """Start a task in the background.

        Parameters
        ----------
        task : callable
            Called with a `ProgressCallback` ``(fraction, message)``; its
            return value becomes the result of the job.

        Returns
        -------
        str
            Identifier of the job.
        """
        job_id = uuid.uuid4().hex
        job = _Job(job_id)
        with self._lock:
            self._jobs[job_id] = job
        self._executor.submit(self._run, job, task)
        return job_id

    @staticmethod
    def _run(job: _Job, task: Task) -> None:
        """Run `task`, recording its progress, result or error in `job`."""

        def report(fraction: float, message: str) -> None:
            job.update(
                progress=min(max(fraction, MIN_PROGRESS), MAX_PROGRESS),
                message=message,
            )

        job.update(status=JobStatus.RUNNING)
        try:
            job.result = task(report)
        # Thread boundary: any failure is logged and reported to the client.
        except Exception as error:  # pylint: disable=broad-exception-caught
            logger.exception("Job %s failed", job.snapshot().job_id)
            job.update(status=JobStatus.FAILED, error=str(error) or repr(error))
        else:
            job.update(status=JobStatus.SUCCEEDED, progress=MAX_PROGRESS)

    def _job(self, job_id: str) -> _Job:
        with self._lock:
            try:
                return self._jobs[job_id]
            except KeyError:
                raise UnknownJobError(job_id) from None

    def snapshot(self, job_id: str) -> JobSnapshot:
        """Return the current state of a job.

        Parameters
        ----------
        job_id : str
            Identifier returned by `submit`.

        Returns
        -------
        JobSnapshot
            Latest state of the job.

        Raises
        ------
        UnknownJobError
            If `job_id` was never submitted.
        """
        return self._job(job_id).snapshot()

    def wait_for_change(self, job_id: str, version: int, timeout: float) -> JobSnapshot:
        """Block until the job changes from `version`, or until `timeout`.

        Parameters
        ----------
        job_id : str
            Identifier returned by `submit`.
        version : int
            Version already known by the caller; any other version returns
            at once.
        timeout : float
            Maximum waiting time, in seconds.

        Returns
        -------
        JobSnapshot
            Latest state of the job; its version equals `version` on timeout.

        Raises
        ------
        UnknownJobError
            If `job_id` was never submitted.
        """
        return self._job(job_id).wait_for_change(version, timeout)

    def result(self, job_id: str) -> object:
        """Return the value returned by the task of a succeeded job.

        Parameters
        ----------
        job_id : str
            Identifier returned by `submit`.

        Returns
        -------
        object
            Return value of the task.

        Raises
        ------
        UnknownJobError
            If `job_id` was never submitted.
        JobNotSucceededError
            If the job is still running or failed.
        """
        job = self._job(job_id)
        status = job.snapshot().status
        if status is not JobStatus.SUCCEEDED:
            raise JobNotSucceededError(f"Job {job_id} is {status}")
        return job.result

    def shutdown(self, wait: bool = True) -> None:
        """Stop accepting jobs and release the worker threads.

        Parameters
        ----------
        wait : bool, optional
            Wait for the running jobs to finish. Default is True.
        """
        self._executor.shutdown(wait=wait, cancel_futures=True)

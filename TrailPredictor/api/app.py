"""FastAPI application serving the API and the HTML/JS front end."""

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from importlib import resources
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import TypeAdapter

from TrailPredictor.api import activity_routes, dataset_routes, heart_rate_routes
from TrailPredictor.api.activity_service import ActivityService
from TrailPredictor.api.dataset_service import DatasetService
from TrailPredictor.api.heart_rate_service import HeartRateService
from TrailPredictor.api.jobs import JobManager, JobSnapshot, UnknownJobError

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
# JavaScript libraries bundled with the Python packages: served from there,
# their version always matches the JSON produced by plotly and pydeck.
PACKAGE_SCRIPTS = {
    "plotly.min.js": ("plotly", "package_data/plotly.min.js"),
    "deckgl.min.js": ("pydeck", "nbextension/static/index.js"),
}
JAVASCRIPT_MEDIA_TYPE = "text/javascript"
APP_TITLE = "Trail Predictor"
# Longest wait between two server-sent events; a comment line keeps the
# connection alive and lets the server notice a closed client.
SSE_KEEPALIVE_S = 15.0
SSE_MEDIA_TYPE = "text/event-stream"

_SNAPSHOT_JSON = TypeAdapter(JobSnapshot)
# The front end changes with the code: the browser must revalidate its copy
# (a cheap 304 thanks to the ETag) instead of reusing a stale one.
REVALIDATE = {"Cache-Control": "no-cache"}


class RevalidatedStaticFiles(StaticFiles):
    """Static files that the browser revalidates at each use."""

    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        """Return the file with a ``Cache-Control: no-cache`` header.

        Parameters
        ----------
        *args, **kwargs
            Arguments of `StaticFiles.file_response`.

        Returns
        -------
        starlette.responses.Response
            The file response.
        """
        response = super().file_response(*args, **kwargs)
        response.headers.update(REVALIDATE)
        return response


def _job_manager(request: Request) -> JobManager:
    manager: JobManager = request.app.state.job_manager
    return manager


def _snapshot_or_404(manager: JobManager, job_id: str) -> JobSnapshot:
    try:
        return manager.snapshot(job_id)
    except UnknownJobError:
        raise HTTPException(status_code=404, detail="Unknown job") from None


async def job_events(
    manager: JobManager, job_id: str, keepalive_s: float = SSE_KEEPALIVE_S
) -> AsyncIterator[str]:
    """Yield the server-sent events of a job until it finishes.

    Parameters
    ----------
    manager : JobManager
        Manager running the job.
    job_id : str
        Identifier of the job.
    keepalive_s : float, optional
        Longest wait between two events, in seconds. Default is
        ``SSE_KEEPALIVE_S``.

    Yields
    ------
    str
        ``data: <JobSnapshot JSON>`` event at every change of the job, the
        last one with a finished status, or a ``: keepalive`` comment when
        nothing changed for `keepalive_s` seconds.

    Raises
    ------
    UnknownJobError
        If `job_id` was never submitted.
    """
    version = -1  # unknown: the current state is sent at once
    while True:
        # the wait blocks a worker thread, not the event loop
        snapshot = await asyncio.to_thread(
            manager.wait_for_change, job_id, version, keepalive_s
        )
        if snapshot.version == version:
            yield ": keepalive\n\n"
            continue
        version = snapshot.version
        yield f"data: {_SNAPSHOT_JSON.dump_json(snapshot).decode()}\n\n"
        if snapshot.finished:
            return


router = APIRouter(prefix="/api")


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the server is up.

    Returns
    -------
    dict[str, str]
        ``{"status": "ok"}``.
    """
    return {"status": "ok"}


@router.get("/jobs/{job_id}")
def job_status(job_id: str, request: Request) -> JobSnapshot:
    """Return the current state of a job.

    Parameters
    ----------
    job_id : str
        Identifier of the job.
    request : fastapi.Request
        Incoming request, giving access to the job manager.

    Returns
    -------
    JobSnapshot
        Latest state of the job.

    Raises
    ------
    fastapi.HTTPException
        404 if the job is unknown.
    """
    return _snapshot_or_404(_job_manager(request), job_id)


@router.get("/jobs/{job_id}/events")
def job_event_stream(job_id: str, request: Request) -> StreamingResponse:
    """Stream the progress of a job as server-sent events.

    Parameters
    ----------
    job_id : str
        Identifier of the job.
    request : fastapi.Request
        Incoming request, giving access to the job manager.

    Returns
    -------
    fastapi.responses.StreamingResponse
        ``text/event-stream`` response fed by `job_events`.

    Raises
    ------
    fastapi.HTTPException
        404 if the job is unknown.
    """
    manager = _job_manager(request)
    _snapshot_or_404(manager, job_id)
    return StreamingResponse(
        job_events(manager, job_id),
        media_type=SSE_MEDIA_TYPE,
        headers={"Cache-Control": "no-cache"},
    )


def package_script_path(name: str) -> Path:
    """Return the file of a JavaScript library shipped by a Python package.

    Parameters
    ----------
    name : str
        Key of ``PACKAGE_SCRIPTS``.

    Returns
    -------
    pathlib.Path
        Path of the script in the installed package.

    Raises
    ------
    KeyError
        If `name` is not in ``PACKAGE_SCRIPTS``.
    """
    package, relative_path = PACKAGE_SCRIPTS[name]
    return Path(str(resources.files(package).joinpath(relative_path)))


assets_router = APIRouter()


@assets_router.get("/lib/{name}", include_in_schema=False)
def package_script(name: str) -> FileResponse:
    """Serve a JavaScript library shipped by a Python package.

    Parameters
    ----------
    name : str
        Key of ``PACKAGE_SCRIPTS``.

    Returns
    -------
    fastapi.responses.FileResponse
        The script.

    Raises
    ------
    fastapi.HTTPException
        404 if `name` is unknown.
    """
    try:
        path = package_script_path(name)
    except KeyError:
        raise HTTPException(status_code=404, detail="Unknown script") from None
    return FileResponse(path, media_type=JAVASCRIPT_MEDIA_TYPE, headers=REVALIDATE)


def create_app(
    job_manager: JobManager | None = None,
    static_dir: Path = STATIC_DIR,
    activity_service: ActivityService | None = None,
    dataset_service: DatasetService | None = None,
    heart_rate_service: HeartRateService | None = None,
) -> FastAPI:
    """Build the application: the ``/api`` routes and the static front end.

    Parameters
    ----------
    job_manager : JobManager or None, optional
        Manager of the background jobs. Default is None: the application
        creates its own and shuts it down when it stops.
    static_dir : pathlib.Path, optional
        Directory of the front end, whose ``index.html`` is served at ``/``.
        Default is ``STATIC_DIR``.
    activity_service : ActivityService or None, optional
        Catalogue and cache of the activities. Default is None: an
        `ActivityService` reading the data directory.
    dataset_service : DatasetService or None, optional
        Cache of the datasets and dataset chosen in the dataset page. Default
        is None: a `DatasetService` reading the data directory.
    heart_rate_service : HeartRateService or None, optional
        Saved and new heart-rate zone optimisations. Default is None: a
        `HeartRateService` on the models directory and `dataset_service`.

    Returns
    -------
    fastapi.FastAPI
        The application, to be served by uvicorn.

    Examples
    --------
    Development server with automatic reload::

        poetry run uvicorn TrailPredictor.api:create_app --factory --reload
    """
    manager = JobManager() if job_manager is None else job_manager

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        """Shut the job manager down when the application stops.

        Yields
        ------
        None
            Control back to the server while it runs.
        """
        yield
        if job_manager is None:  # owned by the application
            manager.shutdown(wait=False)

    app = FastAPI(title=APP_TITLE, lifespan=lifespan)
    app.state.job_manager = manager
    app.state.activity_service = (
        ActivityService() if activity_service is None else activity_service
    )
    app.state.dataset_service = (
        DatasetService() if dataset_service is None else dataset_service
    )
    app.state.heart_rate_service = (
        HeartRateService(app.state.dataset_service)
        if heart_rate_service is None
        else heart_rate_service
    )
    app.include_router(router)
    app.include_router(activity_routes.router)
    app.include_router(dataset_routes.router)
    app.include_router(heart_rate_routes.router)
    app.include_router(assets_router)
    # mounted last: the API routes take precedence over the static files
    app.mount(
        "/", RevalidatedStaticFiles(directory=static_dir, html=True), name="static"
    )
    return app

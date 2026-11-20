"""Routes of the activity page: selector, loading job, charts and 3D map."""

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, Request

from TrailPredictor.api.activity_service import (
    ActivityNotLoadedError,
    ActivityService,
    ActivitySummary,
    ChartsView,
    TrackView,
    XAxis,
    charts_view,
    track_view,
)
from TrailPredictor.api.jobs import JobCreated, JobManager
from TrailPredictor.loading import UnknownActivityError
from TrailPredictor.plot.track_3d import ColorBy, MapTexture

# Same bounds as the Streamlit sliders
SegmentWidth = Annotated[int, Query(ge=5, le=1000, multiple_of=5)]
DropThreshold = Annotated[float, Query(ge=0.5, le=10.0)]
DEFAULT_SEGMENT_WIDTH = 100  # m
DEFAULT_DROP_THRESHOLD = 2.0  # m

router = APIRouter(prefix="/api/activities", tags=["activities"])


def _service(request: Request) -> ActivityService:
    service: ActivityService = request.app.state.activity_service
    return service


def _check_known(service: ActivityService, activity_id: int) -> None:
    """Raise 503 without summary file, 404 for an unknown activity."""
    try:
        service.file_name(activity_id)
    except FileNotFoundError as error:
        raise HTTPException(503, f"Activity summary not found: {error}") from error
    except UnknownActivityError as error:
        raise HTTPException(404, str(error)) from error


@router.get("")
def list_activities(request: Request) -> list[ActivitySummary]:
    """List the running activities, longest first.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the activity service.

    Returns
    -------
    list of ActivitySummary
        Entries of the activity selector.

    Raises
    ------
    fastapi.HTTPException
        503 if the activity summary file does not exist.
    """
    try:
        return _service(request).summaries()
    except FileNotFoundError as error:
        raise HTTPException(503, f"Activity summary not found: {error}") from error


@router.post("/{activity_id}/load", status_code=202)
def load_activity(
    activity_id: int,
    request: Request,
    segment_width: SegmentWidth = DEFAULT_SEGMENT_WIDTH,
) -> JobCreated:
    """Start loading an activity in the background.

    Parameters
    ----------
    activity_id : int
        Strava identifier.
    request : fastapi.Request
        Incoming request, giving access to the services.
    segment_width : int, optional
        Segment length, in metres. Default is ``DEFAULT_SEGMENT_WIDTH``.

    Returns
    -------
    JobCreated
        Job to follow; the activity is cached when it succeeds.

    Raises
    ------
    fastapi.HTTPException
        503 without summary file, 404 if the activity is unknown.
    """
    service = _service(request)
    _check_known(service, activity_id)
    jobs: JobManager = request.app.state.job_manager
    job_id = jobs.submit(
        lambda report: service.load(activity_id, segment_width, report)
    )
    return JobCreated(job_id=job_id)


@router.get("/{activity_id}/charts")
def activity_charts(
    activity_id: int,
    request: Request,
    segment_width: SegmentWidth = DEFAULT_SEGMENT_WIDTH,
    drop_threshold: DropThreshold = DEFAULT_DROP_THRESHOLD,
    x_axis: XAxis = "distance",
) -> ChartsView:
    """Return the metrics and charts of a loaded activity.

    Parameters
    ----------
    activity_id : int
        Strava identifier.
    request : fastapi.Request
        Incoming request, giving access to the activity service.
    segment_width : int, optional
        Segment length, in metres. Default is ``DEFAULT_SEGMENT_WIDTH``.
    drop_threshold : float, optional
        Altitude change, in metres, above which a segment is uphill.
        Default is ``DEFAULT_DROP_THRESHOLD``.
    x_axis : {"distance", "time"}, optional
        Abscissa of the heart-rate chart. Default is ``"distance"``.

    Returns
    -------
    ChartsView
        Metrics and Plotly figures.

    Raises
    ------
    fastapi.HTTPException
        409 if the activity was not loaded with this segment width.
    """
    try:
        plot = _service(request).plot(activity_id, segment_width, drop_threshold)
    except ActivityNotLoadedError as error:
        raise HTTPException(409, str(error)) from error
    return charts_view(plot, x_axis)


@router.get("/{activity_id}/track")
def activity_track(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    activity_id: int,
    request: Request,
    segment_width: SegmentWidth = DEFAULT_SEGMENT_WIDTH,
    drop_threshold: DropThreshold = DEFAULT_DROP_THRESHOLD,
    color_by: ColorBy = "altitude",
    texture: MapTexture = "osm",
    show_km_markers: bool = True,
) -> TrackView:
    """Return the 3D map of the GPS track of a loaded activity.

    Parameters
    ----------
    activity_id : int
        Strava identifier.
    request : fastapi.Request
        Incoming request, giving access to the activity service.
    segment_width : int, optional
        Segment length, in metres. Default is ``DEFAULT_SEGMENT_WIDTH``.
    drop_threshold : float, optional
        Altitude change, in metres, above which a segment is uphill.
        Default is ``DEFAULT_DROP_THRESHOLD``.
    color_by : {"altitude", "heart_rate", "pace"}, optional
        Variable colouring the track. Default is ``"altitude"``.
    texture : {"osm", "satellite"}, optional
        Imagery draped over the terrain. Default is ``"osm"``.
    show_km_markers : bool, optional
        Add a marker at each kilometre. Default is True.

    Returns
    -------
    TrackView
        deck.gl JSON and colour legend.

    Raises
    ------
    fastapi.HTTPException
        409 if the activity was not loaded with this segment width, 422 if
        it has no valid GPS coordinates.
    """
    try:
        plot = _service(request).plot(activity_id, segment_width, drop_threshold)
        return track_view(plot, color_by, texture, show_km_markers)
    except ActivityNotLoadedError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, str(error)) from error

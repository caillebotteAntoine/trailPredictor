"""Routes of the dataset page: loading, feature table, presets and charts."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from TrailPredictor.api.dataset_service import (
    DatasetNotLoadedError,
    DatasetParams,
    DatasetSelection,
    DatasetService,
    FeatureFilters,
    FeaturesView,
    SavedZones,
    dataset_chart,
    features_view,
    preset_features,
)
from TrailPredictor.api.figures import ChartView
from TrailPredictor.api.jobs import JobCreated, JobManager
from TrailPredictor.feature_selection import PERCENT, FeatureGroup, Preset
from TrailPredictor.modeling.heart_rate import cuts_to_zones, parse_cuts
from TrailPredictor.speed_columns import Target

# Same bounds and defaults as the Streamlit inputs
DEFAULT_MIN_DISTANCE, DEFAULT_MAX_DISTANCE = 40, 60  # km
DEFAULT_SEGMENT_WIDTH = 1000  # m
DEFAULT_TARGET: Target = "pace"

router = APIRouter(prefix="/api/dataset", tags=["dataset"])


def dataset_params(
    min_distance: Annotated[int, Query(ge=1, le=200)] = DEFAULT_MIN_DISTANCE,
    max_distance: Annotated[int, Query(ge=2, le=300)] = DEFAULT_MAX_DISTANCE,
    segment_width: Annotated[
        int, Query(ge=100, le=5000, multiple_of=100)
    ] = DEFAULT_SEGMENT_WIDTH,
) -> DatasetParams:
    """Read the parameters identifying a dataset from the query string.

    Parameters
    ----------
    min_distance : int, optional
        Minimum activity distance, in km. Default is ``DEFAULT_MIN_DISTANCE``.
    max_distance : int, optional
        Maximum activity distance, in km. Default is ``DEFAULT_MAX_DISTANCE``.
    segment_width : int, optional
        Segment length, in metres. Default is ``DEFAULT_SEGMENT_WIDTH``.

    Returns
    -------
    DatasetParams
        The validated parameters.

    Raises
    ------
    fastapi.HTTPException
        422 if `min_distance` is not less than `max_distance`.
    """
    if min_distance >= max_distance:
        raise HTTPException(422, "min_distance must be less than max_distance")
    return DatasetParams(min_distance, max_distance, segment_width)


def feature_filters(
    search: str = "",
    groups: Annotated[list[FeatureGroup] | None, Query()] = None,
    min_abs: Annotated[float, Query(ge=0.0, le=1.0)] = 0.0,
    max_missing: Annotated[float, Query(ge=0.0, le=PERCENT)] = PERCENT,
) -> FeatureFilters:
    """Read the filters of the feature table from the query string.

    Parameters
    ----------
    search : str, optional
        Case-insensitive substring of the feature name. Default is ``""``.
    groups : list of FeatureGroup or None, optional
        Allowed groups, repeated in the query; all groups when omitted.
    min_abs : float, optional
        Minimum absolute Spearman correlation. Default is 0.
    max_missing : float, optional
        Maximum percentage of missing values. Default is 100.

    Returns
    -------
    FeatureFilters
        The validated filters.
    """
    return FeatureFilters(
        search=search, groups=groups or [], min_abs=min_abs, max_missing=max_missing
    )


def heart_rate_cuts(cuts: str) -> list[int]:
    """Parse the heart-rate zone cuts typed by the user.

    Parameters
    ----------
    cuts : str
        Increasing cuts, in bpm, separated by commas, semicolons or spaces.

    Returns
    -------
    list of int
        The cuts.

    Raises
    ------
    fastapi.HTTPException
        422 if the cuts are invalid.
    """
    try:
        parsed = parse_cuts(cuts)
        cuts_to_zones(parsed)  # checks the order and the bounds
    except ValueError as error:
        raise HTTPException(422, f"Invalid cuts: {error}") from error
    return parsed


Params = Annotated[DatasetParams, Depends(dataset_params)]
Filters = Annotated[FeatureFilters, Depends(feature_filters)]
Cuts = Annotated[list[int], Depends(heart_rate_cuts)]


def _service(request: Request) -> DatasetService:
    service: DatasetService = request.app.state.dataset_service
    return service


@router.post("/load", status_code=202)
def load_dataset(request: Request, params: Params) -> JobCreated:
    """Start loading a dataset in the background.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the services.
    params : DatasetParams
        Distance range and segment width.

    Returns
    -------
    JobCreated
        Job to follow; the dataset is cached when it succeeds.
    """
    service = _service(request)
    jobs: JobManager = request.app.state.job_manager
    job_id = jobs.submit(lambda report: service.load(params, report))
    return JobCreated(job_id=job_id)


@router.get("/features")
def dataset_features(
    request: Request,
    params: Params,
    filters: Filters,
    target: Target = DEFAULT_TARGET,
) -> FeaturesView:
    """Return the feature table of a loaded dataset.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.
    params : DatasetParams
        Distance range and segment width.
    filters : FeatureFilters
        Filters of the table.
    target : Target, optional
        Target the features are correlated with. Default is
        ``DEFAULT_TARGET``.

    Returns
    -------
    FeaturesView
        Filtered table and its context.

    Raises
    ------
    fastapi.HTTPException
        409 if the dataset was not loaded.
    """
    service = _service(request)
    try:
        return features_view(
            service.dataset(params, target),
            service.correlations(params, target),
            filters,
        )
    except DatasetNotLoadedError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/preset")
def dataset_preset(
    request: Request,
    params: Params,
    filters: Filters,
    preset: Preset,
    target: Target = DEFAULT_TARGET,
) -> list[str]:
    """Return the features of a preset, among the filtered features.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.
    params : DatasetParams
        Distance range and segment width.
    filters : FeatureFilters
        Filters applied before the preset.
    preset : Preset
        Predefined selection.
    target : Target, optional
        Target the features are ranked with. Default is ``DEFAULT_TARGET``.

    Returns
    -------
    list of str
        Names of the features of the preset.

    Raises
    ------
    fastapi.HTTPException
        409 if the dataset was not loaded.
    """
    try:
        correlations = _service(request).correlations(params, target)
    except DatasetNotLoadedError as error:
        raise HTTPException(409, str(error)) from error
    return preset_features(correlations, preset, filters)


@router.get("/chart")
def dataset_chart_view(
    request: Request,
    params: Params,
    cuts: Cuts,
    target: Target = DEFAULT_TARGET,
    feature: str | None = None,
) -> ChartView:
    """Return a chart of a loaded dataset, drawn for every target.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.
    params : DatasetParams
        Distance range and segment width.
    cuts : list of int
        Heart-rate zone cuts, in bpm.
    target : Target, optional
        Current target. Default is ``DEFAULT_TARGET``.
    feature : str or None, optional
        Feature of a target-vs-feature chart; the target histogram when
        omitted.

    Returns
    -------
    ChartView
        The figure, or the error that prevented it.

    Raises
    ------
    fastapi.HTTPException
        409 if the dataset was not loaded, 422 if the cuts are invalid.
    """
    try:
        dataset = _service(request).dataset(params, target, cuts)
    except DatasetNotLoadedError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, f"Invalid cuts: {error}") from error
    return dataset_chart(dataset, feature)


@router.get("/cuts")
def parsed_cuts(cuts: Cuts) -> list[int]:
    """Parse and check the heart-rate zone cuts typed by the user.

    Parameters
    ----------
    cuts : list of int
        Cuts parsed from the ``cuts`` query parameter.

    Returns
    -------
    list of int
        The cuts, in bpm.
    """
    return cuts


@router.get("/saved-zones")
def dataset_saved_zones(request: Request) -> list[SavedZones]:
    """List the zones of the saved heart-rate optimisations.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.

    Returns
    -------
    list of SavedZones
        Zones offered in addition to the typed cuts.
    """
    return _service(request).saved_zones()


@router.get("/current")
def current_dataset(request: Request) -> DatasetSelection | None:
    """Return the dataset chosen in the dataset page.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.

    Returns
    -------
    DatasetSelection or None
        Last selection, or None if no dataset was chosen yet.
    """
    return _service(request).current


@router.put("/current")
def select_dataset(request: Request, selection: DatasetSelection) -> DatasetSelection:
    """Choose the dataset used by the other pages.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the dataset service.
    selection : DatasetSelection
        Loaded dataset, target and zones.

    Returns
    -------
    DatasetSelection
        The selection, as stored.

    Raises
    ------
    fastapi.HTTPException
        409 if the dataset was not loaded, 422 if the cuts are invalid.
    """
    try:
        _service(request).select(selection)
    except DatasetNotLoadedError as error:
        raise HTTPException(409, str(error)) from error
    except ValueError as error:
        raise HTTPException(422, f"Invalid cuts: {error}") from error
    return selection

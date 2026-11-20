"""Routes of the heart-rate zones page: saved results and optimisations."""

from typing import Annotated

import pandas as pd
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel

from TrailPredictor.api.figures import ChartView
from TrailPredictor.api.heart_rate_service import (
    HeartRateService,
    NoCurrentDatasetError,
    OptimizationStarted,
    PageContext,
    ResultView,
    SavedResultView,
    UnknownResultError,
    result_view,
    saved_result_view,
    trials_chart,
)
from TrailPredictor.api.jobs import JobManager
from TrailPredictor.modeling.heart_rate import (
    MIN_ZONES,
    HeartRateConfig,
    SavedHeartRateResult,
)
from TrailPredictor.speed_columns import Target

router = APIRouter(prefix="/api/heart-rate", tags=["heart-rate"])

ResultKey = Annotated[str, Query(min_length=1)]


class OptimizationRequest(BaseModel):
    """Body of the optimisation route.

    Attributes
    ----------
    target : Target
        Target the RMSE is computed on.
    config : HeartRateConfig
        Search space and budget; validated, unknown fields rejected.
    """

    target: Target
    config: HeartRateConfig


def _service(request: Request) -> HeartRateService:
    service: HeartRateService = request.app.state.heart_rate_service
    return service


def _saved_and_trials(
    service: HeartRateService, key: str
) -> tuple[SavedHeartRateResult, pd.DataFrame]:
    """Return a saved optimisation and its trials, or raise the HTTP error."""
    try:
        saved = service.saved_result(key)
        return saved, saved.load_trials()
    except UnknownResultError as error:
        raise HTTPException(404, f"Unknown result: {error}") from error
    except (OSError, pd.errors.ParserError) as error:
        raise HTTPException(422, f"Unreadable trials: {error}") from error


@router.get("/context")
def page_context(request: Request) -> PageContext:
    """Return the current dataset and the bounds of the optimisation form.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the heart-rate service.

    Returns
    -------
    PageContext
        Current dataset, form defaults and bounds.
    """
    return _service(request).context()


@router.get("/saved")
def saved_optimizations(request: Request) -> list[SavedResultView]:
    """List the saved optimisations.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the heart-rate service.

    Returns
    -------
    list of SavedResultView
        Optimisations found in the models directory.
    """
    return [saved_result_view(saved) for saved in _service(request).saved_results()]


@router.post("/optimize", status_code=202)
def optimize(request: Request, body: OptimizationRequest) -> OptimizationStarted:
    """Run (or reload) an optimisation on the current dataset in a job.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the services.
    body : OptimizationRequest
        Target and configuration.

    Returns
    -------
    OptimizationStarted
        Job to follow and key of the result.

    Raises
    ------
    fastapi.HTTPException
        409 if no dataset was chosen in the dataset page.
    """
    jobs: JobManager = request.app.state.job_manager
    try:
        return _service(request).start_optimization(
            body.target, body.config, jobs.submit
        )
    except NoCurrentDatasetError as error:
        raise HTTPException(409, str(error)) from error


@router.get("/result")
def optimization_result(request: Request, key: ResultKey) -> ResultView:
    """Return the metrics, table and main chart of a saved optimisation.

    The reference RMSE is computed on the dataset of the dataset page, if
    any, which can take a few seconds the first time.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the heart-rate service.
    key : str
        Key of the optimisation, from the saved list.

    Returns
    -------
    ResultView
        The view of the optimisation.

    Raises
    ------
    fastapi.HTTPException
        404 if the key is unknown, 422 if its trials cannot be read.
    """
    service = _service(request)
    saved, trials = _saved_and_trials(service, key)
    try:
        return result_view(saved, trials, service.reference(saved))
    except ValueError as error:
        raise HTTPException(422, f"Invalid trials: {error}") from error


@router.get("/trials-chart")
def optimization_trials_chart(
    request: Request,
    key: ResultKey,
    n_zones: Annotated[int, Query(ge=MIN_ZONES)],
) -> ChartView:
    """Return the chart of every trial of a number of zones.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request, giving access to the heart-rate service.
    key : str
        Key of the optimisation, from the saved list.
    n_zones : int
        Number of zones to show.

    Returns
    -------
    ChartView
        The figure, or the error that prevented it.

    Raises
    ------
    fastapi.HTTPException
        404 if the key is unknown, 422 if its trials cannot be read.
    """
    saved, trials = _saved_and_trials(_service(request), key)
    try:
        return trials_chart(saved, trials, n_zones)
    except ValueError as error:
        raise HTTPException(422, f"Invalid trials: {error}") from error

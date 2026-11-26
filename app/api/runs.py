from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy import func, select

from app.api.crud import SessionDep
from app.api.schemas import (
    ComparisonRow,
    GenerationRead,
    ResultsRead,
    RunCreate,
    RunDetail,
    RunSummary,
    SolutionDetail,
    SolutionSummary,
    TrajectoriesRead,
    TrajectoryScenario,
)
from app.application import runs_service
from app.infrastructure.executor import Dispatcher
from app.infrastructure.models import Generation, Run, Solution

router = APIRouter(prefix="/runs", tags=["runs"])

ProfileName = Literal[
    "pareto",
    "best_detection",
    "lowest_consumption",
    "balanced",
    "grid",
    "random_best_detection",
    "random_lowest_consumption",
    "random_balanced",
]
GA_PROFILES = ("best_detection", "lowest_consumption", "balanced")
BASELINES = (
    "grid",
    "random_best_detection",
    "random_lowest_consumption",
    "random_balanced",
)


def get_dispatcher(request: Request) -> Dispatcher:
    return request.app.state.dispatcher


DispatcherDep = Annotated[Dispatcher, Depends(get_dispatcher)]


def _fetch(session, run_id: int) -> Run:
    run = session.get(Run, run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Corrida no encontrada")
    return run


def _require_completed(run: Run) -> None:
    if run.status != "completed":
        raise HTTPException(status_code=409, detail="La corrida no ha terminado")


@router.post("", response_model=RunSummary, status_code=202)
def create_run(payload: RunCreate, session: SessionDep, dispatcher: DispatcherDep):
    run = runs_service.create_run(session, payload)
    dispatcher.submit(run.id)
    session.refresh(run)
    return run


@router.get("", response_model=list[RunSummary])
def list_runs(session: SessionDep, status: str | None = None):
    query = select(Run).order_by(Run.id.desc())
    if status is not None:
        query = query.where(Run.status == status)
    return session.scalars(query).all()


@router.get("/{run_id}", response_model=RunDetail)
def get_run(run_id: int, session: SessionDep):
    run = _fetch(session, run_id)
    latest = session.scalar(
        select(Generation)
        .where(Generation.run_id == run_id)
        .order_by(Generation.gen.desc())
        .limit(1)
    )
    count = session.scalar(select(func.count(Solution.id)).where(Solution.run_id == run_id))
    summary = RunSummary.model_validate(run).model_dump()
    return RunDetail(
        **summary,
        params=run.params,
        latest_generation=GenerationRead.model_validate(latest) if latest else None,
        solution_count=count or 0,
    )


@router.post("/{run_id}/cancel", response_model=RunSummary)
def cancel_run(run_id: int, session: SessionDep):
    return runs_service.request_cancel(session, _fetch(session, run_id))


@router.delete("/{run_id}", status_code=204)
def delete_run(run_id: int, session: SessionDep):
    runs_service.delete_run(session, _fetch(session, run_id))


@router.get("/{run_id}/generations", response_model=list[GenerationRead])
def list_generations(run_id: int, session: SessionDep):
    _fetch(session, run_id)
    return session.scalars(
        select(Generation).where(Generation.run_id == run_id).order_by(Generation.gen)
    ).all()


@router.get("/{run_id}/results", response_model=ResultsRead)
def get_results(run_id: int, session: SessionDep):
    run = _fetch(session, run_id)
    _require_completed(run)
    solutions = session.scalars(
        select(Solution).where(Solution.run_id == run_id).order_by(Solution.id)
    ).all()
    by_profile: dict[str, list[Solution]] = {}
    for solution in solutions:
        by_profile.setdefault(solution.profile, []).append(solution)

    def first(name: str) -> SolutionSummary | None:
        items = by_profile.get(name)
        return SolutionSummary.model_validate(items[0]) if items else None

    profiles = {name: first(name) for name in GA_PROFILES if first(name)}
    baselines = {name: first(name) for name in BASELINES if first(name)}
    comparison = [
        ComparisonRow(
            method=("ga_" + name if name in GA_PROFILES else name),
            feasible=summary.feasible,
            r_de=summary.r_de,
            t_mission_min=summary.t_mission_min,
            e_total_wh=summary.e_total_wh,
        )
        for name, summary in {**profiles, **baselines}.items()
    ]
    return ResultsRead(
        run_id=run_id,
        profiles=profiles,
        pareto=[SolutionSummary.model_validate(s) for s in by_profile.get("pareto", [])],
        baselines=baselines,
        comparison=comparison,
    )


@router.get("/{run_id}/solutions/{solution_id}", response_model=SolutionDetail)
def get_solution(run_id: int, solution_id: int, session: SessionDep):
    solution = session.scalar(
        select(Solution).where(Solution.run_id == run_id, Solution.id == solution_id)
    )
    if solution is None:
        raise HTTPException(status_code=404, detail="Solución no encontrada")
    return solution


@router.get("/{run_id}/trajectories", response_model=TrajectoriesRead)
def get_trajectories(run_id: int, session: SessionDep, profile: ProfileName = "balanced"):
    run = _fetch(session, run_id)
    _require_completed(run)
    solution = session.scalar(
        select(Solution)
        .where(Solution.run_id == run_id, Solution.profile == profile)
        .order_by(Solution.id)
        .limit(1)
    )
    if solution is None:
        raise HTTPException(status_code=404, detail="La corrida no tiene ese perfil")
    config = run.scenario_snapshot["config"]
    scenario = TrajectoryScenario(
        width_m=config["width_m"],
        height_m=config["height_m"],
        cell_size_m=config["cell_size_m"],
        base=config["base"],
        last_seen=config.get("last_seen"),
        forbidden_zones=config.get("forbidden_zones", []),
    )
    return TrajectoriesRead(
        profile=profile, scenario=scenario, solution=SolutionDetail.model_validate(solution)
    )

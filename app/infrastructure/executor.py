import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.application.runs_service import load_inputs
from app.domain.baselines import grid_solution, random_front
from app.domain.profiles import select_profiles
from app.domain.simulator import Evaluator
from app.domain.solution import describe_solution
from app.domain.solve import GenerationStats, optimize
from app.infrastructure.db import SessionLocal
from app.infrastructure.models import Generation, Run, Solution

SEARCH_PROGRESS_SHARE = 0.9


class Dispatcher(Protocol):
    def submit(self, run_id: int) -> None: ...

    def shutdown(self) -> None: ...


def _now() -> datetime:
    return datetime.now(UTC)


def run_job(run_id: int, session_factory: Callable[[], Session]) -> None:
    with session_factory() as session:
        run = session.get(Run, run_id)
        if run is None or run.status != "queued":
            return
        run.status = "running"
        run.started_at = _now()
        session.commit()
        try:
            _execute(session, run)
        except Exception as exc:
            session.rollback()
            run = session.get(Run, run_id)
            run.status = "failed"
            run.error = f"{type(exc).__name__}: {exc}"[:2000]
            run.finished_at = _now()
            session.commit()


def _execute(session: Session, run: Run) -> None:
    terrain, drones, cameras, person, params = load_inputs(run)
    evaluator = Evaluator(terrain, drones, cameras, person, params)

    def on_generation(stats: GenerationStats) -> None:
        session.add(
            Generation(
                run_id=run.id,
                gen=stats.gen,
                n_feasible=stats.n_feasible,
                best_r_de=stats.best_r_de,
                best_t_mission_min=stats.best_t_mission_min,
                best_e_total_wh=stats.best_e_total_wh,
                hypervolume=stats.hypervolume,
            )
        )
        run.progress = SEARCH_PROGRESS_SHARE * stats.gen / params.generations
        session.commit()

    def should_stop() -> bool:
        return bool(session.scalar(select(Run.cancel_requested).where(Run.id == run.id)))

    result = optimize(evaluator, on_generation, should_stop)
    if result is None:
        run.status = "cancelled"
        run.finished_at = _now()
        session.commit()
        return

    _persist(session, run, evaluator, result)
    run.status = "completed"
    run.progress = 1.0
    run.finished_at = _now()
    session.commit()


def _solution(run_id: int, profile: str, evaluator: Evaluator, x) -> Solution:
    metrics, route = describe_solution(evaluator, x)
    return Solution(
        run_id=run_id,
        profile=profile,
        feasible=metrics.feasible,
        constraint_violation=metrics.constraint_violation,
        r_de=metrics.r_de,
        t_mission_min=metrics.t_mission_min,
        e_total_wh=metrics.e_total_wh,
        area_explored_pct=metrics.area_explored_pct,
        area_detected_pct=metrics.area_detected_pct,
        chromosome=[float(v) for v in x],
        route=route,
    )


def _persist(session: Session, run: Run, evaluator: Evaluator, result) -> None:
    feasible = result.feasible
    X, F = (result.X[feasible], result.F[feasible]) if feasible.any() else (result.X, result.F)

    for x in X:
        session.add(_solution(run.id, "pareto", evaluator, x))
    if feasible.any():
        for name, index in select_profiles(F).items():
            session.add(_solution(run.id, name, evaluator, X[index]))

    session.add(_solution(run.id, "grid", evaluator, grid_solution(evaluator)))

    if evaluator.params.random_baseline_evals > 0:
        random_x, random_f = random_front(
            evaluator, evaluator.params.random_baseline_evals, evaluator.params.seed + 1
        )
        for name, index in select_profiles(random_f).items():
            session.add(_solution(run.id, f"random_{name}", evaluator, random_x[index]))
    session.commit()


def execute_run(run_id: int) -> None:
    run_job(run_id, SessionLocal)


class ProcessDispatcher:
    def __init__(self, workers: int) -> None:
        self._workers = workers
        self._pool: ProcessPoolExecutor | None = None

    def submit(self, run_id: int) -> None:
        if self._pool is None:
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers, mp_context=multiprocessing.get_context("spawn")
            )
        self._pool.submit(execute_run, run_id)

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)


class InlineDispatcher:
    def __init__(self, session_factory: Callable[[], Session]) -> None:
        self._session_factory = session_factory

    def submit(self, run_id: int) -> None:
        run_job(run_id, self._session_factory)

    def shutdown(self) -> None:
        return None

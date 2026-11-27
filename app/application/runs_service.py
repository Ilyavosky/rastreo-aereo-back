import secrets
from datetime import UTC, datetime
from typing import Any

from fastapi import HTTPException
from sqlalchemy import select, update
from sqlalchemy.orm import Session

from app.config import settings, visibility_factors
from app.domain.terrain import Terrain, build_terrain
from app.domain.types import CameraSpec, DroneSpec, PersonSpec, SolverParams
from app.infrastructure.models import Camera, Drone, Person, Run, Scenario

ACTIVE_STATUSES = ("queued", "running")


def _row_dict(row: Any) -> dict[str, Any]:
    return {column.name: getattr(row, column.name) for column in row.__table__.columns}


def _pick(value, default):
    return default if value is None else value


def _fail(message: str) -> HTTPException:
    return HTTPException(status_code=422, detail=message)


def _unique(ids: list[int]) -> list[int]:
    return list(dict.fromkeys(ids))


def _check_limit(name: str, value: int, limit: int | None) -> None:
    if limit is not None and value > limit:
        raise _fail(f"{name} no puede ser mayor que {limit}")


def resolve_params(payload) -> dict[str, Any]:
    n_drones = _pick(payload.n_drones, settings.default_n_drones)
    n_points = _pick(payload.n_points, settings.default_n_points)
    population = _pick(payload.population_size, settings.default_population_size)
    generations = _pick(payload.generations, settings.default_generations)
    factor = _pick(payload.random_baseline_factor, settings.random_baseline_factor)

    _check_limit("n_drones", n_drones, settings.limit_max_drones)
    _check_limit("n_points", n_points, settings.limit_max_points)
    _check_limit("population_size", population, settings.limit_max_population)
    _check_limit("generations", generations, settings.limit_max_generations)

    return {
        "n_drones": n_drones,
        "n_points": n_points,
        "population_size": population,
        "generations": generations,
        "crossover_prob": _pick(payload.crossover_prob, settings.default_crossover_prob),
        "crossover_eta": _pick(payload.crossover_eta, settings.default_crossover_eta),
        "mutation_sigma": _pick(payload.mutation_sigma, settings.default_mutation_sigma),
        "seed": _pick(payload.seed, secrets.randbelow(2**31)),
        "battery_reserve": _pick(payload.battery_reserve, settings.battery_reserve),
        "min_separation_m": _pick(payload.min_separation_m, settings.min_separation_m),
        "base_clearance_m": settings.base_clearance_m,
        "separation_sampling_s": settings.separation_sampling_s,
        "separation_max_samples": settings.separation_max_samples,
        "random_baseline_evals": int(factor * population * generations),
    }


def _load_options(session: Session, model, ids: list[int], label: str) -> list[dict[str, Any]]:
    rows = {row.id: row for row in session.scalars(select(model).where(model.id.in_(ids)))}
    missing = [i for i in ids if i not in rows]
    if missing:
        raise _fail(f"No existen {label} con id {missing}")
    inactive = [i for i in ids if not rows[i].active]
    if inactive:
        raise _fail(f"Los {label} con id {inactive} están inactivos")
    return [_row_dict(rows[i]) for i in ids]


def create_run(session: Session, payload) -> Run:
    scenario = session.get(Scenario, payload.scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Escenario no encontrado")
    person = session.get(Person, payload.person_id)
    if person is None:
        raise HTTPException(status_code=404, detail="Persona no encontrada")

    params = resolve_params(payload)
    drones = _load_options(session, Drone, _unique(payload.drone_ids), "drones")
    cameras = _load_options(session, Camera, _unique(payload.camera_ids), "cámaras")

    run = Run(
        scenario_id=scenario.id,
        status="queued",
        cancel_requested=False,
        progress=0.0,
        params=params,
        scenario_snapshot={"config": scenario.config, "visibility": visibility_factors()},
        catalog_snapshot={"drones": drones, "cameras": cameras, "person": _row_dict(person)},
    )
    session.add(run)
    session.commit()
    session.refresh(run)
    return run


def load_inputs(
    run: Run,
) -> tuple[Terrain, list[DroneSpec], list[CameraSpec], PersonSpec, SolverParams]:
    snapshot = run.scenario_snapshot
    terrain = build_terrain(snapshot["config"], snapshot["visibility"])
    catalog = run.catalog_snapshot
    return (
        terrain,
        [DroneSpec.from_dict(d) for d in catalog["drones"]],
        [CameraSpec.from_dict(c) for c in catalog["cameras"]],
        PersonSpec.from_dict(catalog["person"]),
        SolverParams.from_dict(run.params),
    )


def request_cancel(session: Session, run: Run) -> Run:
    if run.status not in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="La corrida ya terminó")
    if run.status == "queued":
        run.status = "cancelled"
        run.finished_at = datetime.now(UTC)
    run.cancel_requested = True
    session.commit()
    session.refresh(run)
    return run


def delete_run(session: Session, run: Run) -> None:
    if run.status in ACTIVE_STATUSES:
        raise HTTPException(status_code=409, detail="Cancele la corrida antes de eliminarla")
    session.delete(run)
    session.commit()


def fail_interrupted_runs(session: Session) -> None:
    session.execute(
        update(Run)
        .where(Run.status.in_(ACTIVE_STATUSES))
        .values(
            status="failed",
            error="Interrumpida por el reinicio del servidor",
            finished_at=datetime.now(UTC),
        )
    )
    session.commit()


def request_cancel_all_active(session: Session) -> None:
    session.execute(
        update(Run).where(Run.status.in_(ACTIVE_STATUSES)).values(cancel_requested=True)
    )
    session.commit()

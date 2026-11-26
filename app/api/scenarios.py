import numpy as np
from fastapi import HTTPException

from app.api.crud import SessionDep, build_crud_router
from app.api.schemas import PriorityMapRead, ScenarioCreate, ScenarioRead
from app.config import settings, visibility_factors
from app.domain.terrain import build_terrain, priority_map
from app.infrastructure.models import Scenario

router = build_crud_router(
    prefix="/scenarios",
    tag="scenarios",
    model=Scenario,
    create_schema=ScenarioCreate,
    read_schema=ScenarioRead,
)


@router.get("/{scenario_id}/priority-map", response_model=PriorityMapRead)
def get_priority_map(scenario_id: int, session: SessionDep):
    scenario = session.get(Scenario, scenario_id)
    if scenario is None:
        raise HTTPException(status_code=404, detail="Recurso no encontrado")
    terrain = build_terrain(scenario.config, visibility_factors())
    grid = priority_map(
        terrain,
        scenario.config,
        settings.priority_sigma_m,
        settings.priority_path_weight,
        settings.priority_path_width_m,
    )
    return PriorityMapRead(
        rows=terrain.n_rows,
        cols=terrain.n_cols,
        cell_size_m=terrain.cell_size_m,
        values=np.round(grid, 8).tolist(),
    )

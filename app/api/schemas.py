import math
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings


class DroneCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    battery_wh: float = Field(gt=0)
    v_min: float = Field(gt=0)
    v_max: float = Field(gt=0)
    h_min: float = Field(gt=0)
    h_max: float = Field(gt=0)
    v_vertical: float = Field(gt=0)
    p_base_w: float = Field(gt=0)
    c_drag: float = Field(ge=0)
    c_altitude_j_m: float = Field(ge=0)
    active: bool = True

    @model_validator(mode="after")
    def check_ranges(self):
        if self.v_max <= self.v_min:
            raise ValueError("v_max debe ser mayor que v_min")
        if self.h_max <= self.h_min:
            raise ValueError("h_max debe ser mayor que h_min")
        return self


class DroneRead(DroneCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int


class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    modality: Literal["optical", "thermal"]
    fov_deg: float = Field(gt=0, lt=180)
    n_px_width: int = Field(ge=1)
    power_w: float = Field(ge=0)
    battery_wh: float = Field(gt=0)
    max_range_m: float = Field(gt=0)
    active: bool = True


class CameraRead(CameraCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int


class PersonCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    posture: Literal["standing", "sitting", "lying"]
    critical_dim_m: float = Field(gt=0)
    px_min_optical: int = Field(ge=1)
    px_min_thermal: int = Field(ge=1)


class PersonRead(PersonCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int


class Point(BaseModel):
    x: float
    y: float


class VegetationZone(BaseModel):
    type: Literal["open", "brush", "dense"]
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def check_corners(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("La zona debe cumplir x1 > x0 y y1 > y0")
        return self


class ForbiddenZone(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def check_corners(self):
        if self.x1 <= self.x0 or self.y1 <= self.y0:
            raise ValueError("La zona debe cumplir x1 > x0 y y1 > y0")
        return self


class HillSpec(BaseModel):
    x: float
    y: float
    height_m: float = Field(ge=0)
    sigma_m: float = Field(gt=0)


class PathLine(BaseModel):
    kind: Literal["trail", "river"]
    points: list[Point] = Field(min_length=2)


class ScenarioConfig(BaseModel):
    width_m: float = Field(gt=0)
    height_m: float = Field(gt=0)
    cell_size_m: float = Field(gt=0)
    base: Point
    last_seen: Point | None = None
    vegetation_default: Literal["open", "brush", "dense"] = "open"
    vegetation_zones: list[VegetationZone] = []
    forbidden_zones: list[ForbiddenZone] = []
    hills: list[HillSpec] = []
    paths: list[PathLine] = []

    @model_validator(mode="after")
    def check_geometry(self):
        n_cells = math.ceil(self.width_m / self.cell_size_m) * math.ceil(
            self.height_m / self.cell_size_m
        )
        if n_cells > settings.limit_max_grid_cells:
            raise ValueError(
                f"La cuadrícula tiene {n_cells} celdas y el máximo permitido es "
                f"{settings.limit_max_grid_cells}"
            )
        points = [self.base] + ([self.last_seen] if self.last_seen else [])
        points += [p for path in self.paths for p in path.points]
        points += [Point(x=h.x, y=h.y) for h in self.hills]
        for p in points:
            if not (0 <= p.x <= self.width_m and 0 <= p.y <= self.height_m):
                raise ValueError("Todos los puntos deben estar dentro del mapa")
        for zone in self.forbidden_zones:
            if zone.x0 < self.base.x < zone.x1 and zone.y0 < self.base.y < zone.y1:
                raise ValueError("La base no puede estar dentro de una zona prohibida")
        return self


class ScenarioCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    config: ScenarioConfig


class ScenarioRead(ScenarioCreate):
    model_config = ConfigDict(from_attributes=True)

    id: int
    created_at: datetime


class TemplateRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    slug: str
    name: str
    description: str
    config: ScenarioConfig


class RunCreate(BaseModel):
    scenario_id: int
    person_id: int
    drone_ids: list[int] = Field(min_length=1)
    camera_ids: list[int] = Field(min_length=1)
    n_drones: int | None = Field(default=None, gt=0)
    n_points: int | None = Field(default=None, gt=0)
    population_size: int | None = Field(default=None, gt=1)
    generations: int | None = Field(default=None, gt=0)
    crossover_prob: float | None = Field(default=None, gt=0, le=1)
    crossover_eta: float | None = Field(default=None, gt=0)
    mutation_sigma: float | None = Field(default=None, gt=0, lt=1)
    seed: int | None = Field(default=None, ge=0)
    battery_reserve: float | None = Field(default=None, ge=0, lt=1)
    min_separation_m: float | None = Field(default=None, ge=0)
    random_baseline_factor: float | None = Field(default=None, ge=0)


class RunSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    scenario_id: int | None
    status: str
    progress: float
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None


class GenerationRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    gen: int
    n_feasible: int
    best_r_de: float | None
    best_t_mission_min: float | None
    best_e_total_wh: float | None
    hypervolume: float | None


class RunDetail(RunSummary):
    params: dict
    latest_generation: GenerationRead | None
    solution_count: int


class SolutionSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    profile: str
    feasible: bool
    constraint_violation: float
    r_de: float
    t_mission_min: float
    e_total_wh: float
    area_explored_pct: float
    area_detected_pct: float


class SolutionDetail(SolutionSummary):
    chromosome: list[float]
    route: dict


class ComparisonRow(BaseModel):
    method: str
    feasible: bool
    r_de: float
    t_mission_min: float
    e_total_wh: float


class ResultsRead(BaseModel):
    run_id: int
    profiles: dict[str, SolutionSummary]
    pareto: list[SolutionSummary]
    baselines: dict[str, SolutionSummary]
    comparison: list[ComparisonRow]


class TrajectoryScenario(BaseModel):
    width_m: float
    height_m: float
    cell_size_m: float
    base: Point
    last_seen: Point | None
    forbidden_zones: list[ForbiddenZone]


class TrajectoriesRead(BaseModel):
    profile: str
    scenario: TrajectoryScenario
    solution: SolutionDetail


class PriorityMapRead(BaseModel):
    rows: int
    cols: int
    cell_size_m: float
    values: list[list[float]]

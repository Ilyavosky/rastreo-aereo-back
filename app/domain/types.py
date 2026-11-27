import math
from dataclasses import dataclass, fields
from typing import Any


def _pick(cls, data: dict[str, Any]):
    names = {f.name for f in fields(cls)}
    return cls(**{key: value for key, value in data.items() if key in names})


def fov_from_optics(focal_length_mm: float, sensor_width_mm: float) -> float:
    return math.degrees(2.0 * math.atan(sensor_width_mm / (2.0 * focal_length_mm)))


@dataclass(frozen=True)
class DroneSpec:
    id: int
    name: str
    battery_wh: float
    v_min: float
    v_max: float
    h_min: float
    h_max: float
    v_vertical: float
    p_base_w: float
    c_drag: float
    c_altitude_j_m: float
    mass_kg: float
    a_max_ms2: float
    accel_efficiency: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DroneSpec":
        return _pick(cls, data)


@dataclass(frozen=True)
class CameraSpec:
    id: int
    name: str
    modality: str
    focal_length_mm: float
    sensor_width_mm: float
    n_px_width: int
    power_w: float
    battery_wh: float
    max_range_m: float

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CameraSpec":
        return _pick(cls, data)

    @property
    def fov_deg(self) -> float:
        return fov_from_optics(self.focal_length_mm, self.sensor_width_mm)


@dataclass(frozen=True)
class PersonSpec:
    id: int
    name: str
    critical_dim_m: float
    px_min_optical: int
    px_min_thermal: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PersonSpec":
        return _pick(cls, data)

    def res_max(self, modality: str) -> float:
        pixels = self.px_min_optical if modality == "optical" else self.px_min_thermal
        return self.critical_dim_m / pixels


@dataclass(frozen=True)
class Rect:
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Hill:
    x: float
    y: float
    height_m: float
    sigma_m: float


@dataclass(frozen=True)
class SolverParams:
    n_drones: int
    n_points: int
    population_size: int
    generations: int
    crossover_prob: float
    crossover_eta: float
    mutation_sigma: float
    seed: int
    battery_reserve: float
    min_separation_m: float
    base_clearance_m: float
    separation_sampling_s: float
    separation_max_samples: int
    random_baseline_evals: int

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SolverParams":
        return _pick(cls, data)

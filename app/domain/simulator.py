from dataclasses import dataclass, field

import numpy as np

from app.domain.chromosome import Layout
from app.domain.geometry import closest_on_segments, half_fov_tan, max_detection_height
from app.domain.terrain import Terrain
from app.domain.types import CameraSpec, DroneSpec, PersonSpec, SolverParams

JOULES_PER_WH = 3600.0


@dataclass
class DroneReport:
    drone_id: int
    camera_id: int
    time_s: float
    energy_wh: float
    camera_energy_wh: float
    battery_remaining_pct: float
    distance_m: float


@dataclass
class Metrics:
    r_de: float
    t_mission_min: float
    e_total_wh: float
    area_explored_pct: float
    area_detected_pct: float
    g: np.ndarray
    drones: list[DroneReport] = field(default_factory=list)

    @property
    def f(self) -> np.ndarray:
        return np.array([-self.r_de, self.t_mission_min, self.e_total_wh])

    @property
    def constraint_violation(self) -> float:
        return float(np.maximum(self.g, 0.0).sum())

    @property
    def feasible(self) -> bool:
        return self.constraint_violation <= 0.0


class Evaluator:
    def __init__(
        self,
        terrain: Terrain,
        drones: list[DroneSpec],
        cameras: list[CameraSpec],
        person: PersonSpec,
        params: SolverParams,
    ) -> None:
        self.terrain = terrain
        self.drones = drones
        self.cameras = cameras
        self.person = person
        self.params = params
        self.layout = Layout(params.n_drones, params.n_points, drones, cameras, terrain)
        self.tan_half = [half_fov_tan(c.fov_deg) for c in cameras]
        self.h_detect_max = [
            max_detection_height(person.res_max(c.modality), c.fov_deg, c.n_px_width)
            for c in cameras
        ]
        self.n_constr = 2 * params.n_drones + 1

    def evaluate(self, x: np.ndarray) -> Metrics:
        terrain, params = self.terrain, self.params
        P, V, td, tc = self.layout.split(x)
        P, V, td, tc = P[0], V[0], td[0], tc[0]

        explored_all = np.zeros(len(terrain.cell_xy), dtype=bool)
        detected_all = np.zeros(len(terrain.cell_xy), dtype=bool)
        reports: list[DroneReport] = []
        node_times, node_xy, node_alt = [], [], []
        g_battery, g_camera = [], []

        for d in range(params.n_drones):
            drone = self.drones[td[d]]
            camera = self.cameras[tc[d]]
            xy = np.vstack([terrain.base, P[d, :, :2], terrain.base])
            h = np.concatenate([[0.0], P[d, :, 2], [0.0]])
            alt = terrain.ground(xy) + h

            length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            v_leg = np.concatenate([V[d], V[d, -1:]])
            d_alt = np.abs(np.diff(alt))
            t_leg = length / v_leg + d_alt / drone.v_vertical
            time_s = float(t_leg.sum())

            energy_j = (
                drone.p_base_w * time_s
                + drone.c_drag * float(np.sum(v_leg**2 * length))
                + drone.c_altitude_j_m * float(d_alt.sum())
            )
            energy_wh = energy_j / JOULES_PER_WH
            camera_wh = camera.power_w * time_s / JOULES_PER_WH

            t, dist = closest_on_segments(terrain.cell_xy, xy[:-1], xy[1:])
            ht = h[:-1, None] + t * np.diff(h)[:, None]
            explored = dist <= ht * self.tan_half[tc[d]]
            detected = (
                explored
                & (ht <= self.h_detect_max[tc[d]])
                & (np.hypot(dist, ht) <= camera.max_range_m)
            )
            explored_all |= explored.any(axis=0)
            detected_all |= detected.any(axis=0)

            reports.append(
                DroneReport(
                    drone_id=drone.id,
                    camera_id=camera.id,
                    time_s=time_s,
                    energy_wh=energy_wh,
                    camera_energy_wh=camera_wh,
                    battery_remaining_pct=100.0 * (1.0 - energy_wh / drone.battery_wh),
                    distance_m=float(length.sum()),
                )
            )
            g_battery.append(energy_wh / (drone.battery_wh * (1.0 - params.battery_reserve)) - 1.0)
            g_camera.append(camera_wh / camera.battery_wh - 1.0)
            node_times.append(np.concatenate([[0.0], np.cumsum(t_leg)]))
            node_xy.append(xy)
            node_alt.append(alt)

        area, vis = terrain.cell_area, terrain.cell_vis
        total = terrain.total_area
        g_sep = self._separation(node_times, node_xy, node_alt)
        return Metrics(
            r_de=float((area * vis * detected_all).sum() / total * 100.0),
            t_mission_min=max(r.time_s for r in reports) / 60.0,
            e_total_wh=sum(r.energy_wh for r in reports),
            area_explored_pct=float((area * explored_all).sum() / total * 100.0),
            area_detected_pct=float((area * detected_all).sum() / total * 100.0),
            g=np.array(g_battery + g_camera + [g_sep]),
            drones=reports,
        )

    def _separation(self, node_times, node_xy, node_alt) -> float:
        params = self.params
        if params.n_drones < 2:
            return -1.0
        t_max = max(times[-1] for times in node_times)
        n_samples = int(
            min(t_max / params.separation_sampling_s + 2, params.separation_max_samples)
        )
        grid = np.linspace(0.0, t_max, max(n_samples, 2))
        base = self.terrain.base
        positions, away = [], []
        for times, xy, alt in zip(node_times, node_xy, node_alt, strict=True):
            px = np.interp(grid, times, xy[:, 0])
            py = np.interp(grid, times, xy[:, 1])
            pz = np.interp(grid, times, alt)
            positions.append(np.stack([px, py, pz], axis=1))
            away.append(np.hypot(px - base[0], py - base[1]) > params.base_clearance_m)
        closest = np.inf
        for i in range(params.n_drones):
            for j in range(i + 1, params.n_drones):
                valid = away[i] & away[j]
                if valid.any():
                    gap = np.linalg.norm(positions[i] - positions[j], axis=1)[valid].min()
                    closest = min(closest, float(gap))
        if not np.isfinite(closest):
            return -1.0
        return (params.min_separation_m - closest) / params.min_separation_m

    def evaluate_many(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        F = np.empty((len(X), 3))
        G = np.empty((len(X), self.n_constr))
        for i, row in enumerate(X):
            metrics = self.evaluate(row)
            F[i], G[i] = metrics.f, metrics.g
        return F, G

import math
from dataclasses import dataclass, field

import numpy as np

from app.domain.capacity import displacement_range_m
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
    energy_base_wh: float = 0.0
    energy_drag_wh: float = 0.0
    energy_climb_wh: float = 0.0
    energy_accel_wh: float = 0.0
    avg_speed_ms: float = 0.0
    range_m: float = 0.0
    best_range_speed_ms: float = 0.0
    g_battery: float = 0.0
    g_camera: float = 0.0
    g_stop: float = 0.0
    legs: list[dict] = field(default_factory=list)


@dataclass
class Flight:
    v_start: np.ndarray
    v_reach: np.ndarray
    v_end: np.ndarray
    accel: np.ndarray
    t_accel: np.ndarray
    t_cruise: np.ndarray
    t_brake: np.ndarray
    t_vertical: np.ndarray
    e_accel_j: np.ndarray
    g_stop: float

    @property
    def duration(self) -> np.ndarray:
        return self.t_accel + self.t_cruise + self.t_brake + self.t_vertical


def simulate_flight(
    drone: DroneSpec, length: np.ndarray, d_alt: np.ndarray, v_cmd: np.ndarray
) -> Flight:
    n_legs = len(length)
    a = drone.a_max_ms2
    mass, eta = drone.mass_kg, drone.accel_efficiency
    v_start = [0.0] * n_legs
    v_reach = [0.0] * n_legs
    v_end = [0.0] * n_legs
    accel = [0.0] * n_legs
    t_acc = [0.0] * n_legs
    t_cru = [0.0] * n_legs
    t_brk = [0.0] * n_legs
    e_acc = [0.0] * n_legs
    g_stop = -1.0
    speed = 0.0
    lengths, targets = length.tolist(), v_cmd.tolist()
    for i in range(n_legs):
        dist, target = lengths[i], targets[i]
        v_start[i] = speed
        if i == n_legs - 1:
            if target >= speed:
                target = min(target, math.sqrt((2.0 * a * dist + speed * speed) / 2.0))
            change = abs(target * target - speed * speed) / (2.0 * a)
            brake = target * target / (2.0 * a)
            need = change + brake
            g_stop = (need - dist) / max(dist, 1.0)
            t_acc[i] = abs(target - speed) / a
            t_brk[i] = target / a
            t_cru[i] = max(dist - need, 0.0) / target if target > 0.0 else 0.0
            if target > speed:
                e_acc[i] = 0.5 * mass * (target * target - speed * speed) / eta
                accel[i] = a
            elif target < speed:
                accel[i] = -a
            v_reach[i] = target
            v_end[i] = 0.0
            continue
        if target >= speed:
            need = (target * target - speed * speed) / (2.0 * a)
            reached = target if need <= dist else math.sqrt(speed * speed + 2.0 * a * dist)
            change = (reached * reached - speed * speed) / (2.0 * a)
            e_acc[i] = 0.5 * mass * (reached * reached - speed * speed) / eta
            accel[i] = a if reached > speed else 0.0
        else:
            need = (speed * speed - target * target) / (2.0 * a)
            reached = target if need <= dist else math.sqrt(speed * speed - 2.0 * a * dist)
            change = (speed * speed - reached * reached) / (2.0 * a)
            accel[i] = -a
        t_acc[i] = abs(reached - speed) / a
        t_cru[i] = (dist - change) / reached if reached > 0.0 and dist > change else 0.0
        v_reach[i] = reached
        v_end[i] = reached
        speed = reached
    return Flight(
        v_start=np.array(v_start),
        v_reach=np.array(v_reach),
        v_end=np.array(v_end),
        accel=np.array(accel),
        t_accel=np.array(t_acc),
        t_cruise=np.array(t_cru),
        t_brake=np.array(t_brk),
        t_vertical=d_alt / drone.v_vertical,
        e_accel_j=np.array(e_acc),
        g_stop=g_stop,
    )


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
        self.n_constr = 3 * params.n_drones + 1

    def evaluate(self, x: np.ndarray, record: bool = False) -> Metrics:
        terrain, params = self.terrain, self.params
        P, V, td, tc = self.layout.split(x)
        P, V, td, tc = P[0], V[0], td[0], tc[0]

        explored_all = np.zeros(len(terrain.cell_xy), dtype=bool)
        detected_all = np.zeros(len(terrain.cell_xy), dtype=bool)
        reports: list[DroneReport] = []
        node_times, node_xy, node_alt = [], [], []
        g_battery, g_camera, g_stop = [], [], []

        for d in range(params.n_drones):
            drone = self.drones[td[d]]
            camera = self.cameras[tc[d]]
            xy = np.vstack([terrain.base, P[d, :, :2], terrain.base])
            h = np.concatenate([[0.0], P[d, :, 2], [0.0]])
            alt = terrain.ground(xy) + h

            length = np.linalg.norm(np.diff(xy, axis=0), axis=1)
            v_cmd = np.concatenate([V[d], V[d, -1:]])
            d_alt = np.abs(np.diff(alt))
            flight = simulate_flight(drone, length, d_alt, v_cmd)
            t_leg = flight.duration
            time_s = float(t_leg.sum())

            base_j = drone.p_base_w * t_leg
            drag_j = drone.c_drag * flight.v_reach**2 * length
            climb_j = drone.c_altitude_j_m * d_alt
            leg_j = base_j + drag_j + climb_j + flight.e_accel_j
            energy_wh = float(leg_j.sum()) / JOULES_PER_WH
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

            range_m, best_speed = displacement_range_m(drone, params.battery_reserve)
            g_bat = energy_wh / (drone.battery_wh * (1.0 - params.battery_reserve)) - 1.0
            g_cam = camera_wh / camera.battery_wh - 1.0
            report = DroneReport(
                drone_id=drone.id,
                camera_id=camera.id,
                time_s=time_s,
                energy_wh=energy_wh,
                camera_energy_wh=camera_wh,
                battery_remaining_pct=100.0 * (1.0 - energy_wh / drone.battery_wh),
                distance_m=float(length.sum()),
                energy_base_wh=float(base_j.sum()) / JOULES_PER_WH,
                energy_drag_wh=float(drag_j.sum()) / JOULES_PER_WH,
                energy_climb_wh=float(climb_j.sum()) / JOULES_PER_WH,
                energy_accel_wh=float(flight.e_accel_j.sum()) / JOULES_PER_WH,
                avg_speed_ms=float(length.sum()) / time_s if time_s > 0.0 else 0.0,
                range_m=range_m,
                best_range_speed_ms=best_speed,
                g_battery=g_bat,
                g_camera=g_cam,
                g_stop=flight.g_stop,
            )
            if record:
                report.legs = self._legs(
                    d,
                    drone,
                    camera,
                    xy,
                    h,
                    length,
                    d_alt,
                    flight,
                    (leg_j, base_j, drag_j, climb_j),
                )
            reports.append(report)
            g_battery.append(g_bat)
            g_camera.append(g_cam)
            g_stop.append(flight.g_stop)
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
            g=np.array(g_battery + g_camera + g_stop + [g_sep]),
            drones=reports,
        )

    def _legs(self, d, drone, camera, xy, h, length, d_alt, flight, joules):
        leg_j, base_j, drag_j, climb_j = joules
        n_legs = len(length)
        duration = flight.duration
        starts = np.concatenate([[0.0], np.cumsum(duration)])
        spent = np.cumsum(leg_j) / JOULES_PER_WH
        rows = []
        for i in range(n_legs):
            kind = "outbound" if i == 0 else ("return" if i == n_legs - 1 else "search")
            rows.append(
                {
                    "drone_index": d,
                    "seq": i,
                    "drone_id": drone.id,
                    "kind": kind,
                    "x0": float(xy[i, 0]),
                    "y0": float(xy[i, 1]),
                    "h0": float(h[i]),
                    "x1": float(xy[i + 1, 0]),
                    "y1": float(xy[i + 1, 1]),
                    "h1": float(h[i + 1]),
                    "distance_m": float(length[i]),
                    "altitude_change_m": float(h[i + 1] - h[i]),
                    "speed_start_ms": float(flight.v_start[i]),
                    "speed_cruise_ms": float(flight.v_reach[i]),
                    "speed_end_ms": float(flight.v_end[i]),
                    "accel_ms2": float(flight.accel[i]),
                    "t_start_s": float(starts[i]),
                    "t_end_s": float(starts[i + 1]),
                    "t_accel_s": float(flight.t_accel[i]),
                    "t_cruise_s": float(flight.t_cruise[i]),
                    "t_brake_s": float(flight.t_brake[i]),
                    "t_vertical_s": float(flight.t_vertical[i]),
                    "energy_base_wh": float(base_j[i]) / JOULES_PER_WH,
                    "energy_drag_wh": float(drag_j[i]) / JOULES_PER_WH,
                    "energy_climb_wh": float(climb_j[i]) / JOULES_PER_WH,
                    "energy_accel_wh": float(flight.e_accel_j[i]) / JOULES_PER_WH,
                    "battery_remaining_pct": float(100.0 * (1.0 - spent[i] / drone.battery_wh)),
                    "camera_energy_wh": float(camera.power_w * duration[i] / JOULES_PER_WH),
                }
            )
        return rows

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

import math

import numpy as np
import pytest

from app.domain.simulator import Evaluator
from app.domain.terrain import build_terrain
from app.domain.types import CameraSpec, DroneSpec, PersonSpec, SolverParams

VISIBILITY = {"open": 0.95, "brush": 0.70, "dense": 0.45}
CONFIG = dict(
    width_m=100,
    height_m=100,
    cell_size_m=50,
    base=dict(x=0, y=0),
    vegetation_default="open",
    vegetation_zones=[],
    forbidden_zones=[],
    hills=[],
    paths=[],
)
DRONE = dict(
    id=1,
    name="d",
    battery_wh=100,
    v_min=1,
    v_max=20,
    h_min=5,
    h_max=50,
    v_vertical=2,
    p_base_w=60,
    c_drag=0.02,
    c_altitude_j_m=25,
    mass_kg=2,
    a_max_ms2=2,
    accel_efficiency=0.5,
)
CAMERA = dict(
    id=1,
    name="c",
    modality="optical",
    focal_length_mm=5,
    sensor_width_mm=10,
    n_px_width=1000,
    power_w=10,
    battery_wh=20,
    max_range_m=100,
)
PERSON = dict(id=1, name="p", critical_dim_m=0.5, px_min_optical=5, px_min_thermal=5)


def make_evaluator(n_drones=1, clearance=5.0, battery_wh=100, n_points=1):
    params = SolverParams(
        n_drones=n_drones,
        n_points=n_points,
        population_size=8,
        generations=2,
        crossover_prob=0.9,
        crossover_eta=15,
        mutation_sigma=0.05,
        seed=1,
        battery_reserve=0.15,
        min_separation_m=10,
        base_clearance_m=clearance,
        separation_sampling_s=0.5,
        separation_max_samples=4000,
        random_baseline_evals=0,
    )
    drone = DroneSpec.from_dict({**DRONE, "battery_wh": battery_wh})
    return Evaluator(
        build_terrain(CONFIG, VISIBILITY),
        [drone],
        [CameraSpec.from_dict(CAMERA)],
        PersonSpec.from_dict(PERSON),
        params,
    )


def encode(evaluator, points, speeds):
    layout = evaluator.layout
    n, k = evaluator.params.n_drones, evaluator.params.n_points
    P = np.array(points, dtype=float).reshape(1, n, k, 3)
    V = np.array(speeds, dtype=float).reshape(1, n, k)
    zeros = np.zeros((1, n), dtype=int)
    return layout.encode(P, V, zeros, zeros)[0]


def test_single_drone_matches_hand_calculation():
    evaluator = make_evaluator()
    metrics = evaluator.evaluate(encode(evaluator, [[50, 50, 10]], [5]))

    leg = math.sqrt(50**2 + 50**2)
    distance = 2 * leg
    ramp_s = 5 / 2
    ramp_m = 5**2 / (2 * 2)
    time_s = 2 * ramp_s + (distance - 2 * ramp_m) / 5 + 20 / 2
    accel_j = 0.5 * 2 * 5**2 / 0.5
    energy_j = 60 * time_s + 0.02 * 25 * distance + 25 * 20 + accel_j
    assert metrics.t_mission_min == pytest.approx(time_s / 60)
    assert metrics.e_total_wh == pytest.approx(energy_j / 3600)
    assert metrics.area_detected_pct == pytest.approx(25.0)
    assert metrics.area_explored_pct == pytest.approx(25.0)
    assert metrics.r_de == pytest.approx(25.0 * 0.95)
    assert metrics.feasible
    assert metrics.g[-1] == -1.0


def test_battery_violation_makes_solution_infeasible():
    evaluator = make_evaluator(battery_wh=0.5)
    metrics = evaluator.evaluate(encode(evaluator, [[50, 50, 10]], [5]))
    assert not metrics.feasible
    assert metrics.constraint_violation > 0.8


def test_flying_above_detection_height_explores_but_does_not_detect():
    evaluator = make_evaluator()
    metrics = evaluator.evaluate(encode(evaluator, [[50, 50, 50]], [5]))
    detect_limit = evaluator.h_detect_max[0]
    assert detect_limit == pytest.approx(50.0)
    low = evaluator.evaluate(encode(evaluator, [[50, 50, 49]], [5]))
    assert low.area_detected_pct >= metrics.area_detected_pct
    evaluator.h_detect_max[0] = 20.0
    high = evaluator.evaluate(encode(evaluator, [[50, 50, 40]], [5]))
    assert high.area_explored_pct > 0
    assert high.area_detected_pct < high.area_explored_pct


def test_identical_routes_violate_separation():
    evaluator = make_evaluator(n_drones=2)
    x = encode(evaluator, [[50, 50, 10], [50, 50, 10]], [5, 5])
    metrics = evaluator.evaluate(x)
    assert metrics.g[-1] == pytest.approx(1.0)
    assert not metrics.feasible


def test_base_clearance_ignores_takeoff_area():
    diverging = [[50, 50, 10], [50, 10, 10]]
    tight = make_evaluator(n_drones=2, clearance=5.0)
    relaxed = make_evaluator(n_drones=2, clearance=20.0)
    assert tight.evaluate(encode(tight, diverging, [5, 5])).g[-1] > 0
    assert relaxed.evaluate(encode(relaxed, diverging, [5, 5])).g[-1] < 0


def test_evaluate_many_matches_single_evaluation():
    evaluator = make_evaluator()
    x = encode(evaluator, [[50, 50, 10]], [5])
    F, G = evaluator.evaluate_many(np.vstack([x, x]))
    single = evaluator.evaluate(x)
    assert np.allclose(F[0], single.f) and np.allclose(G[1], single.g)


def test_camera_fov_is_derived_from_focal_length_and_sensor():
    camera = CameraSpec.from_dict(CAMERA)
    assert camera.fov_deg == pytest.approx(90.0)
    longer = CameraSpec.from_dict({**CAMERA, "focal_length_mm": 10})
    assert longer.fov_deg < camera.fov_deg


def test_speeding_up_between_legs_costs_extra_energy():
    evaluator = make_evaluator(n_points=2)
    points = [[40, 0, 10], [80, 0, 10]]
    steady = evaluator.evaluate(encode(evaluator, points, [8, 8]))
    surging = evaluator.evaluate(encode(evaluator, points, [4, 12]))
    assert surging.drones[0].energy_accel_wh > 0
    assert steady.drones[0].energy_accel_wh == pytest.approx(0.5 * 2 * 8**2 / 0.5 / 3600)
    assert surging.drones[0].energy_accel_wh > steady.drones[0].energy_accel_wh


def test_short_leg_limits_the_speed_that_can_be_reached():
    evaluator = make_evaluator()
    metrics = evaluator.evaluate(encode(evaluator, [[3, 0, 10]], [15]), record=True)
    first = metrics.drones[0].legs[0]
    assert first["speed_cruise_ms"] == pytest.approx(math.sqrt(2 * 2 * 3))
    assert first["speed_cruise_ms"] < 15


def test_recorded_legs_add_up_to_the_totals():
    evaluator = make_evaluator(n_points=3)
    points = [[30, 0, 10], [60, 30, 20], [80, 80, 15]]
    metrics = evaluator.evaluate(encode(evaluator, points, [5, 7, 6]), record=True)
    report = metrics.drones[0]
    legs = report.legs
    assert [leg["kind"] for leg in legs] == ["outbound", "search", "search", "return"]
    assert legs[0]["speed_start_ms"] == 0.0 and legs[-1]["speed_end_ms"] == 0.0
    assert sum(leg["t_end_s"] - leg["t_start_s"] for leg in legs) == pytest.approx(report.time_s)
    total_wh = sum(
        leg["energy_base_wh"]
        + leg["energy_drag_wh"]
        + leg["energy_climb_wh"]
        + leg["energy_accel_wh"]
        for leg in legs
    )
    assert total_wh == pytest.approx(report.energy_wh)
    assert legs[-1]["battery_remaining_pct"] == pytest.approx(report.battery_remaining_pct)
    assert all(
        a["t_end_s"] == pytest.approx(b["t_start_s"]) for a, b in zip(legs, legs[1:], strict=False)
    )


def test_displacement_range_uses_best_speed_inside_limits():
    from app.domain.capacity import displacement_range_m

    drone = DroneSpec.from_dict(DRONE)
    range_m, speed = displacement_range_m(drone, 0.15)
    assert drone.v_min <= speed <= drone.v_max
    assert range_m == pytest.approx(
        100 * 3600 * 0.85 / (drone.p_base_w / speed + drone.c_drag * speed**2)
    )

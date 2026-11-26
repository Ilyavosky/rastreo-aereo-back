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
)
CAMERA = dict(
    id=1,
    name="c",
    modality="optical",
    fov_deg=90,
    n_px_width=1000,
    power_w=10,
    battery_wh=20,
    max_range_m=100,
)
PERSON = dict(id=1, name="p", critical_dim_m=0.5, px_min_optical=5, px_min_thermal=5)


def make_evaluator(n_drones=1, clearance=5.0, battery_wh=100):
    params = SolverParams(
        n_drones=n_drones,
        n_points=1,
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
    n = evaluator.params.n_drones
    P = np.array(points, dtype=float).reshape(1, n, 1, 3)
    V = np.array(speeds, dtype=float).reshape(1, n, 1)
    zeros = np.zeros((1, n), dtype=int)
    return layout.encode(P, V, zeros, zeros)[0]


def test_single_drone_matches_hand_calculation():
    evaluator = make_evaluator()
    metrics = evaluator.evaluate(encode(evaluator, [[50, 50, 10]], [5]))

    distance = 2 * math.sqrt(50**2 + 50**2)
    time_s = distance / 5 + 20 / 2
    energy_j = 60 * time_s + 0.02 * 25 * distance + 25 * 20
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

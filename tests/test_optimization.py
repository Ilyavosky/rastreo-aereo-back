import numpy as np

from app.domain.baselines import grid_solution, random_front
from app.domain.profiles import select_profiles
from app.domain.simulator import Evaluator
from app.domain.solve import optimize
from app.domain.terrain import build_terrain
from app.domain.types import CameraSpec, DroneSpec, PersonSpec, SolverParams
from tests.payloads import CAMERA, DRONE, PERSON, SCENARIO_CONFIG, THERMAL

VISIBILITY = {"open": 0.95, "brush": 0.70, "dense": 0.45}


def make_evaluator(seed=7, generations=3):
    params = SolverParams(
        n_drones=2,
        n_points=4,
        population_size=12,
        generations=generations,
        crossover_prob=0.9,
        crossover_eta=15,
        mutation_sigma=0.05,
        seed=seed,
        battery_reserve=0.15,
        min_separation_m=10,
        base_clearance_m=50,
        separation_sampling_s=2,
        separation_max_samples=4000,
        random_baseline_evals=60,
    )
    return Evaluator(
        build_terrain(SCENARIO_CONFIG, VISIBILITY),
        [DroneSpec.from_dict({**DRONE, "id": 1})],
        [CameraSpec.from_dict({**CAMERA, "id": 1}), CameraSpec.from_dict({**THERMAL, "id": 2})],
        PersonSpec.from_dict({**PERSON, "id": 1}),
        params,
    )


def test_same_seed_gives_same_front():
    first = optimize(make_evaluator())
    second = optimize(make_evaluator())
    assert np.allclose(first.F, second.F)


def test_generation_callback_and_early_stop():
    seen = []
    result = optimize(make_evaluator(generations=5), on_generation=lambda s: seen.append(s.gen))
    assert seen == [1, 2, 3, 4, 5]
    assert result.generations_done == 5
    assert optimize(make_evaluator(generations=5), should_stop=lambda: True) is None


def test_result_respects_bounds_and_repair():
    evaluator = make_evaluator()
    result = optimize(evaluator)
    layout = evaluator.layout
    assert np.all(result.X >= layout.xl - 1e-9) and np.all(result.X <= layout.xu + 1e-9)
    for x in result.X:
        for rect in evaluator.terrain.forbidden:
            P = layout.split(x)[0][0]
            inside = (
                (P[..., 0] > rect.x0)
                & (P[..., 0] < rect.x1)
                & (P[..., 1] > rect.y0)
                & (P[..., 1] < rect.y1)
            )
            assert not inside.any()


def test_repair_clips_types_heights_and_forbidden_points():
    evaluator = make_evaluator()
    layout = evaluator.layout
    X = np.tile((layout.xl + layout.xu) / 2.0, (1, 1))
    P = X[:, layout.slice_p].reshape(1, 2, 4, 3)
    P[..., 0], P[..., 1], P[..., 2] = 350.0, 350.0, 1000.0
    X[:, layout.slice_p] = P.reshape(1, -1)
    X[:, layout.slice_td] = 9.7
    repaired = layout.repair(X)
    P2, V2, td, tc = layout.split(repaired)
    rect = evaluator.terrain.forbidden[0]
    assert not (
        (P2[..., 0] > rect.x0)
        & (P2[..., 0] < rect.x1)
        & (P2[..., 1] > rect.y0)
        & (P2[..., 1] < rect.y1)
    ).any()
    assert np.all(P2[..., 2] <= evaluator.drones[0].h_max)
    assert np.all(td == 0)


def test_profiles_pick_expected_extremes():
    F = np.array([[-50.0, 10.0, 30.0], [-20.0, 5.0, 10.0], [-40.0, 6.0, 15.0]])
    profiles = select_profiles(F)
    assert profiles["best_detection"] == 0
    assert profiles["lowest_consumption"] == 1
    assert profiles["balanced"] == 2
    assert select_profiles(np.empty((0, 3))) == {}


def test_baselines_produce_evaluable_solutions():
    evaluator = make_evaluator()
    metrics = evaluator.evaluate(grid_solution(evaluator))
    assert metrics.r_de > 0
    X, F = random_front(evaluator, 60, seed=3)
    assert len(X) == len(F)
    assert F.shape[1] == 3

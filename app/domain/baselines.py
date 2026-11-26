import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from app.domain.simulator import Evaluator


def grid_solution(evaluator: Evaluator) -> np.ndarray:
    layout, terrain, params = evaluator.layout, evaluator.terrain, evaluator.params
    drone = evaluator.drones[0]
    height = float(np.clip(evaluator.h_detect_max[0], drone.h_min, drone.h_max))
    speed = (drone.v_min + drone.v_max) / 2.0
    n_drones, n_points = params.n_drones, params.n_points
    lanes = max(1, n_points // 2)
    band = terrain.width_m / n_drones

    P = np.zeros((1, n_drones, n_points, 3))
    for d in range(n_drones):
        waypoints: list[tuple[float, float]] = []
        for lane in range(lanes):
            x = d * band + (lane + 0.5) * band / lanes
            ys = (0.0, terrain.height_m) if lane % 2 == 0 else (terrain.height_m, 0.0)
            waypoints.extend([(x, ys[0]), (x, ys[1])])
        if n_points < 2:
            waypoints = [(d * band + band / 2.0, terrain.height_m / 2.0)]
        waypoints = waypoints[:n_points]
        while len(waypoints) < n_points:
            waypoints.append(waypoints[-1])
        for k, (x, y) in enumerate(waypoints):
            P[0, d, k] = (x, y, height)
    V = np.full((1, n_drones, n_points), speed)
    td = np.zeros((1, n_drones), dtype=int)
    tc = np.zeros((1, n_drones), dtype=int)
    return layout.repair(layout.encode(P, V, td, tc))[0]


def random_front(
    evaluator: Evaluator, n_evals: int, seed: int, chunk: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    layout = evaluator.layout
    rng = np.random.default_rng(seed)
    kept_x: list[np.ndarray] = []
    kept_f: list[np.ndarray] = []
    done = 0
    while done < n_evals:
        size = min(chunk, n_evals - done)
        X = layout.repair(rng.uniform(layout.xl, layout.xu, size=(size, layout.n_var)))
        F, G = evaluator.evaluate_many(X)
        feasible = (np.maximum(G, 0.0).sum(axis=1)) <= 0.0
        if feasible.any():
            kept_x.append(X[feasible])
            kept_f.append(F[feasible])
        done += size
    if not kept_x:
        return np.empty((0, layout.n_var)), np.empty((0, 3))
    X_all, F_all = np.vstack(kept_x), np.vstack(kept_f)
    front = NonDominatedSorting().do(F_all, only_non_dominated_front=True)
    return X_all[front], F_all[front]

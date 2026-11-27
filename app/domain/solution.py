import numpy as np

from app.domain.simulator import Evaluator, Metrics


def describe_solution(evaluator: Evaluator, x: np.ndarray) -> tuple[Metrics, dict]:
    metrics = evaluator.evaluate(x, record=True)
    layout = evaluator.layout
    P, V, td, tc = layout.split(x)
    P, V, td, tc = P[0], V[0], td[0], tc[0]
    drones = []
    for d, report in enumerate(metrics.drones):
        drone, camera = evaluator.drones[td[d]], evaluator.cameras[tc[d]]
        drones.append(
            {
                "drone_id": drone.id,
                "drone_name": drone.name,
                "camera_id": camera.id,
                "camera_name": camera.name,
                "waypoints": [
                    {
                        "x": float(P[d, k, 0]),
                        "y": float(P[d, k, 1]),
                        "h": float(P[d, k, 2]),
                        "v": float(V[d, k]),
                    }
                    for k in range(evaluator.params.n_points)
                ],
                "time_s": report.time_s,
                "energy_wh": report.energy_wh,
                "camera_energy_wh": report.camera_energy_wh,
                "battery_remaining_pct": report.battery_remaining_pct,
                "distance_m": report.distance_m,
                "avg_speed_ms": report.avg_speed_ms,
                "range_m": report.range_m,
                "legs": [
                    {
                        "seq": leg["seq"],
                        "kind": leg["kind"],
                        "distance_m": leg["distance_m"],
                        "speed_ms": leg["speed_cruise_ms"],
                        "duration_s": leg["t_end_s"] - leg["t_start_s"],
                    }
                    for leg in report.legs
                ],
            }
        )
    base = evaluator.terrain.base
    return metrics, {"base": {"x": float(base[0]), "y": float(base[1])}, "drones": drones}

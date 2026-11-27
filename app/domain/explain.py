import numpy as np

from app.domain.geometry import half_fov_tan
from app.domain.simulator import Evaluator

FORMULAS = {
    "swath_m": "ancho = 2 · H · tan(FOV/2)",
    "fov_deg": "FOV = 2 · atan(sensor / (2 · focal))",
    "ground_resolution_m_per_px": "res = ancho / n_px",
    "res_max_m_per_px": "res_max = dimensión_crítica / px_mínimos",
    "h_detect_max_m": "H_det_max = res_max · n_px / (2 · tan(FOV/2))",
    "time_s": "t = t_aceleración + t_crucero + t_frenado + Σ|Δh| / v_vertical",
    "energy_base_wh": "E_base = P_base · t",
    "energy_drag_wh": "E_arrastre = c_drag · Σ v² · L",
    "energy_climb_wh": "E_altura = c_alt · Σ|Δh|",
    "energy_accel_wh": "E_acel = Σ ½ · m · (v_alcanzada² − v_inicial²) / η, solo cuando se acelera",
    "g_battery": "E_total / (batería · (1 − reserva)) − 1 ≤ 0",
    "g_camera": "E_cámara / batería_cámara − 1 ≤ 0",
    "g_stop": "(distancia_para_cambiar_velocidad_y_frenar − L_regreso) / L_regreso ≤ 0",
    "g_separation": "(separación_mínima − distancia_mínima_entre_drones) / separación_mínima ≤ 0",
    "r_de": "R_de = Σ(área · visibilidad · detectada) / área_total · 100",
    "t_mission_min": "T = máx(t_dron) / 60",
    "e_total_wh": "E_total = Σ E_dron",
}


def _constraint(value: float, formula: str) -> dict:
    return {"value": float(value), "satisfied": bool(value <= 0.0), "formula": formula}


def explain_solution(evaluator: Evaluator, x: np.ndarray) -> dict:
    metrics = evaluator.evaluate(x, record=True)
    params, person = evaluator.params, evaluator.person
    P, _, td, tc = evaluator.layout.split(x)
    P, td, tc = P[0], td[0], tc[0]
    n = params.n_drones

    drones = []
    for d, report in enumerate(metrics.drones):
        drone, camera = evaluator.drones[td[d]], evaluator.cameras[tc[d]]
        tan_half = half_fov_tan(camera.fov_deg)
        px_min = person.px_min_optical if camera.modality == "optical" else person.px_min_thermal
        h_det = float(evaluator.h_detect_max[tc[d]])
        mean_h = float(P[d, :, 2].mean())
        legs = report.legs
        drones.append(
            {
                "drone": {"id": drone.id, "name": drone.name, "mass_kg": drone.mass_kg},
                "camera": {"id": camera.id, "name": camera.name, "modality": camera.modality},
                "optics": {
                    "focal_length_mm": camera.focal_length_mm,
                    "sensor_width_mm": camera.sensor_width_mm,
                    "fov_deg": camera.fov_deg,
                    "half_fov_tan": tan_half,
                    "n_px_width": camera.n_px_width,
                    "person_critical_dim_m": person.critical_dim_m,
                    "px_min": px_min,
                    "res_max_m_per_px": person.critical_dim_m / px_min,
                    "h_detect_max_m": h_det,
                    "swath_at_h_detect_max_m": 2.0 * h_det * tan_half,
                    "mean_height_m": mean_h,
                    "swath_at_mean_height_m": 2.0 * mean_h * tan_half,
                    "ground_resolution_at_mean_height_m_per_px": 2.0
                    * mean_h
                    * tan_half
                    / camera.n_px_width,
                    "person_detectable_at_mean_height": bool(mean_h <= h_det),
                    "max_range_m": camera.max_range_m,
                },
                "flight": {
                    "time_s": report.time_s,
                    "t_accel_s": float(sum(leg["t_accel_s"] for leg in legs)),
                    "t_cruise_s": float(sum(leg["t_cruise_s"] for leg in legs)),
                    "t_brake_s": float(sum(leg["t_brake_s"] for leg in legs)),
                    "t_vertical_s": float(sum(leg["t_vertical_s"] for leg in legs)),
                    "distance_m": report.distance_m,
                    "avg_speed_ms": report.avg_speed_ms,
                    "a_max_ms2": drone.a_max_ms2,
                    "displacement_range_m": report.range_m,
                    "best_range_speed_ms": report.best_range_speed_ms,
                },
                "energy": {
                    "base_wh": report.energy_base_wh,
                    "drag_wh": report.energy_drag_wh,
                    "climb_wh": report.energy_climb_wh,
                    "accel_wh": report.energy_accel_wh,
                    "total_wh": report.energy_wh,
                    "battery_wh": drone.battery_wh,
                    "usable_wh": drone.battery_wh * (1.0 - params.battery_reserve),
                    "battery_remaining_pct": report.battery_remaining_pct,
                    "camera_wh": report.camera_energy_wh,
                },
                "constraints": {
                    "battery": _constraint(report.g_battery, FORMULAS["g_battery"]),
                    "camera": _constraint(report.g_camera, FORMULAS["g_camera"]),
                    "stop": _constraint(report.g_stop, FORMULAS["g_stop"]),
                },
            }
        )
    separation = float(metrics.g[3 * n])
    return {
        "formulas": FORMULAS,
        "drones": drones,
        "separation": _constraint(separation, FORMULAS["g_separation"]),
        "objectives": {
            "r_de": metrics.r_de,
            "t_mission_min": metrics.t_mission_min,
            "e_total_wh": metrics.e_total_wh,
            "area_explored_pct": metrics.area_explored_pct,
            "area_detected_pct": metrics.area_detected_pct,
        },
        "feasible": metrics.feasible,
        "constraint_violation": metrics.constraint_violation,
    }

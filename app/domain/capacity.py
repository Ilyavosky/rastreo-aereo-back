from app.domain.types import DroneSpec


def best_range_speed(drone: DroneSpec) -> float:
    if drone.c_drag <= 0:
        return drone.v_max
    optimum = (drone.p_base_w / (2.0 * drone.c_drag)) ** (1.0 / 3.0)
    return float(min(max(optimum, drone.v_min), drone.v_max))


def energy_per_meter_j(drone: DroneSpec, speed: float) -> float:
    return drone.p_base_w / speed + drone.c_drag * speed**2


def displacement_range_m(drone: DroneSpec, reserve: float) -> tuple[float, float]:
    speed = best_range_speed(drone)
    usable_j = drone.battery_wh * 3600.0 * (1.0 - reserve)
    return usable_j / energy_per_meter_j(drone, speed), speed

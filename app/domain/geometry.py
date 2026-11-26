import numpy as np


def half_fov_tan(fov_deg: float) -> float:
    return float(np.tan(np.radians(fov_deg) / 2.0))


def swath_width(height, fov_deg: float):
    return 2.0 * height * half_fov_tan(fov_deg)


def ground_resolution(height, fov_deg: float, n_px_width: int):
    return swath_width(height, fov_deg) / n_px_width


def max_detection_height(res_max: float, fov_deg: float, n_px_width: int) -> float:
    return res_max * n_px_width / (2.0 * half_fov_tan(fov_deg))


def closest_on_segments(points: np.ndarray, a: np.ndarray, b: np.ndarray):
    ab = b - a
    denom = np.einsum("sk,sk->s", ab, ab)
    ap = points[None, :, :] - a[:, None, :]
    num = np.einsum("snk,sk->sn", ap, ab)
    positive = denom > 0
    safe = np.where(positive, denom, 1.0)
    t = np.where(positive[:, None], num / safe[:, None], 0.0)
    t = np.clip(t, 0.0, 1.0)
    closest = a[:, None, :] + t[:, :, None] * ab[:, None, :]
    dist = np.linalg.norm(points[None, :, :] - closest, axis=2)
    return t, dist

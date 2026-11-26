from dataclasses import dataclass
from typing import Any

import numpy as np

from app.domain.geometry import closest_on_segments
from app.domain.types import Hill, Rect


@dataclass(frozen=True)
class Terrain:
    width_m: float
    height_m: float
    cell_size_m: float
    n_rows: int
    n_cols: int
    base: np.ndarray
    forbidden: tuple[Rect, ...]
    hills: tuple[Hill, ...]
    cell_xy: np.ndarray
    cell_area: np.ndarray
    cell_vis: np.ndarray

    @property
    def total_area(self) -> float:
        return float(self.cell_area.sum())

    def ground(self, xy: np.ndarray) -> np.ndarray:
        elevation = np.zeros(xy.shape[:-1])
        for hill in self.hills:
            d2 = (xy[..., 0] - hill.x) ** 2 + (xy[..., 1] - hill.y) ** 2
            elevation = elevation + hill.height_m * np.exp(-d2 / (2.0 * hill.sigma_m**2))
        return elevation


def build_terrain(config: dict[str, Any], visibility: dict[str, float]) -> Terrain:
    width = float(config["width_m"])
    height = float(config["height_m"])
    cell = float(config["cell_size_m"])
    n_cols = int(np.ceil(width / cell))
    n_rows = int(np.ceil(height / cell))

    x_start = np.arange(n_cols) * cell
    y_start = np.arange(n_rows) * cell
    x_size = np.minimum(cell, width - x_start)
    y_size = np.minimum(cell, height - y_start)
    xx, yy = np.meshgrid(x_start + x_size / 2.0, y_start + y_size / 2.0)
    aw, ah = np.meshgrid(x_size, y_size)
    cell_xy = np.stack([xx.ravel(), yy.ravel()], axis=1)
    cell_area = (aw * ah).ravel()

    cell_vis = np.full(len(cell_xy), visibility[config.get("vegetation_default", "open")])
    for zone in config.get("vegetation_zones", []):
        inside = (
            (cell_xy[:, 0] >= zone["x0"])
            & (cell_xy[:, 0] < zone["x1"])
            & (cell_xy[:, 1] >= zone["y0"])
            & (cell_xy[:, 1] < zone["y1"])
        )
        cell_vis[inside] = visibility[zone["type"]]

    base = np.array([config["base"]["x"], config["base"]["y"]], dtype=float)
    forbidden = tuple(
        Rect(z["x0"], z["y0"], z["x1"], z["y1"]) for z in config.get("forbidden_zones", [])
    )
    hills = tuple(
        Hill(h["x"], h["y"], h["height_m"], h["sigma_m"]) for h in config.get("hills", [])
    )
    return Terrain(
        width_m=width,
        height_m=height,
        cell_size_m=cell,
        n_rows=n_rows,
        n_cols=n_cols,
        base=base,
        forbidden=forbidden,
        hills=hills,
        cell_xy=cell_xy,
        cell_area=cell_area,
        cell_vis=cell_vis,
    )


def priority_map(
    terrain: Terrain,
    config: dict[str, Any],
    sigma_m: float,
    path_weight: float,
    path_width_m: float,
) -> np.ndarray:
    weights = np.zeros(len(terrain.cell_xy))
    last_seen = config.get("last_seen")
    if last_seen is not None:
        center = np.array([last_seen["x"], last_seen["y"]], dtype=float)
        d2 = ((terrain.cell_xy - center) ** 2).sum(axis=1)
        weights += np.exp(-d2 / (2.0 * sigma_m**2))
    for path in config.get("paths", []):
        pts = np.array([[p["x"], p["y"]] for p in path["points"]], dtype=float)
        _, dist = closest_on_segments(terrain.cell_xy, pts[:-1], pts[1:])
        weights += path_weight * np.exp(-(dist.min(axis=0) ** 2) / (2.0 * path_width_m**2))
    total = weights.sum()
    if total > 0:
        weights = weights / total
    return weights.reshape(terrain.n_rows, terrain.n_cols)

import numpy as np
import pytest

from app.domain.geometry import (
    closest_on_segments,
    ground_resolution,
    max_detection_height,
    swath_width,
)
from app.domain.terrain import build_terrain, priority_map

VISIBILITY = {"open": 0.95, "brush": 0.70, "dense": 0.45}
CONFIG = dict(
    width_m=100,
    height_m=100,
    cell_size_m=30,
    base=dict(x=0, y=0),
    last_seen=dict(x=50, y=50),
    vegetation_default="open",
    vegetation_zones=[dict(type="dense", x0=0, y0=0, x1=30, y1=30)],
    forbidden_zones=[],
    hills=[dict(x=50, y=50, height_m=20, sigma_m=10)],
    paths=[],
)


def test_swath_and_resolution():
    assert swath_width(50.0, 90.0) == pytest.approx(100.0)
    assert ground_resolution(50.0, 90.0, 1000) == pytest.approx(0.1)


def test_max_detection_height_inverts_resolution():
    height = max_detection_height(0.05, 84.0, 4000)
    assert ground_resolution(height, 84.0, 4000) == pytest.approx(0.05)


def test_closest_on_segments_handles_zero_length_segment():
    points = np.array([[3.0, 4.0]])
    t, dist = closest_on_segments(points, np.array([[0.0, 0.0]]), np.array([[0.0, 0.0]]))
    assert t[0, 0] == 0.0
    assert dist[0, 0] == pytest.approx(5.0)


def test_terrain_grid_covers_full_area_with_partial_cells():
    terrain = build_terrain(CONFIG, VISIBILITY)
    assert (terrain.n_rows, terrain.n_cols) == (4, 4)
    assert terrain.total_area == pytest.approx(100 * 100)
    assert terrain.cell_area.min() == pytest.approx(10 * 10)


def test_terrain_visibility_zone_overrides_default():
    terrain = build_terrain(CONFIG, VISIBILITY)
    assert terrain.cell_vis[0] == pytest.approx(0.45)
    assert terrain.cell_vis[-1] == pytest.approx(0.95)


def test_ground_elevation_peaks_at_hill():
    terrain = build_terrain(CONFIG, VISIBILITY)
    peak = terrain.ground(np.array([[50.0, 50.0]]))[0]
    assert peak == pytest.approx(20.0)
    assert terrain.ground(np.array([[0.0, 100.0]]))[0] < 0.01


def test_priority_map_sums_to_one_and_peaks_near_last_seen():
    terrain = build_terrain(CONFIG, VISIBILITY)
    grid = priority_map(terrain, CONFIG, 20.0, 0.5, 10.0)
    assert grid.sum() == pytest.approx(1.0)
    row, col = np.unravel_index(grid.argmax(), grid.shape)
    assert (row, col) in {(1, 1), (1, 2), (2, 1), (2, 2)}

import numpy as np

from app.domain.terrain import Terrain
from app.domain.types import CameraSpec, DroneSpec

TYPE_EPSILON = 1e-9


class Layout:
    def __init__(
        self,
        n_drones: int,
        n_points: int,
        drones: list[DroneSpec],
        cameras: list[CameraSpec],
        terrain: Terrain,
    ) -> None:
        self.n_drones = n_drones
        self.n_points = n_points
        self.drones = drones
        self.cameras = cameras
        self.terrain = terrain

        n_p = n_drones * n_points * 3
        n_v = n_drones * n_points
        self.slice_p = slice(0, n_p)
        self.slice_v = slice(n_p, n_p + n_v)
        self.slice_td = slice(n_p + n_v, n_p + n_v + n_drones)
        self.slice_tc = slice(n_p + n_v + n_drones, n_p + n_v + 2 * n_drones)
        self.n_var = n_p + n_v + 2 * n_drones

        self.h_min = np.array([d.h_min for d in drones])
        self.h_max = np.array([d.h_max for d in drones])
        self.v_min = np.array([d.v_min for d in drones])
        self.v_max = np.array([d.v_max for d in drones])

        xl = np.empty(self.n_var)
        xu = np.empty(self.n_var)
        p_low = np.tile([0.0, 0.0, self.h_min.min()], n_drones * n_points)
        p_high = np.tile([terrain.width_m, terrain.height_m, self.h_max.max()], n_drones * n_points)
        xl[self.slice_p], xu[self.slice_p] = p_low, p_high
        xl[self.slice_v], xu[self.slice_v] = self.v_min.min(), self.v_max.max()
        xl[self.slice_td], xu[self.slice_td] = 0.0, float(len(drones))
        xl[self.slice_tc], xu[self.slice_tc] = 0.0, float(len(cameras))
        self.xl, self.xu = xl, xu

    def type_index(self, genes: np.ndarray, n_types: int) -> np.ndarray:
        clipped = np.clip(genes, 0.0, n_types * (1.0 - TYPE_EPSILON))
        return np.floor(clipped).astype(int)

    def split(self, X: np.ndarray):
        X = np.atleast_2d(X)
        n = X.shape[0]
        P = X[:, self.slice_p].reshape(n, self.n_drones, self.n_points, 3)
        V = X[:, self.slice_v].reshape(n, self.n_drones, self.n_points)
        td = self.type_index(X[:, self.slice_td], len(self.drones))
        tc = self.type_index(X[:, self.slice_tc], len(self.cameras))
        return P, V, td, tc

    def encode(self, P: np.ndarray, V: np.ndarray, td: np.ndarray, tc: np.ndarray) -> np.ndarray:
        n = P.shape[0]
        X = np.empty((n, self.n_var))
        X[:, self.slice_p] = P.reshape(n, -1)
        X[:, self.slice_v] = V.reshape(n, -1)
        X[:, self.slice_td] = td + 0.5
        X[:, self.slice_tc] = tc + 0.5
        return X

    def repair(self, X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=float, copy=True)
        n = X.shape[0]
        P = X[:, self.slice_p].reshape(n, self.n_drones, self.n_points, 3).copy()
        V = X[:, self.slice_v].reshape(n, self.n_drones, self.n_points).copy()
        td = self.type_index(X[:, self.slice_td], len(self.drones))

        P[..., 0] = np.clip(P[..., 0], 0.0, self.terrain.width_m)
        P[..., 1] = np.clip(P[..., 1], 0.0, self.terrain.height_m)
        for _ in range(2):
            for rect in self.terrain.forbidden:
                self._push_out(P, rect)
        P[..., 2] = np.clip(P[..., 2], self.h_min[td][:, :, None], self.h_max[td][:, :, None])
        V = np.clip(V, self.v_min[td][:, :, None], self.v_max[td][:, :, None])

        X[:, self.slice_p] = P.reshape(n, -1)
        X[:, self.slice_v] = V.reshape(n, -1)
        X[:, self.slice_td] = np.clip(
            X[:, self.slice_td], 0.0, len(self.drones) * (1.0 - TYPE_EPSILON)
        )
        X[:, self.slice_tc] = np.clip(
            X[:, self.slice_tc], 0.0, len(self.cameras) * (1.0 - TYPE_EPSILON)
        )
        return X

    def _push_out(self, P: np.ndarray, rect) -> None:
        x, y = P[..., 0], P[..., 1]
        inside = (x > rect.x0) & (x < rect.x1) & (y > rect.y0) & (y < rect.y1)
        if not inside.any():
            return
        width, height = self.terrain.width_m, self.terrain.height_m
        distances = np.stack([x - rect.x0, rect.x1 - x, y - rect.y0, rect.y1 - y])
        valid = np.array([rect.x0 >= 0.0, rect.x1 <= width, rect.y0 >= 0.0, rect.y1 <= height])
        if not valid.any():
            return
        distances[~valid] = np.inf
        edge = np.argmin(distances, axis=0)
        x_new = np.where(edge == 0, rect.x0, np.where(edge == 1, rect.x1, x))
        y_new = np.where(edge == 2, rect.y0, np.where(edge == 3, rect.y1, y))
        P[..., 0] = np.where(inside, x_new, x)
        P[..., 1] = np.where(inside, y_new, y)

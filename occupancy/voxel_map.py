"""3D voxel occupancy map for collision queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import Point

from data_loader.geojson_loader import BuildingFootprint, WorldBounds

logger = logging.getLogger(__name__)


@dataclass
class VoxelMap:
    """3D voxel occupancy grid with distance transform support."""

    grid: np.ndarray  # shape (nz, ny, nx), 1 = occupied
    resolution_m: float
    origin: tuple[float, float, float]  # world coords of voxel (0,0,0)
    bounds: WorldBounds

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.grid.shape

    def world_to_voxel(self, x: float, y: float, z: float) -> tuple[int, int, int]:
        ix = int((x - self.origin[0]) / self.resolution_m)
        iy = int((y - self.origin[1]) / self.resolution_m)
        iz = int((z - self.origin[2]) / self.resolution_m)
        return ix, iy, iz

    def voxel_to_world(self, ix: int, iy: int, iz: int) -> tuple[float, float, float]:
        return (
            self.origin[0] + (ix + 0.5) * self.resolution_m,
            self.origin[1] + (iy + 0.5) * self.resolution_m,
            self.origin[2] + (iz + 0.5) * self.resolution_m,
        )

    def in_bounds(self, ix: int, iy: int, iz: int) -> bool:
        nz, ny, nx = self.grid.shape
        return 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz

    def is_occupied(self, x: float, y: float, z: float) -> bool:
        ix, iy, iz = self.world_to_voxel(x, y, z)
        if not self.in_bounds(ix, iy, iz):
            return True
        return bool(self.grid[iz, iy, ix])

    def nearest_obstacle_distance(self, x: float, y: float, z: float) -> float:
        """Horizontal distance to nearest occupied voxel at similar altitude."""
        iz, _, _ = self.world_to_voxel(x, y, z)
        iz = int(np.clip(iz, 0, self.grid.shape[0] - 1))
        slice_2d = self.grid[iz]
        if slice_2d.sum() == 0:
            return float("inf")
        ix, iy, _ = self.world_to_voxel(x, y, z)
        dist_map = distance_transform_edt(1 - slice_2d) * self.resolution_m
        iy_c = int(np.clip(iy, 0, dist_map.shape[0] - 1))
        ix_c = int(np.clip(ix, 0, dist_map.shape[1] - 1))
        return float(dist_map[iy_c, ix_c])

    def local_slice(
        self, x: float, y: float, z: float, size: int = 16
    ) -> np.ndarray:
        """Extract local occupancy patch centered on (x,y) at altitude z."""
        iz, _, _ = self.world_to_voxel(x, y, z)
        iz = int(np.clip(iz, 0, self.grid.shape[0] - 1))
        ix, iy, _ = self.world_to_voxel(x, y, z)
        half = size // 2
        patch = np.zeros((size, size), dtype=np.float32)
        for dy in range(-half, half):
            for dx in range(-half, half):
                sx, sy = ix + dx, iy + dy
                if self.in_bounds(sx, sy, iz):
                    patch[dy + half, dx + half] = float(self.grid[iz, sy, sx])
        return patch.flatten()

    def cost_map_2d(self, altitude_m: float) -> np.ndarray:
        """2D cost map at a given altitude (0=free, 1=occupied)."""
        iz, _, _ = self.world_to_voxel(0, 0, altitude_m)
        iz = int(np.clip(iz, 0, self.grid.shape[0] - 1))
        return self.grid[iz].astype(np.float32)


def build_voxel_map(
    buildings: Sequence[BuildingFootprint],
    bounds: WorldBounds,
    resolution_m: float = 5.0,
    max_height_m: float = 150.0,
    clearance_m: float = 0.0,
) -> VoxelMap:
    """Voxelize building footprints into a 3D occupancy grid."""
    x0 = bounds.min_x - bounds.margin
    y0 = bounds.min_y - bounds.margin
    z0 = 0.0
    nx = int(np.ceil(bounds.width / resolution_m))
    ny = int(np.ceil(bounds.height / resolution_m))
    nz = int(np.ceil(max_height_m / resolution_m))
    grid = np.zeros((nz, ny, nx), dtype=np.uint8)

    for building in buildings:
        minx, miny, maxx, maxy = building.polygon.bounds
        ix0 = max(0, int((minx - x0) / resolution_m))
        iy0 = max(0, int((miny - y0) / resolution_m))
        ix1 = min(nx, int(np.ceil((maxx - x0) / resolution_m)) + 1)
        iy1 = min(ny, int(np.ceil((maxy - y0) / resolution_m)) + 1)
        z_layers = int(np.ceil((building.height_m + clearance_m) / resolution_m))
        z_layers = min(z_layers, nz)

        for iy in range(iy0, iy1):
            cy = y0 + (iy + 0.5) * resolution_m
            for ix in range(ix0, ix1):
                cx = x0 + (ix + 0.5) * resolution_m
                if building.polygon.contains(Point(cx, cy)):
                    grid[:z_layers, iy, ix] = 1

    logger.info("Built voxel map %s (resolution=%.1fm)", grid.shape, resolution_m)
    return VoxelMap(
        grid=grid,
        resolution_m=resolution_m,
        origin=(x0, y0, z0),
        bounds=bounds,
    )

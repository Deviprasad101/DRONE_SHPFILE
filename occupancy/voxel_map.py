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

    def tall_obstacle_map_2d(self, altitude_m: float, clearance_m: float = 8.0) -> np.ndarray:
        """2D obstacle map for buildings that reach into the drone's flight band.

        A cell is marked occupied if any voxel layer between
        (altitude_m - clearance_m) and (altitude_m + clearance_m) is filled.
        This correctly ignores short buildings the drone can fly over, while
        blocking buildings that are genuinely too tall to fly past safely.
        """
        from scipy.ndimage import binary_dilation

        nz = self.grid.shape[0]
        iz_low = max(0, int((altitude_m - clearance_m) / self.resolution_m))
        iz_high = min(nz - 1, int((altitude_m + clearance_m) / self.resolution_m) + 1)
        # Any building that occupies at least one layer in the flight band is an obstacle
        band = self.grid[iz_low:iz_high + 1]
        occ = (np.max(band, axis=0) > 0).astype(np.uint8)
        # Dilate horizontally to add a safety clearance buffer
        if clearance_m > 0:
            radius = max(1, int(np.ceil(clearance_m / self.resolution_m)))
            structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            occ = binary_dilation(occ, structure=structure).astype(np.uint8)
        return occ.astype(np.float32)

    def segment_collides(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        altitude_m: float,
        clearance_m: float = 8.0,
    ) -> bool:
        """True if the world-space line segment (x0,y0)→(x1,y1) crosses any obstacle.

        Uses Bresenham's line algorithm to trace every grid cell along the segment
        and checks each cell against the altitude-aware obstacle band. This is the
        only correct way to validate interpolated or smoothed path segments.
        """
        occ = self.tall_obstacle_map_2d(altitude_m, clearance_m)
        ix0, iy0, _ = self.world_to_voxel(x0, y0, altitude_m)
        ix1, iy1, _ = self.world_to_voxel(x1, y1, altitude_m)
        h, w = occ.shape
        dx = abs(ix1 - ix0)
        dy = abs(iy1 - iy0)
        sx = 1 if ix0 < ix1 else -1
        sy = 1 if iy0 < iy1 else -1
        err = dx - dy
        cx, cy = ix0, iy0
        while True:
            if 0 <= cx < w and 0 <= cy < h and occ[cy, cx] > 0:
                return True
            if cx == ix1 and cy == iy1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                cx += sx
            if e2 < dx:
                err += dx
                cy += sy
        return False

    def footprint_map_2d(self, clearance_m: float = 0.0) -> np.ndarray:
        """2D occupancy from building footprints (any height), with optional clearance buffer."""
        from scipy.ndimage import binary_dilation

        occ = (np.max(self.grid, axis=0) > 0).astype(np.uint8)
        if clearance_m > 0:
            radius = max(1, int(np.ceil(clearance_m / self.resolution_m)))
            structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
            occ = binary_dilation(occ, structure=structure).astype(np.uint8)
        return occ.astype(np.float32)


def build_voxel_map(
    buildings: Sequence[BuildingFootprint],
    bounds: WorldBounds,
    resolution_m: float = 5.0,
    max_height_m: float = 200.0,
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

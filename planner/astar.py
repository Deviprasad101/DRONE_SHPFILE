"""A* global path planner with cost maps and path smoothing."""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from data_loader.geojson_loader import WorldBounds
from occupancy.voxel_map import VoxelMap

logger = logging.getLogger(__name__)


@dataclass
class PathPlan:
    """Global path with waypoints in 3D."""

    waypoints: list[tuple[float, float, float]]
    grid_resolution_m: float
    flight_altitude_m: float

    @property
    def length(self) -> float:
        if len(self.waypoints) < 2:
            return 0.0
        total = 0.0
        for a, b in zip(self.waypoints[:-1], self.waypoints[1:]):
            total += float(np.linalg.norm(np.array(b) - np.array(a)))
        return total


def generate_cost_map(
    voxel_map: VoxelMap,
    altitude_m: float,
    clearance_weight: float = 2.0,
) -> np.ndarray:
    """Build navigation cost map from occupancy (higher = costlier)."""
    occ = voxel_map.cost_map_2d(altitude_m)
    from scipy.ndimage import distance_transform_edt

    free_dist = distance_transform_edt(1 - occ) * voxel_map.resolution_m
    cost = occ.astype(np.float32) * 100.0
    cost += clearance_weight / (free_dist + 0.5)
    return cost


def _heuristic(a: tuple[int, int], b: tuple[int, int]) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _world_to_grid(
    x: float, y: float, voxel_map: VoxelMap
) -> tuple[int, int]:
    ix, iy, _ = voxel_map.world_to_voxel(x, y, 0)
    return ix, iy


def _grid_to_world(
    ix: int, iy: int, altitude_m: float, voxel_map: VoxelMap
) -> tuple[float, float, float]:
    x, y, _ = voxel_map.voxel_to_world(ix, iy, 0)
    return x, y, altitude_m


def astar_plan(
    start: tuple[float, float],
    goal: tuple[float, float],
    voxel_map: VoxelMap,
    flight_altitude_m: float = 40.0,
    clearance_m: float = 3.0,
) -> PathPlan:
    """Plan collision-free path using A* on a 2D cost map at flight altitude."""
    cost_map = generate_cost_map(voxel_map, flight_altitude_m)
    occ = voxel_map.cost_map_2d(flight_altitude_m)
    height, width = occ.shape
    resolution = voxel_map.resolution_m

    start_cell = _world_to_grid(start[0], start[1], voxel_map)
    goal_cell = _world_to_grid(goal[0], goal[1], voxel_map)

    def in_bounds(cell: tuple[int, int]) -> bool:
        return 0 <= cell[0] < width and 0 <= cell[1] < height

    def passable(cell: tuple[int, int]) -> bool:
        return occ[cell[1], cell[0]] == 0

    if not in_bounds(start_cell) or not in_bounds(goal_cell):
        return _direct_plan(start, goal, flight_altitude_m, resolution)

    if not passable(start_cell):
        start_cell = _nearest_free(start_cell, occ, width, height)
    if not passable(goal_cell):
        goal_cell = _nearest_free(goal_cell, occ, width, height)

    open_set: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_set, (0.0, start_cell))
    came_from: dict[tuple[int, int], tuple[int, int] | None] = {start_cell: None}
    g_score: dict[tuple[int, int], float] = {start_cell: 0.0}
    neighbors = [
        (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, 1), (-1, 1), (1, -1), (-1, -1),
    ]

    found = False
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_cell:
            found = True
            break
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            if not in_bounds(neighbor) or not passable(neighbor):
                continue
            step = 1.414 if dx and dy else 1.0
            cell_cost = float(cost_map[neighbor[1], neighbor[0]])
            tentative = g_score[current] + step * (1.0 + cell_cost)
            if tentative < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative
                f = tentative + _heuristic(neighbor, goal_cell)
                heapq.heappush(open_set, (f, neighbor))

    if not found:
        logger.warning("A* failed; using direct path")
        return _direct_plan(start, goal, flight_altitude_m, resolution)

    cells: list[tuple[int, int]] = []
    cell: tuple[int, int] | None = goal_cell
    while cell is not None:
        cells.append(cell)
        cell = came_from[cell]
    cells.reverse()

    waypoints = [_grid_to_world(ix, iy, flight_altitude_m, voxel_map) for ix, iy in cells]
    waypoints = smooth_path(waypoints)
    logger.info("Planned %d waypoints (%.1fm)", len(waypoints), PathPlan(waypoints, resolution, flight_altitude_m).length)
    return PathPlan(waypoints=waypoints, grid_resolution_m=resolution, flight_altitude_m=flight_altitude_m)


def smooth_path(
    waypoints: Sequence[tuple[float, float, float]],
    window: int = 3,
    min_spacing_m: float = 15.0,
) -> list[tuple[float, float, float]]:
    """Gaussian smooth and subsample waypoints."""
    if len(waypoints) <= 2:
        return list(waypoints)

    pts = np.array(waypoints)
    for dim in range(3):
        pts[:, dim] = gaussian_filter(pts[:, dim], sigma=window, mode="nearest")

    simplified = [tuple(pts[0])]
    for wp in pts[1:]:
        last = np.array(simplified[-1])
        if float(np.linalg.norm(wp - last)) >= min_spacing_m:
            simplified.append(tuple(wp))
    if simplified[-1] != tuple(pts[-1]):
        simplified.append(tuple(pts[-1]))
    return simplified


def nearest_waypoint_index(
    position: tuple[float, float, float],
    waypoints: Sequence[tuple[float, float, float]],
) -> int:
    """Index of closest waypoint."""
    if not waypoints:
        return 0
    dists = [float(np.linalg.norm(np.array(position) - np.array(w))) for w in waypoints]
    return int(np.argmin(dists))


def _direct_plan(
    start: tuple[float, float],
    goal: tuple[float, float],
    altitude: float,
    resolution: float,
) -> PathPlan:
    return PathPlan(
        waypoints=[(start[0], start[1], altitude), (goal[0], goal[1], altitude)],
        grid_resolution_m=resolution,
        flight_altitude_m=altitude,
    )


def _nearest_free(
    cell: tuple[int, int], occ: np.ndarray, width: int, height: int, radius: int = 25
) -> tuple[int, int]:
    for r in range(1, radius + 1):
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                nx, ny = cell[0] + dx, cell[1] + dy
                if 0 <= nx < width and 0 <= ny < height and occ[ny, nx] == 0:
                    return nx, ny
    return cell

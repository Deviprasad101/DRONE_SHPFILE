"""PyBullet drone simulation with kinematic fallback (footprint-based collision)."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from data_loader.geojson_loader import BuildingFootprint, WorldBounds
from planner.astar import PathPlan

logger = logging.getLogger(__name__)

try:
    import pybullet as p
    import pybullet_data

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False
    logger.warning("PyBullet unavailable; using kinematic fallback.")


@dataclass
class DroneState:
    """Drone pose and velocity."""

    position: np.ndarray
    velocity: np.ndarray
    yaw: float
    collision: bool = False


@dataclass
class StepResult:
    """Simulation step output."""

    state: DroneState
    info: dict[str, float] = field(default_factory=dict)


class DroneSimulator:
    """3D drone simulator with PyBullet physics (or kinematic fallback)."""

    def __init__(
        self,
        buildings: Sequence[BuildingFootprint],
        bounds: WorldBounds,
        timestep: float = 0.02,
        max_velocity: float = 8.0,
        max_yaw_rate: float = 1.5,
        drone_mass: float = 1.5,
        drone_size: float = 0.4,
        gui: bool = False,
    ) -> None:
        self.buildings = list(buildings)
        self.bounds = bounds
        self.timestep = timestep
        self.max_velocity = max_velocity
        self.max_yaw_rate = max_yaw_rate
        self.drone_mass = drone_mass
        self.drone_size = drone_size
        self.gui = gui

        self.client_id: int | None = None
        self.drone_id: int | None = None
        self.building_body_ids: list[int] = []
        self.planned_path: PathPlan | None = None
        self.trajectory: list[np.ndarray] = []
        self._prev_action: np.ndarray | None = None
        self.goal = np.zeros(3, dtype=np.float32)

        self._pos = np.zeros(3, dtype=np.float32)
        self._vel = np.zeros(3, dtype=np.float32)
        self._yaw = 0.0
        self._use_pybullet = HAS_PYBULLET

    def initialize_world(self) -> None:
        """Connect physics engine and load static building bodies."""
        if not self._use_pybullet:
            return
        mode = p.GUI if self.gui else p.DIRECT
        self.client_id = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81, physicsClientId=self.client_id)
        p.setTimeStep(self.timestep, physicsClientId=self.client_id)
        self.load_buildings()

    def load_buildings(self) -> None:
        """Create axis-aligned box colliders from building footprints."""
        if not self._use_pybullet:
            return
        assert self.client_id is not None

        ground_half = [self.bounds.width / 2, self.bounds.height / 2, 0.25]
        ground_col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=ground_half,
            physicsClientId=self.client_id,
        )
        p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=ground_col,
            basePosition=[self.bounds.center[0], self.bounds.center[1], -0.25],
            physicsClientId=self.client_id,
        )

        for building in self.buildings:
            minx, miny, maxx, maxy = building.polygon.bounds
            half_x = max((maxx - minx) / 2.0, 0.5)
            half_y = max((maxy - miny) / 2.0, 0.5)
            half_z = max(building.height_m / 2.0, 0.5)
            center_x = (minx + maxx) / 2.0
            center_y = (miny + maxy) / 2.0

            col = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[half_x, half_y, half_z],
                physicsClientId=self.client_id,
            )
            bid = p.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=col,
                basePosition=[center_x, center_y, half_z],
                physicsClientId=self.client_id,
            )
            self.building_body_ids.append(bid)

    def spawn_drone(self, position: tuple[float, float, float]) -> None:
        """Create drone rigid body at position."""
        if not self._use_pybullet:
            self._pos = np.array(position, dtype=np.float32)
            return
        if self.client_id is None:
            self.initialize_world()
        assert self.client_id is not None
        half = self.drone_size / 2.0
        col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half, half, half / 2],
            physicsClientId=self.client_id,
        )
        vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half, half, half / 2],
            rgbaColor=[0.1, 0.6, 0.95, 1.0], physicsClientId=self.client_id,
        )
        self.drone_id = p.createMultiBody(
            baseMass=self.drone_mass,
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=list(position),
            physicsClientId=self.client_id,
        )

    def reset(
        self,
        start: tuple[float, float, float],
        goal: tuple[float, float, float],
        planned_path: PathPlan | None = None,
    ) -> DroneState:
        """Reset drone to start pose."""
        if self._use_pybullet:
            if self.client_id is None:
                self.initialize_world()
                self.spawn_drone(start)
            assert self.drone_id is not None and self.client_id is not None
            p.resetBasePositionAndOrientation(
                self.drone_id, start, p.getQuaternionFromEuler([0, 0, 0]),
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                self.drone_id, [0, 0, 0], [0, 0, 0], physicsClientId=self.client_id,
            )
        else:
            self._pos = np.array(start, dtype=np.float32)
            self._vel = np.zeros(3, dtype=np.float32)
            self._yaw = 0.0

        self.planned_path = planned_path
        self.trajectory = [np.array(start, dtype=np.float32)]
        self._prev_action = None
        self.goal = np.array(goal, dtype=np.float32)
        return self._read_state(collision=False)

    def step(self, action: np.ndarray) -> StepResult:
        """Apply [vx, vy, vz, yaw_rate] and advance simulation."""
        action = np.clip(
            action,
            [-self.max_velocity] * 3 + [-self.max_yaw_rate],
            [self.max_velocity] * 3 + [self.max_yaw_rate],
        )
        vx, vy, vz, yaw_rate = action

        if self._use_pybullet:
            assert self.drone_id is not None and self.client_id is not None
            pos, orn = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.client_id
            )
            yaw = p.getEulerFromQuaternion(orn)[2] + yaw_rate * self.timestep
            new_pos = [
                pos[0] + vx * self.timestep,
                pos[1] + vy * self.timestep,
                pos[2] + vz * self.timestep,
            ]
            p.resetBasePositionAndOrientation(
                self.drone_id, new_pos, p.getQuaternionFromEuler([0, 0, yaw]),
                physicsClientId=self.client_id,
            )
            p.resetBaseVelocity(
                self.drone_id, [vx, vy, vz], [0, 0, yaw_rate],
                physicsClientId=self.client_id,
            )
            p.stepSimulation(physicsClientId=self.client_id)
        else:
            self._vel = np.array([vx, vy, vz], dtype=np.float32)
            self._pos += self._vel * self.timestep
            self._yaw += yaw_rate * self.timestep

        collision = self.check_collision()
        state = self._read_state(collision)
        self.trajectory.append(state.position.copy())
        self._prev_action = action.copy()
        return StepResult(state=state)

    def _read_state(self, collision: bool) -> DroneState:
        if self._use_pybullet:
            assert self.drone_id is not None and self.client_id is not None
            pos, orn = p.getBasePositionAndOrientation(
                self.drone_id, physicsClientId=self.client_id
            )
            vel, _ = p.getBaseVelocity(self.drone_id, physicsClientId=self.client_id)
            yaw = float(p.getEulerFromQuaternion(orn)[2])
            return DroneState(
                position=np.array(pos, dtype=np.float32),
                velocity=np.array(vel, dtype=np.float32),
                yaw=yaw, collision=collision,
            )
        return DroneState(
            position=self._pos.copy(),
            velocity=self._vel.copy(),
            yaw=self._yaw, collision=collision,
        )

    def check_collision(self) -> bool:
        """Detect building or ground collision using footprints."""
        state = self._read_state(False)
        from shapely.geometry import Point

        pt = Point(state.position[0], state.position[1])
        half = self.drone_size / 2.0
        for b in self.buildings:
            if b.polygon.contains(pt) and state.position[2] < b.height_m + half:
                return True
        if state.position[2] < 0.5:
            return True

        if self._use_pybullet and self.drone_id is not None and self.client_id is not None:
            for c in p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client_id):
                if c[2] in self.building_body_ids:
                    return True
        return False

    def nearest_obstacle_distance(self) -> float:
        from shapely.geometry import Point

        pos = self._read_state(False).position
        pt = Point(pos[0], pos[1])
        dists = [b.polygon.distance(pt) for b in self.buildings]
        return float(min(dists)) if dists else 999.0

    def render(self) -> None:
        """Render frame (GUI mode)."""
        pass

    def disconnect(self) -> None:
        if self._use_pybullet and self.client_id is not None:
            p.disconnect(self.client_id)
            self.client_id = None

    def get_topdown_raster(self, size: int = 64) -> np.ndarray:
        pos = self._read_state(False).position
        span = self.bounds.width / 4.0
        x0, y0 = pos[0] - span / 2, pos[1] - span / 2
        cell = span / size
        raster = np.zeros((size, size), dtype=np.float32)
        from shapely.geometry import Point

        for iy in range(size):
            for ix in range(size):
                cx = x0 + (ix + 0.5) * cell
                cy = y0 + (iy + 0.5) * cell
                pt = Point(cx, cy)
                for b in self.buildings:
                    if b.polygon.contains(pt) and pos[2] <= b.height_m:
                        raster[iy, ix] = 1.0
                        break
        return raster

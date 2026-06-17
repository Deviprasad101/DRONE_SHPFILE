"""Gymnasium environment for autonomous drone navigation."""

from __future__ import annotations

import logging
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from data_loader.geojson_loader import load_and_convert
from occupancy.voxel_map import VoxelMap, build_voxel_map
from planner.astar import PathPlan, astar_plan, nearest_waypoint_index
from simulation.simulator import DroneSimulator

logger = logging.getLogger(__name__)


class DroneNavigationEnv(gym.Env):
    """Production drone navigation environment with A* waypoints and voxel occupancy."""

    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        geo = config.get("geojson", {})
        sim = config.get("simulation", {})
        env = config.get("environment", {})
        plan = config.get("planner", {})
        occ_cfg = config.get("occupancy", {})

        self.buildings, self.frame, bounds = load_and_convert(
            geo.get("path", "data/buildings.geojson"),
            default_height_m=geo.get("default_height_m", 15.0),
            max_buildings=geo.get("max_buildings"),
            origin_lat=config.get("coordinates", {}).get("origin_lat"),
            origin_lon=config.get("coordinates", {}).get("origin_lon"),
        )
        self.bounds = bounds.with_margin(sim.get("ground_size_margin", 50.0))

        self.voxel_map: VoxelMap = build_voxel_map(
            self.buildings,
            self.bounds,
            resolution_m=occ_cfg.get("resolution_m", plan.get("grid_resolution_m", 5.0)),
            max_height_m=sim.get("safe_altitude_max", 150.0),
            clearance_m=plan.get("clearance_m", 3.0),
        )

        self.sim = DroneSimulator(
            buildings=self.buildings,
            bounds=self.bounds,
            timestep=sim.get("timestep", 0.02),
            max_velocity=sim.get("max_velocity", 8.0),
            max_yaw_rate=sim.get("max_yaw_rate", 1.5),
            drone_mass=sim.get("drone_mass", 1.5),
            drone_size=sim.get("drone_size", 0.4),
            gui=sim.get("gui", False),
        )

        self.max_steps = env.get("max_episode_steps", 500)
        self.goal_tolerance = env.get("goal_tolerance_m", 3.0)
        self.occ_size = env.get("occupancy_grid_size", 16)
        self.image_size = env.get("observation_image_size", 64)
        self.reward_cfg = env.get("reward", {})
        self.safe_alt_min = sim.get("safe_altitude_min", 5.0)
        self.safe_alt_max = sim.get("safe_altitude_max", 150.0)
        self.plan_cfg = plan
        self.flight_altitude = plan.get("flight_altitude_m", 40.0)

        max_v = sim.get("max_velocity", 8.0)
        max_yaw = sim.get("max_yaw_rate", 1.5)
        self.action_space = spaces.Box(
            low=np.array([-max_v, -max_v, -max_v, -max_yaw], np.float32),
            high=np.array([max_v, max_v, max_v, max_yaw], np.float32),
        )
        # pos(3)+vel(3)+goal(3)+dist(1)+obstacle(1)+waypoint(3)+local_occ(N^2) = 14+N^2
        vec_dim = 14 + self.occ_size ** 2
        self.observation_space = spaces.Dict({
            "vector": spaces.Box(-np.inf, np.inf, (vec_dim,), np.float32),
            "image": spaces.Box(0, 1, (1, self.image_size, self.image_size), np.float32),
        })

        self.goal: np.ndarray | None = None
        self.planned_path: PathPlan | None = None
        self._step_count = 0
        self._prev_dist = 0.0
        self._prev_altitude = 0.0
        self._episode_reward = 0.0

    def _sample_free(self, altitude: float) -> tuple[float, float, float]:
        occ = self.voxel_map.cost_map_2d(altitude)
        free = np.argwhere(occ == 0)
        if len(free) == 0:
            cx, cy = self.bounds.center
            return cx, cy, altitude
        iy, ix = free[self.np_random.integers(len(free))]
        x, y, _ = self.voxel_map.voxel_to_world(int(ix), int(iy), 0)
        return x, y, altitude

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        alt = self.flight_altitude

        if "start" in options and "goal" in options:
            start, goal = tuple(options["start"]), tuple(options["goal"])
        else:
            start = self._sample_free(alt)
            goal = self._sample_free(alt)
            for _ in range(50):
                if np.linalg.norm(np.array(start[:2]) - np.array(goal[:2])) >= 100:
                    break
                goal = self._sample_free(alt)

        self.goal = np.array(goal, dtype=np.float32)
        self.planned_path = astar_plan(
            start[:2], goal[:2], self.voxel_map,
            flight_altitude_m=alt,
            clearance_m=self.plan_cfg.get("clearance_m", 3.0),
        )
        state = self.sim.reset(start, goal, self.planned_path)
        self._step_count = 0
        self._prev_dist = float(np.linalg.norm(self.goal - state.position))
        self._prev_altitude = float(state.position[2])
        self._episode_reward = 0.0
        return self._obs(state), {
            "planned_path": self.planned_path.waypoints,
            "start": start, "goal": goal,
        }

    def step(self, action: np.ndarray):
        result = self.sim.step(action)
        state = result.state
        self._step_count += 1
        dist = float(np.linalg.norm(self.goal - state.position))
        progress = self._prev_dist - dist
        self._prev_dist = dist

        reward, rinfo = self._reward(state, action, progress, dist)
        self._episode_reward += reward

        terminated = dist < self.goal_tolerance or state.collision
        terminated = terminated or not self.bounds.contains(state.position[0], state.position[1])
        truncated = self._step_count >= self.max_steps
        self._prev_altitude = float(state.position[2])

        return self._obs(state), reward, terminated, truncated, {
            **rinfo, "distance_to_goal": dist, "collision": state.collision,
            "episode_reward": self._episode_reward, "position": state.position.tolist(),
        }

    def _reward(self, state, action, progress, dist):
        rc = self.reward_cfg
        r = progress * rc.get("progress_scale", 2.0)
        r += rc.get("goal_bonus", 100.0) if dist < self.goal_tolerance else 0.0

        # Waypoint following
        if self.planned_path and self.planned_path.waypoints:
            idx = nearest_waypoint_index(tuple(state.position), self.planned_path.waypoints)
            wp = np.array(self.planned_path.waypoints[min(idx + 1, len(self.planned_path.waypoints) - 1)])
            wp_dist = float(np.linalg.norm(state.position - wp))
            r += rc.get("waypoint_scale", 0.5) * max(0, 10.0 - wp_dist)

        # Smooth flight
        smooth = 0.0
        if self.sim._prev_action is not None:
            smooth = float(np.linalg.norm(action - self.sim._prev_action))
        r -= rc.get("smoothness_penalty_scale", 0.1) * smooth

        # Altitude change penalty
        alt_change = abs(state.position[2] - self._prev_altitude)
        r -= rc.get("altitude_change_penalty", 0.05) * alt_change

        r -= rc.get("collision_penalty", 50.0) if state.collision else 0.0
        r -= rc.get("boundary_penalty", 30.0) if not self.bounds.contains(state.position[0], state.position[1]) else 0.0
        r -= rc.get("time_penalty", 0.05)
        r -= rc.get("energy_penalty_scale", 0.01) * float(np.linalg.norm(action[:3]))

        return r, {
            "progress": progress * rc.get("progress_scale", 2.0),
            "smoothness_penalty": smooth * rc.get("smoothness_penalty_scale", 0.1),
            "energy_penalty": float(np.linalg.norm(action[:3])) * rc.get("energy_penalty_scale", 0.01),
        }

    def _obs(self, state) -> dict[str, np.ndarray]:
        dist = float(np.linalg.norm(self.goal - state.position))
        obs_dist = self.voxel_map.nearest_obstacle_distance(
            state.position[0], state.position[1], state.position[2]
        )
        local = self.voxel_map.local_slice(
            state.position[0], state.position[1], state.position[2], self.occ_size
        )
        wp = np.zeros(3, np.float32)
        if self.planned_path and self.planned_path.waypoints:
            idx = nearest_waypoint_index(tuple(state.position), self.planned_path.waypoints)
            wp = np.array(self.planned_path.waypoints[min(idx + 1, len(self.planned_path.waypoints) - 1)], np.float32)

        vector = np.concatenate([
            state.position, state.velocity, self.goal,
            [dist, obs_dist], wp, local,
        ]).astype(np.float32)
        image = self.sim.get_topdown_raster(self.image_size)[None]
        return {"vector": vector, "image": image}

    def render(self):
        self.sim.render()

    def close(self):
        self.sim.disconnect()

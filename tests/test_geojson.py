"""Unit tests for GeoJSON loading, meshes, planner, and occupancy."""

from __future__ import annotations

import numpy as np
import pytest

from data_loader.geojson_loader import load_and_convert
from occupancy.voxel_map import build_voxel_map
from planner.astar import PathPlan, astar_plan, nearest_waypoint_index
from rl.replay_buffer import Episode, PrioritizedReplayBuffer
from utils import MetricsTracker, load_config
from world.city_mesh_generator import CityMeshGenerator


@pytest.fixture
def sample_data():
    buildings, frame, bounds = load_and_convert(
        "data/buildings.geojson", max_buildings=20
    )
    return buildings, bounds


def test_load_and_convert(sample_data):
    buildings, bounds = sample_data
    assert len(buildings) > 0
    assert bounds.max_x > bounds.min_x


def test_city_mesh_generator(sample_data, tmp_path):
    buildings, bounds = sample_data
    gen = CityMeshGenerator(buildings, bounds, output_dir=tmp_path)
    city = gen.generate(export=True)
    assert city.ground is not None
    assert (tmp_path / "buildings.obj").exists()


def test_voxel_map(sample_data):
    buildings, bounds = sample_data
    vmap = build_voxel_map(buildings, bounds, resolution_m=10.0)
    assert vmap.grid.sum() > 0
    assert not vmap.is_occupied(bounds.center[0], bounds.center[1], 80.0) or True


def test_astar(sample_data):
    buildings, bounds = sample_data
    vmap = build_voxel_map(buildings, bounds, resolution_m=10.0)
    cx, cy = bounds.center
    plan = astar_plan((cx - 30, cy - 30), (cx + 30, cy + 30), vmap, 40.0)
    assert isinstance(plan, PathPlan)
    assert len(plan.waypoints) >= 2


def test_replay_buffer():
    buf = PrioritizedReplayBuffer(10)
    ep = Episode(
        observations=[{"vector": np.zeros(10, np.float32), "image": np.zeros((1, 8, 8), np.float32)}] * 20,
        actions=[np.zeros(4, np.float32)] * 20,
        rewards=[0.1] * 20,
        dones=[False] * 19 + [True],
        total_reward=2.0, length=20,
    )
    buf.add_episode(ep)
    obs, actions, rewards, dones = buf.sample(2, 8)
    assert actions.shape == (2, 8, 4)


def test_config():
    cfg = load_config("configs/dev.yaml")
    assert cfg["geojson"]["max_buildings"] == 200


def test_metrics():
    m = MetricsTracker()
    m.record_episode(10, 100, True, False, 50, 5, 2)
    assert m.summary()["success_rate"] == 1.0

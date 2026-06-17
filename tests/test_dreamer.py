"""Tests for DreamerV3 components."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rl.actor import Actor
from rl.critic import Critic, LambdaReturn
from rl.dreamer import DreamerAgent
from rl.world_model import WorldModel


@pytest.fixture
def agent_config():
    return {
        "dreamer": {
            "device": "cpu",
            "batch_size": 2,
            "batch_length": 8,
            "hidden_dim": 128,
            "deter_dim": 64,
            "stoch_dim": 8,
            "imagination_horizon": 4,
            "gamma": 0.99,
            "lambda_": 0.95,
            "replay_capacity": 100,
        }
    }


def test_world_model_forward():
    vector_dim = 50
    image_shape = (1, 16, 16)
    batch, time = 2, 8
    wm = WorldModel(vector_dim, image_shape, action_dim=4, hidden_dim=128, deter_dim=64, stoch_dim=8)
    vector = torch.randn(batch, time, vector_dim)
    image = torch.rand(batch, time, *image_shape)
    action = torch.randn(batch, time, 4)
    out = wm(vector, image, action)
    assert out["recon_vector"].shape == vector.shape
    assert out["reward_pred"].shape == (batch, time)


def test_actor_critic():
    feat_dim = 100
    actor = Actor(feat_dim, action_dim=4, hidden_dim=64)
    critic = Critic(feat_dim, hidden_dim=64)
    feat = torch.randn(4, feat_dim)
    action, log_prob = actor.sample(feat)
    assert action.shape == (4, 4)
    value = critic(feat)
    assert value.shape == (4,)


def test_lambda_return():
    rewards = torch.ones(2, 5)
    values = torch.zeros(2, 5)
    continues = torch.ones(2, 5)
    ret = LambdaReturn.compute(rewards, values, continues, gamma=0.9, lambda_=0.9)
    assert ret.shape == (2, 5)
    assert ret[:, 0].mean() > 0


def test_dreamer_agent_action(agent_config):
    agent = DreamerAgent(
        vector_dim=50,
        image_shape=(1, 16, 16),
        action_dim=4,
        action_low=np.array([-1, -1, -1, -1], dtype=np.float32),
        action_high=np.array([1, 1, 1, 1], dtype=np.float32),
        config=agent_config,
        device="cpu",
    )
    obs = {
        "vector": np.random.randn(50).astype(np.float32),
        "image": np.random.rand(1, 16, 16).astype(np.float32),
    }
    action = agent.select_action(obs)
    assert action.shape == (4,)

"""Critic network for DreamerV3 value estimation in latent space."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    """State-value function V(s) in latent feature space."""

    def __init__(self, feat_dim: int, hidden_dim: int = 400) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class LambdaReturn:
    """Compute lambda returns for actor-critic training."""

    @staticmethod
    def compute(
        rewards: torch.Tensor,
        values: torch.Tensor,
        continues: torch.Tensor,
        gamma: float = 0.997,
        lambda_: float = 0.95,
    ) -> torch.Tensor:
        """TD(lambda) returns along time dimension."""
        batch, time = rewards.shape
        returns = torch.zeros_like(rewards)
        next_value = values[:, -1]
        for t in reversed(range(time)):
            next_value = rewards[:, t] + gamma * continues[:, t] * (
                (1 - lambda_) * values[:, t] + lambda_ * next_value
            )
            returns[:, t] = next_value
        return returns

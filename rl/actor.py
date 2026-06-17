"""Actor network for DreamerV3 continuous control."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    """Gaussian policy in latent space."""

    def __init__(
        self,
        feat_dim: int,
        action_dim: int = 4,
        hidden_dim: int = 400,
        min_std: float = 0.1,
        max_std: float = 1.0,
    ) -> None:
        super().__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(features)
        mean = self.mean_head(h)
        std = self.max_std * torch.sigmoid(self.std_head(h)) + self.min_std
        return mean, std

    def dist(self, features: torch.Tensor) -> Normal:
        mean, std = self.forward(features)
        return Normal(mean, std)

    def sample(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        distribution = self.dist(features)
        action = distribution.rsample()
        log_prob = distribution.log_prob(action).sum(-1)
        return action, log_prob

    def entropy(self, features: torch.Tensor) -> torch.Tensor:
        return self.dist(features).entropy().sum(-1)

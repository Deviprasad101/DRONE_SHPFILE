"""Prioritized experience replay buffer for DreamerV3."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Episode:
    """Single episode trajectory."""

    observations: list[dict[str, np.ndarray]]
    actions: list[np.ndarray]
    rewards: list[float]
    dones: list[bool]
    total_reward: float = 0.0
    length: int = 0
    success: bool = False
    collision: bool = False
    metadata: dict = field(default_factory=dict)


class PrioritizedReplayBuffer:
    """Episode replay buffer with optional prioritized sampling."""

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta: float = 0.4,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.episodes: list[Episode] = []
        self.priorities: list[float] = []
        self._position = 0

    def add_episode(self, episode: Episode) -> None:
        """Add a completed episode to the buffer."""
        priority = (abs(episode.total_reward) + 1e-6) ** self.alpha
        if len(self.episodes) < self.capacity:
            self.episodes.append(episode)
            self.priorities.append(priority)
        else:
            self.episodes[self._position] = episode
            self.priorities[self._position] = priority
        self._position = (self._position + 1) % self.capacity

    def sample(
        self, batch_size: int, batch_length: int
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Sample batched sequences for world model training."""
        if not self.episodes:
            raise ValueError("Replay buffer is empty")

        probs = np.array(self.priorities[: len(self.episodes)], dtype=np.float64)
        probs /= probs.sum()
        indices = self.np_random_choice(len(self.episodes), batch_size, probs)

        batch_obs: dict[str, list] = {}
        batch_actions: list[np.ndarray] = []
        batch_rewards: list[np.ndarray] = []
        batch_dones: list[np.ndarray] = []

        for idx in indices:
            ep = self.episodes[idx]
            start = 0
            if len(ep.observations) > batch_length:
                start = np.random.randint(0, len(ep.observations) - batch_length)

            seq_obs = ep.observations[start : start + batch_length]
            seq_actions = ep.actions[start : start + batch_length]
            seq_rewards = ep.rewards[start : start + batch_length]
            seq_dones = ep.dones[start : start + batch_length]

            for key in seq_obs[0]:
                if key not in batch_obs:
                    batch_obs[key] = []
                batch_obs[key].append(
                    np.stack([o[key] for o in seq_obs], axis=0)
                )
            batch_actions.append(np.stack(seq_actions, axis=0))
            batch_rewards.append(np.array(seq_rewards, dtype=np.float32))
            batch_dones.append(np.array(seq_dones, dtype=np.float32))

        obs_arrays = {k: np.stack(v, axis=0) for k, v in batch_obs.items()}
        actions = np.stack(batch_actions, axis=0).astype(np.float32)
        rewards = np.stack(batch_rewards, axis=0).astype(np.float32)
        dones = np.stack(batch_dones, axis=0).astype(np.float32)
        return obs_arrays, actions, rewards, dones

    @staticmethod
    def np_random_choice(n: int, size: int, probs: np.ndarray) -> np.ndarray:
        return np.random.choice(n, size=size, replace=True, p=probs)

    def __len__(self) -> int:
        return len(self.episodes)

    def stats(self) -> dict[str, Any]:
        if not self.episodes:
            return {}
        rewards = [ep.total_reward for ep in self.episodes]
        lengths = [ep.length for ep in self.episodes]
        successes = [ep.success for ep in self.episodes]
        collisions = [ep.collision for ep in self.episodes]
        return {
            "episodes": len(self.episodes),
            "mean_reward": float(np.mean(rewards)),
            "mean_length": float(np.mean(lengths)),
            "success_rate": float(np.mean(successes)),
            "collision_rate": float(np.mean(collisions)),
        }

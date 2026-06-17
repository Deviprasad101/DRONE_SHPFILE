"""Utility helpers for configuration and metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> dict[str, Any]:
    """Load YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@dataclass
class MetricsTracker:
    """Track training and evaluation metrics."""

    success_count: int = 0
    collision_count: int = 0
    episode_count: int = 0
    total_reward: float = 0.0
    total_length: int = 0
    total_path_length: float = 0.0
    total_energy: float = 0.0
    total_flight_time: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)

    def record_episode(
        self,
        reward: float,
        length: int,
        success: bool,
        collision: bool,
        path_length: float = 0.0,
        energy: float = 0.0,
        flight_time: float = 0.0,
    ) -> None:
        self.episode_count += 1
        self.total_reward += reward
        self.total_length += length
        self.success_count += int(success)
        self.collision_count += int(collision)
        self.total_path_length += path_length
        self.total_energy += energy
        self.total_flight_time += flight_time
        self.history.append(
            {
                "reward": reward,
                "length": float(length),
                "success": float(success),
                "collision": float(collision),
                "path_length": path_length,
                "energy": energy,
                "flight_time": flight_time,
            }
        )

    def summary(self) -> dict[str, float]:
        n = max(self.episode_count, 1)
        return {
            "episodes": self.episode_count,
            "success_rate": self.success_count / n,
            "collision_rate": self.collision_count / n,
            "average_reward": self.total_reward / n,
            "average_length": self.total_length / n,
            "average_path_length": self.total_path_length / n,
            "average_energy": self.total_energy / n,
            "average_flight_time": self.total_flight_time / n,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                {"summary": self.summary(), "history": self.history},
                handle,
                indent=2,
            )

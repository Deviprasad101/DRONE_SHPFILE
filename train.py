"""Train DreamerV3 agent for autonomous drone navigation."""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from env.drone_navigation_env import DroneNavigationEnv
from rl.dreamer import DreamerAgent
from rl.replay_buffer import Episode
from utils import MetricsTracker, load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_episode(env: DroneNavigationEnv, agent: DreamerAgent, explore: bool = True) -> Episode:
    """Collect one episode of experience."""
    obs, info = env.reset()
    agent.reset_state()
    episode = Episode(observations=[], actions=[], rewards=[], dones=[])
    energy = 0.0
    positions = [info["start"]]

    while True:
        action = agent.select_action(obs, explore=explore)
        obs, reward, terminated, truncated, step_info = env.step(action)
        done = terminated or truncated
        episode.observations.append(obs)
        episode.actions.append(action)
        episode.rewards.append(reward)
        episode.dones.append(done)
        energy += float(np.linalg.norm(action[:3]))
        positions.append(step_info["position"])
        if done:
            episode.total_reward = sum(episode.rewards)
            episode.length = len(episode.rewards)
            episode.success = step_info["distance_to_goal"] < env.goal_tolerance
            episode.collision = step_info.get("collision", False)
            break

    episode.metadata = {
        "energy": energy,
        "path_length": _path_length(positions),
        "flight_time": episode.length * env.sim.timestep,
        "planned_path": info.get("planned_path", []),
        "trajectory": positions,
    }
    return episode


def _path_length(positions: list) -> float:
    return sum(
        float(np.linalg.norm(np.array(b) - np.array(a)))
        for a, b in zip(positions[:-1], positions[1:])
    )


def evaluate_agent(agent: DreamerAgent, env: DroneNavigationEnv, episodes: int) -> MetricsTracker:
    metrics = MetricsTracker()
    for _ in range(episodes):
        ep = run_episode(env, agent, explore=False)
        meta = getattr(ep, "metadata", {})
        metrics.record_episode(
            ep.total_reward, ep.length, ep.success, ep.collision,
            meta.get("path_length", 0.0), meta.get("energy", 0.0), meta.get("flight_time", 0.0),
        )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train DreamerV3 drone navigation")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    if torch.cuda.is_available() and config.get("dreamer", {}).get("device") == "cpu":
        config["dreamer"]["device"] = "cuda"

    train_cfg = config.get("training", {})
    dreamer_cfg = config.get("dreamer", {})
    log_dir = Path(train_cfg.get("log_dir", "logs"))
    ckpt_dir = Path(train_cfg.get("checkpoint_dir", "checkpoints"))
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir / "tensorboard"))
    logger.info("Building environment from GeoJSON...")
    env = DroneNavigationEnv(config)

    agent = DreamerAgent(
        vector_dim=env.observation_space["vector"].shape[0],
        image_shape=tuple(env.observation_space["image"].shape),
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        config=config,
    )

    resume_path = args.checkpoint or train_cfg.get("resume_checkpoint")
    if (args.resume or train_cfg.get("resume")) and resume_path and Path(resume_path).exists():
        agent.load(resume_path)

    total_steps = train_cfg.get("total_steps", 50_000)
    train_ratio = dreamer_cfg.get("train_ratio", 8)
    batch_length = dreamer_cfg.get("batch_length", 32)
    pretrain = dreamer_cfg.get("pretrain", 100)
    log_every = dreamer_cfg.get("log_every", 10)
    eval_every = dreamer_cfg.get("eval_every", 50)
    ckpt_every = dreamer_cfg.get("checkpoint_every", 100)

    metrics = MetricsTracker()
    episode_idx = 0
    t0 = time.time()

    while agent.global_step < total_steps:
        episode = run_episode(env, agent, explore=True)
        agent.add_episode(episode)
        meta = getattr(episode, "metadata", {})
        metrics.record_episode(
            episode.total_reward, episode.length, episode.success, episode.collision,
            meta.get("path_length", 0.0), meta.get("energy", 0.0), meta.get("flight_time", 0.0),
        )
        episode_idx += 1

        if len(agent.buffer) >= pretrain:
            for _ in range(train_ratio):
                if agent.global_step >= total_steps:
                    break
                tm = agent.train_step(batch_length)
                if agent.global_step % log_every == 0:
                    for k, v in tm.items():
                        writer.add_scalar(f"train/{k}", v, agent.global_step)

        if episode_idx % log_every == 0:
            s = metrics.summary()
            logger.info("Ep %d | reward=%.2f success=%.2f collision=%.2f", episode_idx, s["average_reward"], s["success_rate"], s["collision_rate"])
            for k, v in s.items():
                writer.add_scalar(f"episode/{k}", v, episode_idx)

        if episode_idx % eval_every == 0:
            ev = evaluate_agent(agent, env, train_cfg.get("eval_episodes", 5))
            for k, v in ev.summary().items():
                writer.add_scalar(f"eval/{k}", v, agent.global_step)

        if episode_idx % ckpt_every == 0:
            agent.save(ckpt_dir / f"dreamer_step_{agent.global_step}.pt")
            metrics.save(log_dir / "metrics.json")

    agent.save(ckpt_dir / "dreamer_final.pt")
    metrics.save(log_dir / "metrics_final.json")
    writer.close()
    env.close()
    logger.info("Done in %.1fs | %s", time.time() - t0, metrics.summary())


if __name__ == "__main__":
    main()

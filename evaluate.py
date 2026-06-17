"""Evaluate trained DreamerV3 agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from env.drone_navigation_env import DroneNavigationEnv
from rl.dreamer import DreamerAgent
from train import run_episode
from utils import MetricsTracker, load_config
from visualization.visualizer import NavigationVisualizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DreamerV3 agent")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--start", nargs=3, type=float, default=None)
    parser.add_argument("--goal", nargs=3, type=float, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_cfg = config.get("evaluation", {})
    out_dir = Path(eval_cfg.get("output_dir", "logs/eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    env = DroneNavigationEnv(config)
    agent = DreamerAgent(
        vector_dim=env.observation_space["vector"].shape[0],
        image_shape=tuple(env.observation_space["image"].shape),
        action_dim=env.action_space.shape[0],
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        config=config,
    )
    agent.load(args.checkpoint)

    metrics = MetricsTracker()
    results = []

    for i in range(args.episodes or eval_cfg.get("num_episodes", 10)):
        opts = {}
        if args.start is not None:
            opts["start"] = args.start
        if args.goal is not None:
            opts["goal"] = args.goal

        obs, info = env.reset(options=opts or None)
        agent.reset_state()
        positions = [info["start"]]
        reward_sum = 0.0
        collision = False

        while True:
            action = agent.select_action(obs, explore=False)
            obs, r, term, trunc, si = env.step(action)
            reward_sum += r
            positions.append(si["position"])
            collision |= si.get("collision", False)
            if term or trunc:
                success = si["distance_to_goal"] < env.goal_tolerance
                break

        plen = sum(float(np.linalg.norm(np.array(positions[j+1]) - np.array(positions[j]))) for j in range(len(positions)-1))
        metrics.record_episode(reward_sum, len(positions)-1, success, collision, plen, 0.0, (len(positions)-1)*env.sim.timestep)
        results.append({"start": info["start"], "goal": info["goal"], "trajectory": positions, "success": success, "reward": reward_sum})

        if args.visualize:
            viz = NavigationVisualizer(env.buildings, env.bounds, config.get("visualization", {}).get("show_gui", False))
            viz.show(info["start"], info["goal"], info.get("planned_path"), positions,
                     title=f"Episode {i+1}", output_path=out_dir / f"episode_{i+1}.png")

    summary = metrics.summary()
    with (out_dir / "eval_results.json").open("w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)
    logger.info("Evaluation: %s", summary)
    env.close()


if __name__ == "__main__":
    main()

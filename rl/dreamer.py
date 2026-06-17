"""DreamerV3 agent orchestrating world model, actor, and critic."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from rl.actor import Actor
from rl.critic import Critic, LambdaReturn
from rl.replay_buffer import Episode, PrioritizedReplayBuffer
from rl.world_model import WorldModel

logger = logging.getLogger(__name__)


class DreamerAgent:
    """DreamerV3 model-based RL agent."""

    def __init__(
        self,
        vector_dim: int,
        image_shape: tuple[int, int, int],
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        config: dict[str, Any],
        device: str | None = None,
    ) -> None:
        self.config = config
        dc = config.get("dreamer", {})
        self.device = torch.device(
            device or dc.get("device", "cpu")
        )
        self.action_dim = action_dim
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=self.device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=self.device)
        self.gamma = dc.get("gamma", 0.997)
        self.lambda_ = dc.get("lambda_", 0.95)
        self.imagination_horizon = dc.get("imagination_horizon", 15)
        self.actor_entropy = dc.get("actor_entropy", 3e-4)
        self.grad_clip = dc.get("grad_clip", 100.0)

        self.world_model = WorldModel(
            vector_dim=vector_dim,
            image_shape=image_shape,
            action_dim=action_dim,
            embed_dim=dc.get("hidden_dim", 400) // 2 + 56,
            stoch_dim=dc.get("stoch_dim", 32),
            deter_dim=dc.get("deter_dim", 200),
            hidden_dim=dc.get("hidden_dim", 400),
        ).to(self.device)

        feat_dim = self.world_model.feat_dim
        self.actor = Actor(
            feat_dim=feat_dim,
            action_dim=action_dim,
            hidden_dim=dc.get("hidden_dim", 400),
        ).to(self.device)
        self.critic = Critic(
            feat_dim=feat_dim,
            hidden_dim=dc.get("hidden_dim", 400),
        ).to(self.device)

        self.model_opt = torch.optim.Adam(
            self.world_model.parameters(), lr=dc.get("model_lr", 3e-4)
        )
        self.actor_opt = torch.optim.Adam(
            self.actor.parameters(), lr=dc.get("actor_lr", 8e-5)
        )
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=dc.get("critic_lr", 8e-5)
        )

        self.buffer = PrioritizedReplayBuffer(
            capacity=dc.get("replay_capacity", 100_000),
            alpha=dc.get("prioritized_alpha", 0.6),
            beta=dc.get("prioritized_beta", 0.4),
        )
        self._state: dict[str, torch.Tensor] | None = None
        self.global_step = 0

    def _scale_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map tanh-squashed actions to environment bounds."""
        squashed = torch.tanh(action)
        return self.action_low + 0.5 * (squashed + 1.0) * (self.action_high - self.action_low)

    def _to_tensor_obs(self, obs: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
            "vector": torch.tensor(obs["vector"], dtype=torch.float32, device=self.device),
            "image": torch.tensor(obs["image"], dtype=torch.float32, device=self.device),
        }

    @torch.no_grad()
    def select_action(
        self, obs: dict[str, np.ndarray], explore: bool = True
    ) -> np.ndarray:
        """Select action from current observation."""
        tensor_obs = self._to_tensor_obs(obs)
        vector = tensor_obs["vector"].unsqueeze(0)
        image = tensor_obs["image"].unsqueeze(0)
        action_zero = torch.zeros(1, 1, self.action_dim, device=self.device)

        if self._state is None:
            self._state = self.world_model.rssm.initial(1, self.device)

        embed = self.world_model.encoder(vector, image)
        _, post = self.world_model.rssm.observe(
            embed.unsqueeze(1), action_zero, self._state
        )
        self._state = {
            k: post[k][:, -1] for k in post
        }
        feat = self.world_model.features(self._state)

        if explore:
            action, _ = self.actor.sample(feat)
        else:
            action, _ = self.actor.sample(feat)
            action = self.actor.dist(feat).mean

        scaled = self._scale_action(action)
        return scaled.squeeze(0).cpu().numpy().astype(np.float32)

    def reset_state(self) -> None:
        """Reset recurrent state between episodes."""
        self._state = None

    def train_step(self, batch_length: int) -> dict[str, float]:
        """Perform one joint training step on a replay batch."""
        obs, actions, rewards, dones = self.buffer.sample(
            self.config.get("dreamer", {}).get("batch_size", 16),
            batch_length,
        )

        vector = torch.tensor(obs["vector"], device=self.device)
        image = torch.tensor(obs["image"], device=self.device)
        actions_t = torch.tensor(actions, device=self.device)
        rewards_t = torch.tensor(rewards, device=self.device)
        continues_t = 1.0 - torch.tensor(dones, device=self.device)

        # World model training
        wm_out = self.world_model(vector, image, actions_t)
        recon_loss = self.world_model.reconstruction_loss(
            vector, image, wm_out["recon_vector"], wm_out["recon_image"]
        )
        kl = self.world_model.kl_loss(wm_out["prior"], wm_out["post"])
        reward_loss = F.mse_loss(wm_out["reward_pred"], rewards_t)
        continue_loss = F.binary_cross_entropy(wm_out["continue_pred"], continues_t)
        model_loss = recon_loss + 0.1 * kl + reward_loss + continue_loss

        self.model_opt.zero_grad()
        model_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), self.grad_clip)
        self.model_opt.step()

        # Actor-critic via imagination
        start_feat = wm_out["feat"].detach()[:, -1]
        feats, imagined_actions, imagined_rewards = self.world_model.imagine_ahead(
            start_feat, self.actor, self.imagination_horizon
        )
        values = self.critic(feats.reshape(-1, feats.shape[-1])).reshape(
            feats.shape[0], feats.shape[1]
        )
        continues = torch.ones_like(imagined_rewards)
        lambda_returns = LambdaReturn.compute(
            imagined_rewards, values, continues, self.gamma, self.lambda_
        )

        critic_loss = F.mse_loss(values, lambda_returns.detach())
        self.critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_opt.step()

        feat_flat = feats.reshape(-1, feats.shape[-1])
        action_sample, log_prob = self.actor.sample(feat_flat)
        actor_values = self.critic(feat_flat)
        actor_loss = -(lambda_returns.reshape(-1) * log_prob).mean()
        actor_loss -= self.actor_entropy * self.actor.entropy(feat_flat).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_opt.step()

        self.global_step += 1
        return {
            "model_loss": float(model_loss.item()),
            "recon_loss": float(recon_loss.item()),
            "kl": float(kl.item()),
            "reward_loss": float(reward_loss.item()),
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "value_mean": float(actor_values.mean().item()),
        }

    def add_episode(self, episode: Episode) -> None:
        self.buffer.add_episode(episode)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "world_model": self.world_model.state_dict(),
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "global_step": self.global_step,
            },
            path,
        )
        logger.info("Saved checkpoint to %s", path)

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.global_step = checkpoint.get("global_step", 0)
        logger.info("Loaded checkpoint from %s (step=%d)", path, self.global_step)

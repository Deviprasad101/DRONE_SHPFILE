"""DreamerV3 world model: encoder, RSSM, decoders."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical

from rl.rssm import RSSM


class MultiEncoder(nn.Module):
    """Encode vector state and image observations."""

    def __init__(
        self,
        vector_dim: int,
        image_shape: tuple[int, int, int] = (1, 64, 64),
        embed_dim: int = 256,
        hidden_dim: int = 400,
    ) -> None:
        super().__init__()
        c, h, w = image_shape
        self.image_cnn = nn.Sequential(
            nn.Conv2d(c, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_out = self.image_cnn(torch.zeros(1, c, h, w)).shape[-1]
        self.vector_mlp = nn.Sequential(
            nn.Linear(vector_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(cnn_out + hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim), nn.SiLU(),
        )

    def forward(self, vector: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return self.fusion(torch.cat([self.image_cnn(image), self.vector_mlp(vector)], dim=-1))


class WorldModel(nn.Module):
    """Full world model with latent dynamics learning."""

    def __init__(
        self,
        vector_dim: int,
        image_shape: tuple[int, int, int],
        action_dim: int = 4,
        embed_dim: int = 256,
        stoch_dim: int = 32,
        deter_dim: int = 200,
        hidden_dim: int = 400,
        classes: int = 32,
    ) -> None:
        super().__init__()
        self.encoder = MultiEncoder(vector_dim, image_shape, embed_dim, hidden_dim)
        self.rssm = RSSM(embed_dim, stoch_dim, deter_dim, hidden_dim, action_dim, classes)
        self.feat_dim = deter_dim + stoch_dim * classes
        c, h, w = image_shape
        self.image_shape = image_shape
        self.vector_decoder = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, vector_dim),
        )
        self.image_decoder = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, c * h * w),
        )
        self.reward_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )
        self.continue_head = nn.Sequential(
            nn.Linear(self.feat_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1),
        )

    def features(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        return torch.cat([state["deter"], state["stoch"]], dim=-1)

    def forward(
        self,
        vector: torch.Tensor,
        image: torch.Tensor,
        action: torch.Tensor,
        reset_state: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, Any]:
        batch, time = vector.shape[:2]
        vector_flat = vector.reshape(batch * time, -1)
        image_flat = image.reshape(batch * time, *image.shape[2:])
        embed = self.encoder(vector_flat, image_flat).reshape(batch, time, -1)
        if reset_state is None:
            reset_state = self.rssm.initial(batch, vector.device)
        prior, post = self.rssm.observe(embed, action, reset_state)
        feat = self.features(post)
        return {
            "prior": prior, "post": post, "feat": feat, "embed": embed,
            "recon_vector": self.vector_decoder(feat),
            "recon_image": self.image_decoder(feat).reshape(batch, time, *self.image_shape),
            "reward_pred": self.reward_head(feat).squeeze(-1),
            "continue_pred": torch.sigmoid(self.continue_head(feat).squeeze(-1)),
        }

    def kl_loss(self, prior: dict, post: dict) -> torch.Tensor:
        kl = torch.distributions.kl_divergence(
            OneHotCategorical(logits=post["logit"]),
            OneHotCategorical(logits=prior["logit"]),
        )
        return kl.sum(dim=(-2, -1)).mean()

    def reconstruction_loss(self, vector, image, recon_vector, recon_image) -> torch.Tensor:
        return F.mse_loss(recon_vector, vector) + F.mse_loss(recon_image, image)

    def imagine_ahead(self, start_feat, actor, horizon: int):
        deter = start_feat[:, : self.rssm.deter_dim]
        stoch = start_feat[:, self.rssm.deter_dim :]
        state = {"deter": deter, "stoch": stoch}
        actions, rewards, feats = [], [], []
        for _ in range(horizon):
            feat = self.features(state)
            feats.append(feat)
            action, _ = actor.sample(feat)
            actions.append(action)
            imagined = self.rssm.imagine(action.unsqueeze(1), state)
            state = {k: imagined[k][:, -1] for k in imagined}
            rewards.append(self.reward_head(feat).squeeze(-1))
        return torch.stack(feats, 1), torch.stack(actions, 1), torch.stack(rewards, 1)

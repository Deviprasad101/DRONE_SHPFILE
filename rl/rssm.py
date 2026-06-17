"""Recurrent State Space Model (RSSM) for DreamerV3."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical


class RSSM(nn.Module):
    """Recurrent State Space Model with stochastic and deterministic states."""

    def __init__(
        self,
        embed_dim: int = 256,
        stoch_dim: int = 32,
        deter_dim: int = 200,
        hidden_dim: int = 400,
        action_dim: int = 4,
        classes: int = 32,
    ) -> None:
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.classes = classes
        self.stoch_flat = stoch_dim * classes

        self.gru = nn.GRUCell(embed_dim + action_dim, deter_dim)
        self.prior_net = nn.Sequential(
            nn.Linear(deter_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )
        self.post_net = nn.Sequential(
            nn.Linear(deter_dim + embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.stoch_flat),
        )

    def initial(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            "deter": torch.zeros(batch_size, self.deter_dim, device=device),
            "stoch": torch.zeros(batch_size, self.stoch_flat, device=device),
            "logit": torch.zeros(batch_size, self.stoch_dim, self.classes, device=device),
        }

    def _stats_to_stoch(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = OneHotCategorical(logits=logits)
        sample = dist.sample() + dist.probs - dist.probs.detach()
        stoch = sample.reshape(sample.shape[0], -1)
        return stoch, logits

    def observe(
        self,
        embed: torch.Tensor,
        action: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        priors: list[dict[str, torch.Tensor]] = []
        posts: list[dict[str, torch.Tensor]] = []
        deter = state["deter"]
        time = embed.shape[1]
        for t in range(time):
            deter = self.gru(
                torch.cat([embed[:, t], action[:, t]], dim=-1),
                deter,
            )
            prior_logits = self.prior_net(deter).reshape(-1, self.stoch_dim, self.classes)
            post_logits = self.post_net(
                torch.cat([deter, embed[:, t]], dim=-1)
            ).reshape(-1, self.stoch_dim, self.classes)
            _, prior_logits = self._stats_to_stoch(prior_logits)
            stoch, post_logits = self._stats_to_stoch(post_logits)
            priors.append({"deter": deter, "stoch": stoch, "logit": prior_logits})
            posts.append({"deter": deter, "stoch": stoch, "logit": post_logits})
        prior = {k: torch.stack([p[k] for p in priors], dim=1) for k in priors[0]}
        post = {k: torch.stack([p[k] for p in posts], dim=1) for k in posts[0]}
        return prior, post

    def imagine(
        self,
        action: torch.Tensor,
        state: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        batch, time, action_dim = action.shape
        deters, stochs, logits = [], [], []
        deter = state["deter"]
        for t in range(time):
            deter = self.gru(
                torch.cat([
                    torch.zeros(batch, action_dim, device=action.device),
                    action[:, t],
                ], dim=-1),
                deter,
            )
            prior_logits = self.prior_net(deter).reshape(-1, self.stoch_dim, self.classes)
            stoch, prior_logits = self._stats_to_stoch(prior_logits)
            deters.append(deter)
            stochs.append(stoch)
            logits.append(prior_logits)
        return {
            "deter": torch.stack(deters, dim=1),
            "stoch": torch.stack(stochs, dim=1),
            "logit": torch.stack(logits, dim=1),
        }

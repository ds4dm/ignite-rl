# coding: utf-8

import gym
import torch
import torch.nn as nn
import torch.distributions as distrib
import pytest


class Model(nn.Module):
    """A simple test model."""

    def __init__(
        self, dim_in: int, dim_out: int, continuous: bool = False, critic: bool = False
    ) -> None:
        """Initialize model."""
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.critic = nn.Linear(dim_in, 1) if critic else None
        self.continuous = continuous

    def forward(self, obs: torch.Tensor) -> distrib.Distribution:
        """Forward pass."""
        h = self.lin(obs)
        if self.continuous:
            probs = distrib.Normal(h, 1.0)
        else:
            probs = distrib.Categorical(logits=h)
        if self.critic is not None:
            return probs, self.critic(obs)
        else:
            return probs

    def new_with_critic(self) -> "Model":
        """Return a similar model with a critic."""
        return Model(
            dim_in=self.lin.in_features,
            dim_out=self.lin.out_features,
            continuous=self.continuous,
            critic=True,
        )


@pytest.fixture
def model(env_factory) -> nn.Module:
    """Policy relevant for the enviornment."""
    env = env_factory()
    dim_in, = env.observation_space.shape
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if continuous:
        dim_out, = env.action_space.shape
    else:
        dim_out = env.action_space.n
    return Model(dim_in, dim_out, continuous)

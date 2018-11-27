# coding: utf-8

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distrib
import pytest

from irl.algo.policy_methods import create_reinforce, create_a2c


class Model(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        continuous: bool = False,
        critic: bool = False
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.critic = nn.Linear(dim_in, 1) if critic else None
        self.continuous = continuous

    def forward(self, obs: torch.Tensor) -> distrib.Distribution:
        h = self.lin(obs)
        if self.continuous:
            probs = distrib.Normal(h, 1.)
        else:
            probs = distrib.Categorical(logits=h)
        if self.critic is not None:
            return probs, self.critic(obs)
        else:
            return probs

    def new_with_critic(self) -> "Model":
        return Model(
            dim_in=self.lin.in_features,
            dim_out=self.lin.out_features,
            continuous=self.continuous,
            critic=True)


@pytest.fixture
def model(env) -> nn.Module:
    """Policy relevant for the enviornment."""
    dim_in, = env.observation_space.shape
    continuous = isinstance(env.action_space, gym.spaces.Box)
    if continuous:
        dim_out, = env.action_space.shape
    else:
        dim_out = env.action_space.n
    return Model(dim_in, dim_out, continuous)


def test_reinforce(device, env, model):
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    agent = create_reinforce(
        env=env,
        model=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32
    )

    agent.run(range(10), 2)


def test_a2c(device, env, model):
    model = model.new_with_critic()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    agent = create_a2c(
        env=env,
        model=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32
    )

    agent.run(range(10), 2)

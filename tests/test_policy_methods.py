# coding: utf-8

import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distrib
import pytest

from irl.environment import TensorEnv
from irl.policy_methods import create_reinforce


class Model(nn.Module):
    def __init__(
        self, dim_in: int, dim_out: int, continuous: bool = False
    ) -> None:
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out)
        self.continuous = continuous

    def forward(self, obs: torch.Tensor) -> distrib.Distribution:
        h = self.lin(obs)
        if self.continuous:
            return distrib.Normal(h, 1.)
        else:
            return distrib.Categorical(logits=h)


@pytest.fixture(params=["CartPole-v1", "MountainCarContinuous-v0"])
def env(request) -> gym.Env:
    """RL environment to test against."""
    return TensorEnv(gym.make(request.param))


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

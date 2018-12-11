# coding: utf-8

"""Pytest fixtures."""

import pytest
import torch
import gym

from irl.exploration.environment import TensorEnv


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request) -> torch.device:
    """Device to run code on."""
    _device = torch.device(request.param)
    if _device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip()
    return _device


@pytest.fixture(params=["CartPole-v1", "MountainCarContinuous-v0"])
def env_factory(request) -> gym.Env:
    """RL environment to test against."""
    def factory():
        return TensorEnv(gym.make(request.param))
    return factory

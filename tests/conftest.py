# coding: utf-8

"""Pytest fixtures."""

import pytest
import torch
import gym

from irl.environment import TensorEnv


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request) -> torch.device:
    """Device to run code on."""
    _device = torch.device(request.param)
    if _device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip()
    return _device


@pytest.fixture(params=["CartPole-v1", "MountainCarContinuous-v0"])
def env(request) -> gym.Env:
    """RL environment to test against."""
    return TensorEnv(gym.make(request.param))

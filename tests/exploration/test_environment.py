# coding: utf-8

import torch

from irl.utils import from_numpy_sparse


def test_gym(env):
    assert hasattr(env, "action_space")
    assert isinstance(env.reset(), torch.Tensor)
    action = from_numpy_sparse(env.action_space.sample())
    obs, _, _, _ = env.step(action)
    assert isinstance(obs, torch.Tensor)

# coding: utf-8

import mock
import numpy as np
import pytest
import torch

from irl.exploration import Transition, Trajectory, create_explorer


class Env:
    def __init__(self):
        self.cnt = 0

    def step(self, action):
        self.cnt += 1
        obs = np.random.rand(10)
        return obs, 1., self.cnt > 10, {}

    def reset(self):
        self.cnt = 0
        return np.random.rand(10)

    def close(self):
        pass


def test_transition(device):
    obs = torch.ones(10, device=device)
    next_obs = 2 * torch.ones(10, device=device)
    action = torch.tensor(3, device=device)
    reward = 9.0
    done = True

    t = Transition(obs, action, next_obs, reward, done)

    assert t.observation is obs
    assert t.next_observation is next_obs
    assert t.action is action
    assert t.reward is reward
    assert t.done is done


def test_trajectory(device):
    trajectory = Trajectory()

    for i in range(10):
        trajectory.append(
            i * torch.ones(7, device=device),
            torch.tensor(i, device=device),
            (i+1) * torch.ones(7, device=device),
            reward=float(i),
            done=(i == 9)
        )

    obs_expected = torch.arange(10, device=device, dtype=torch.float) \
                        .unsqueeze(-1).expand(-1, 7)
    assert (trajectory.observations() == obs_expected).all().item()
    assert (
            trajectory.actions() == torch.arange(10, device=device)
        ).all().item()
    assert (trajectory.next_observations() == 1 + obs_expected).all().item()
    assert len(trajectory.all_observations()) == len(
        trajectory.observations()) + 1
    assert (trajectory.rewards() == torch.arange(10).float()).all().item()
    assert (trajectory.dones()[:-1] == 0).all().item()
    assert (trajectory.dones()[-1] == 1).item()
    return_expected_9 = torch.Tensor(
        [sum((j+i) * .9**j for j in range(10-i)) for i in range(10)])
    assert np.allclose(trajectory.returns(.9, False).cpu(),
                       return_expected_9.cpu())
    assert trajectory.returns(.9, True).mean().abs().item() < 1e-5
    assert (trajectory.returns(.9, True).std() - 1).abs().item() < 1e-1
    assert len(trajectory) == 10
    assert len(trajectory[:]) == 10
    assert isinstance(trajectory[:], Trajectory)
    assert trajectory[0].reward == 0


def test_trajectory_merge(device):
    pytest.skip("NotImplemented")


def test_explorer():
    select_action = mock.MagicMock()
    select_action.return_value = 1
    explorer = create_explorer(Env(), select_action)
    explorer.run(range(100), 2)

    assert select_action.call_count == 22
    assert isinstance(explorer.state.transition, Transition)


def test_explorer_cast(device):
    explorer = create_explorer(
        Env(), lambda x, y: None, dtype=torch.float, device=device)
    explorer.run(range(10))

    assert explorer.state.observation.dtype == torch.float
    assert explorer.state.observation.device == device


def test_explorer_trajectory():
    explorer = create_explorer(Env(), lambda x, y: None, store_trajectory=True)
    explorer.run(range(10))
    assert isinstance(explorer.state.trajectory, Trajectory)
    assert len(explorer.state.trajectory) == 10

# coding: utf-8

import mock
import torch

from ignite.engine import Events

from irl.exploration.explorer import Transition, Explorer


class Env:
    def __init__(self):
        self.cnt = 0

    def step(self, action):
        self.cnt += 1
        obs = torch.rand(10)
        return obs, 1., self.cnt > 10, {}

    def reset(self):
        self.cnt = 0
        return torch.rand(10)

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


def test_explorer_mock():
    select_action = mock.MagicMock()
    select_action.return_value = 1, {}
    explorer = Explorer(Env(), select_action)
    explorer.run(range(100), 2)

    assert select_action.call_count == 22
    assert isinstance(explorer.state.transition, Transition)


def test_explorer(env_factory):
    env = env_factory()

    def select_action(engine, iter):
        return env.action_space.sample()

    explorer = Explorer(env, select_action)
    explorer.run(range(10), 2)


def test_explorer_cast(device):
    explorer = Explorer(
        Env(),
        lambda x, y: (None, {}),
        dtype=torch.int,
        device=device)
    explorer.run(range(10))

    # Observation are casted lazily
    @explorer.on(Events.ITERATION_STARTED)
    def _test(engine):
        assert engine.state.observation.device == device

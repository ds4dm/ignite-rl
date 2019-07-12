# coding: utf-8

import torch
import mock
import pytest

from ignite.engine import Events

import irl.algo.value_methods as valmeth
from irl.algo.trainers import QLearningMode


def test_TensorQValues_single():
    q_values = valmeth.TensorQValues(torch.tensor([1.0, 2.0]))
    assert isinstance(q_values, valmeth.Qvalues)
    assert q_values.vvalue().dim() == 1
    assert q_values.vvalue().item() == 2
    assert q_values.greedy().dim() == 0
    assert q_values.greedy().item() == 1
    assert (q_values.get(q_values.greedy()) == q_values.vvalue()).item()


def test_TensorQValues_batch():
    q_values = valmeth.TensorQValues(torch.tensor([[1.0, 2.0], [3.0, 1.0]]))
    assert isinstance(q_values, valmeth.Qvalues)
    assert q_values.vvalue().dim() == 1
    assert (q_values.vvalue() == torch.tensor([2.0, 3.0])).all()
    assert q_values.greedy().dim() == 1
    assert (q_values.greedy() == torch.tensor([1, 0])).all()
    assert (q_values.get(q_values.greedy()) == q_values.vvalue()).all()


def random_action(env):
    """Return a random action function for a discrete action spadce."""

    def _rand_action(observation):
        return torch.tensor(env.action_space.sample())

    return _rand_action


@pytest.mark.parametrize("qlearningmode", list(QLearningMode))
@mock.patch("irl.algo.trainers.create_qlearning_trainer")
def test_qlearner_trainer_called(
    create_trainer_mock, device, env_factory, dqn, qlearningmode
):
    trainer_mock = mock.MagicMock()
    create_trainer_mock.return_value = trainer_mock
    env = env_factory()
    agent = valmeth.create_memory_qlearner(
        dqn=dqn,
        random_action=random_action(env),
        optimizer=None,
        batch_size=2,
        evaluation_mode=qlearningmode,
        device=device,
        dtype=torch.float32,
    )

    agent.run(env, 2)
    create_trainer_mock.assert_called()
    assert trainer_mock.run.call_count > 1


@pytest.mark.parametrize("qlearningmode", list(QLearningMode))
def test_qlearner_optimizer_called(device, env_factory, dqn, qlearningmode):
    optimizer = mock.MagicMock()
    env = env_factory()
    agent = valmeth.create_memory_qlearner(
        dqn=dqn,
        random_action=random_action(env),
        optimizer=optimizer,
        batch_size=2,
        evaluation_mode=qlearningmode,
        device=device,
        dtype=torch.float32,
    )

    agent.run(env, 2)
    optimizer.step.call_count > 1
    optimizer.zero_grad.call_count > 1


@pytest.mark.parametrize("qlearningmode", list(QLearningMode))
def test_qlearner_trainer_override(device, env_factory, dqn, qlearningmode):
    optimizer = mock.MagicMock()
    env = env_factory()
    agent = valmeth.create_memory_qlearner(
        dqn=dqn,
        random_action=random_action(env),
        optimizer=optimizer,
        batch_size=2,
        evaluation_mode=qlearningmode,
        device=device,
        dtype=torch.float32,
    )

    @agent.on(Events.STARTED)
    def _change_trainer(engine):
        engine.state.trainer = mock.MagicMock()

    agent.run(env, 2)
    assert not optimizer.called
    assert agent.state.trainer.run.call_count > 1

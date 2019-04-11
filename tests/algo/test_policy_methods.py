# coding: utf-8

import torch
import mock

from ignite.engine import Events

import irl.algo.policy_methods


def test_reinforce(device, env_factory, model):
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_reinforce(
        env=env_factory(),
        policy=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32,
    )

    agent.run(100, 2)
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()


def test_a2c(device, env_factory, model):
    model = model.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_a2c(
        env=env_factory(),
        actor_critic=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32,
    )

    agent.run(10, 2)
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()


@mock.patch("irl.algo.trainers.create_ppo_trainer")
def test_ppo_trainer_called(create_trainer_mock, device, env_factory, model):
    trainer_mock = mock.MagicMock()
    create_trainer_mock.return_value = trainer_mock
    model = model.new_with_critic()
    agent = irl.algo.policy_methods.create_ppo(
        env=env_factory(),
        actor_critic=model,
        optimizer=None,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    agent.run(10, 2)
    create_trainer_mock.assert_called()
    assert trainer_mock.run.call_count > 1


def test_ppo_optimizer_called(device, env_factory, model):
    model = model.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_ppo(
        env=env_factory(),
        actor_critic=model,
        optimizer=optimizer,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    agent.run(10, 2)
    optimizer.step.call_count > 1
    optimizer.zero_grad.call_count > 1


def test_ppo_trainer_override(device, env_factory, model):
    model = model.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_ppo(
        env=env_factory(),
        actor_critic=model,
        optimizer=optimizer,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    @agent.on(Events.STARTED)
    def _change_trainer(engine):
        engine.state.trainer = mock.MagicMock()

    agent.run(10, 2)
    assert not optimizer.called
    assert agent.state.trainer.run.call_count > 1

# coding: utf-8

import torch
import mock

from ignite.engine import Events

import irl.algo.policy_methods


def test_reinforce(device, env_factory, prob_policy):
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_reinforce(
        policy=prob_policy, optimizer=optimizer, device=device, dtype=torch.float32
    )

    agent.run(env_factory(), 2)
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()


def test_a2c(device, env_factory, prob_policy):
    prob_policy = prob_policy.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_a2c(
        actor_critic=prob_policy,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32,
    )

    agent.run(env_factory(), 2)
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()


@mock.patch("irl.algo.trainers.create_ppo_trainer")
def test_ppo_trainer_called(create_trainer_mock, device, env_factory, prob_policy):
    trainer_mock = mock.MagicMock()
    create_trainer_mock.return_value = trainer_mock
    prob_policy = prob_policy.new_with_critic()
    agent = irl.algo.policy_methods.create_ppo(
        actor_critic=prob_policy,
        optimizer=None,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    agent.run(env_factory(), 2)
    create_trainer_mock.assert_called()
    assert trainer_mock.run.call_count > 1


def test_ppo_optimizer_called(device, env_factory, prob_policy):
    prob_policy = prob_policy.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_ppo(
        actor_critic=prob_policy,
        optimizer=optimizer,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    agent.run(env_factory(), 2)
    optimizer.step.call_count > 1
    optimizer.zero_grad.call_count > 1


def test_ppo_trainer_override(device, env_factory, prob_policy):
    prob_policy = prob_policy.new_with_critic()
    optimizer = mock.MagicMock()
    agent = irl.algo.policy_methods.create_ppo(
        actor_critic=prob_policy,
        optimizer=optimizer,
        dataset_size=4,
        batch_size=2,
        device=device,
        dtype=torch.float32,
    )

    @agent.on(Events.STARTED)
    def _change_trainer(engine):
        engine.state.trainer = mock.MagicMock()

    agent.run(env_factory(), 2)
    assert not optimizer.called
    assert agent.state.trainer.run.call_count > 1

# coding: utf-8

import torch
import mock

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


def test_ppo(device, env_factory, model):
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
    optimizer.step.assert_called()
    optimizer.zero_grad.assert_called()

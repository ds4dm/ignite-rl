# coding: utf-8

import torch
import torch.optim as optim

from irl.algo.policy_methods import create_reinforce, create_a2c


def test_reinforce(device, env_factory, model):
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    agent = create_reinforce(
        env=env_factory(),
        policy=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32
    )

    agent.run(range(100), 2)


def test_a2c(device, env_factory, model):
    model = model.new_with_critic()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    agent = create_a2c(
        env=env_factory(),
        actor_critic=model,
        optimizer=optimizer,
        device=device,
        dtype=torch.float32
    )

    agent.run(range(10), 2)

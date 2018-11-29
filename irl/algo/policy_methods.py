# coding: utf-8

"""Collection of policy learning algorithms."""

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine, Events

from irl.exploration.environment import Environment
from irl.exploration.explorer import Explorer
from irl.exploration.datasets import Trajectories
import irl.exploration.transforms as transforms


def create_reinforce(
    env: Environment,
    model: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = .99,
    exploration: float = .001,
    grad_norm_clip: Optional[float] = 1.,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Engine:
    """Create an agent using Reinforce learning algorithm.

    Parameters
    ----------
    env:
        The environment the agent interacts with.
    model:
        The neural network used to model the policy.
    optimizer:
        The optimizer used to update the `model` parameters.
    discount:
        The discount rate used for computing the returns.
    exploration:
        The entropy bonus for encouraging exploration.
    grad_norm_clip:
        Value to clip the norm of the gradient at before applying an update.
    dtype:
        Type the obseravtions/model are casted to.
    device:
        Device the model runs on.

    Returns
    -------
    agent:
        The ignite engine, exploring the environement and optimizing.

    """
    if device is not None:
        model.to(device, dtype)

    def select_action(engine: Explorer, observation):
        model.train()
        action_distrib = model(observation)
        action = action_distrib.sample()
        engine.store_transition_members(
            log_prob=action_distrib.log_prob(action),
            entropy=action_distrib.entropy()
        )
        return action

    agent = Explorer(
        env=env,
        select_action=select_action,
        dtype=dtype,
        device=device
    )

    agent.register_transition_members("log_prob", "entropy")

    @agent.on(Events.STARTED)
    def add_trajectories_to_engine(engine):
        engine.state.trajectories = Trajectories(
            transforms.WithReturns(discount=discount, normalize=True))

    @agent.on(Events.EPOCH_STARTED)
    def empty_trajectectories(engine):
        engine.state.trajectories.clear()

    @agent.on(Events.ITERATION_COMPLETED)
    def append_transition(engine):
        engine.state.trajectories.append(engine.state.transition)

    @agent.on(Events.EPOCH_COMPLETED)
    def optimize(engine):
        engine.state.trajectories.terminate_trajectory()
        # The setting is simple enough that using a dataloader is overkill.
        optimizer.zero_grad()
        for transition in engine.state.trajectories:
            loss = -transition.retrn * transition.log_prob
            loss -= exploration * transition.entropy
            loss.backward()

        if grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

    return agent


def create_a2c(
    env: Environment,
    model: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = .99,
    exploration: float = .001,
    critic_loss: Callable = F.mse_loss,
    critic_multiplier: float = 1.,
    grad_norm_clip: Optional[float] = 1.,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Engine:
    """Create an agent using Reinforce learning algorithm.

    Parameters
    ----------
    env:
        The environment the agent interacts with.
    model:
        The neural network used to model the policy and critic. Must return a
        tuple (action probalility distribution, critic value).
    optimizer:
        The optimizer used to update the `model` parameters.
    discount:
        The discount rate used for computing the returns.
    exploration:
        The entropy bonus for encouraging exploration.
    critic_loss:
        The loss function used to learn the critic.
    critic_multiplier:
        Multiplier used for the critic loss in the total loss.
    grad_norm_clip:
        Value to clip the norm of the gradient at before applying an update.
    dtype:
        Type the obseravtions/model are casted to.
    device:
        Device the model runs on.

    Returns
    -------
    agent:
        The ignite engine, exploring the environement and optimizing.

    """
    if device is not None:
        model.to(device, dtype)

    def select_action(engine, observation):
        model.train()
        action_distrib, critic_value = model(observation)
        action = action_distrib.sample()
        engine.store_transition_members(
            log_prob=action_distrib.log_prob(action),
            entropy=action_distrib.entropy(),
            critic_value=critic_value
        )
        return action

    agent = Explorer(
        env=env,
        select_action=select_action,
        dtype=dtype,
        device=device
    )

    agent.register_transition_members("log_prob", "entropy", "critic_value")

    @agent.on(Events.STARTED)
    def add_trajectories_to_engine(engine):
        engine.state.trajectories = Trajectories(
            transforms.WithReturns(discount=discount, normalize=True))

    @agent.on(Events.EPOCH_STARTED)
    def empty_trajectectories(engine):
        engine.state.trajectories.clear()

    @agent.on(Events.ITERATION_COMPLETED)
    def append_transition(engine):
        engine.state.trajectories.append(engine.state.transition)

    @agent.on(Events.EPOCH_COMPLETED)
    def optimize(engine):
        engine.state.trajectories.terminate_trajectory()
        # The setting is simple enough that using a dataloader is overkill.
        optimizer.zero_grad()
        for t in engine.state.trajectories:
            loss = -(t.retrn - t.critic_value.detach()) * t.log_prob
            loss -= exploration * t.entropy
            retrn = t.critic_value.new([t.retrn])  # Make tensor on same device
            loss += critic_multiplier * critic_loss(t.critic_value, retrn)
            loss.backward()

        if grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

    return agent

# coding: utf-8

"""Collection of policy learning algorithms."""

from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from ignite.engine import Engine, Events

from irl.environment import Environment
from irl.exploration.explorer import create_explorer
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

    def select_action(engine, timestep):
        model.train()
        action_distrib = model(engine.state.observation)
        action = action_distrib.sample()
        others = {
            "log_prob": action_distrib.log_prob(action),
            "entropy": action_distrib.entropy()
        }
        return action, others

    agent = create_explorer(
        env=env,
        select_action=select_action,
        dtype=dtype,
        device=device
    )

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
        # The setting is simple enough that using a dataloder is overkill.
        optimizer.zero_grad()
        for transition in engine.state.trajectories:
            loss = -transition.retrn * transition.log_prob
            loss -= exploration * transition.entropy
            loss.backward()

        if grad_norm_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
        optimizer.step()

    return agent

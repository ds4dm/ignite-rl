# coding: utf-8

"""Collection of policy learning algorithms."""

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from ignite.engine import Events

from irl.exploration.environment import Environment
from irl.exploration.explorer import Explorer
from irl.exploration.datasets import Trajectories
import irl.algo.trainers as trainers
import irl.exploration.transforms as T


def create_reinforce(
    env: Environment,
    policy: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = 0.99,
    exploration: float = 0.001,
    normalize_returns: bool = False,
    grad_norm_clip: Optional[float] = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Explorer:
    """Create an agent using Reinforce learning algorithm.

    Parameters
    ----------
    env:
        The environment the agent interacts with.
    policy:
        The neural network used to model the policy.
    optimizer:
        The optimizer used to update the `model` parameters.
    discount:
        The discount rate used for computing the returns.
    exploration:
        The entropy bonus for encouraging exploration.
    normalize_returns:
        Whether to normalize the rewards with zero mean and unit variance.
        Computed over an episode. Raise an error for episode of length 1.
    grad_norm_clip:
        Value to clip the norm of the gradient at before applying an update.
    dtype:
        Type the obseravtions/model are casted to.
    device:
        Device the observations/model are moved to.

    Returns
    -------
    agent:
        The ignite engine, exploring the environement and optimizing.

    """
    policy.to(device=device, dtype=dtype)

    def select_action(engine: Explorer, observation):
        policy.train()
        action_distrib = policy(observation)
        action = action_distrib.sample()
        engine.store_transition_members(
            log_prob=action_distrib.log_prob(action), entropy=action_distrib.entropy()
        )
        return action

    agent = Explorer(env=env, select_action=select_action, dtype=dtype, device=device)

    agent.register_transition_members("log_prob", "entropy")

    @agent.on(Events.STARTED)
    def add_trajectories_to_engine(engine):
        engine.state.trajectories = Trajectories(
            T.WithReturns(discount=discount, normalize=normalize_returns)
        )

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
            nn.utils.clip_grad_norm_(policy.parameters(), grad_norm_clip)
        optimizer.step()

    return agent


def create_a2c(
    env: Environment,
    actor_critic: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = 0.99,
    exploration: float = 0.001,
    normalize_returns: bool = False,
    critic_loss: Callable = F.mse_loss,
    critic_multiplier: float = 1.0,
    grad_norm_clip: Optional[float] = 1.0,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Explorer:
    """Create an agent using Reinforce learning algorithm.

    Parameters
    ----------
    env:
        The environment the agent interacts with.
    actor_critic:
        The neural network used to model the policy and critic. Must return a
        tuple (action probalility distribution, critic value).
    optimizer:
        The optimizer used to update the `model` parameters.
    discount:
        The discount rate used for computing the returns.
    exploration:
        The entropy bonus for encouraging exploration.
    normalize_returns:
        Whether to normalize the rewards with zero mean and unit variance.
        Computed over an episode. Raise an error for episode of length 1.
    critic_loss:
        The loss function used to learn the critic.
    critic_multiplier:
        Multiplier used for the critic loss in the total loss.
    grad_norm_clip:
        Value to clip the norm of the gradient at before applying an update.
    dtype:
        Type the obseravtions/model are casted to.
    device:
        Device the observations/model are moved to.

    Returns
    -------
    agent:
        The ignite engine, exploring the environement and optimizing.

    """
    actor_critic.to(device=device, dtype=dtype)

    def select_action(engine, observation):
        actor_critic.train()
        action_distrib, critic_value = actor_critic(observation)
        action = action_distrib.sample()
        engine.store_transition_members(
            log_prob=action_distrib.log_prob(action),
            entropy=action_distrib.entropy(),
            critic_value=critic_value,
        )
        return action

    agent = Explorer(env=env, select_action=select_action, dtype=dtype, device=device)

    agent.register_transition_members("log_prob", "entropy", "critic_value")

    @agent.on(Events.STARTED)
    def add_trajectories_to_engine(engine):
        engine.state.trajectories = Trajectories(
            T.WithReturns(discount=discount, normalize=normalize_returns)
        )

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
            nn.utils.clip_grad_norm_(actor_critic.parameters(), grad_norm_clip)
        optimizer.step()

    return agent


def create_ppo(
    env: Environment,
    actor_critic: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = 0.99,
    lambda_: float = 0.9,
    ppo_clip: float = 0.02,
    exploration_loss_coef: float = 0.001,
    critic_loss_coef: float = 1.0,
    critic_loss_function: Callable = F.mse_loss,
    # FIX normalization
    normalize_advantages: bool = False,
    dataset_size: int = 1024,
    n_epochs: int = 10,
    # FIXME change the way the dataloader is passed on to the function
    batch_size: int = 16,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Explorer:
    """Create an agent using Proximal Policy Optimization learning algorithm.

    Parameters
    ----------
    env:
        The environment the agent interacts with.
    actor_critic:
        The neural network used to model the policy and critic. Must return a
        tuple (action probalility distribution, critic value).
    optimizer:
        The optimizer used to update the `model` parameters.
    discount:
        The discount rate used for computing the returns.
    lambda_:
        Lambda discount as defined in Generalized Advantage Estimation.
    ppo_clip:
        Clip parameter for the PPO loss.
    exploration_loss_coef:
        The entropy bonus for encouraging exploration.
    critic_loss_coef:
        Mutiplier for the critic loss.
    critic_loss_function:
        Loss function used by the critic.
    normalize_advantages:
        Whether to normalize the advantages with zero mean and unit variance.
        Computed over an episode. Raise an error for episode of length 1.
    dataset_size:
        Size of the PPO dataset to collect information from agents.
    n_epoch:
        Number of epoch of optimization to be on a single PPO dataset.
    batch_size:
        Batch size used to optimized over the PPO dataset.
    dtype:
        Type the obseravtions/model are casted to.
    device:
        Device the observations/model are moved to.

    Returns
    -------
        The ignite engine, exploring the environement and optimizing.

    """
    actor_critic.to(device=device, dtype=dtype)

    def select_action(engine, observation):
        with torch.no_grad():
            actor_critic.eval()
            action_distrib, critic_value = actor_critic(observation)
            action = action_distrib.sample()
            engine.store_transition_members(
                log_prob=action_distrib.log_prob(action),
                entropy=action_distrib.entropy(),
                critic_value=critic_value,
            )
            return action

    agent = Explorer(env=env, select_action=select_action, dtype=dtype, device=device)
    agent.register_transition_members("log_prob", "entropy", "critic_value")
    trainer = trainers.create_ppo_trainer(
        actor_critic=actor_critic,
        optimizer=optimizer,
        ppo_clip=ppo_clip,
        exploration_loss_coef=exploration_loss_coef,
        critic_loss_coef=critic_loss_coef,
        critic_loss_function=critic_loss_function,
        device=device,
        dtype=dtype,
    )

    @agent.on(Events.STARTED)
    def add_trajectories_and_trainer_to_engine(engine):
        engine.state.trajectories = Trajectories(
            T.compose(
                T.WithGAE(
                    discount=discount, lambda_=lambda_, normalize=normalize_advantages
                ),
                T.WithReturns(discount=discount, normalize=False),
                T.PinIfCuda(device=device),
            )
        )
        engine.state.trainer = trainer

    @agent.on(Events.ITERATION_COMPLETED)
    def append_transition(engine):
        engine.state.trajectories.append(engine.state.transition.cpu())

    @agent.on(Events.EPOCH_COMPLETED)
    def terminate_trajectory_and_data_collection(engine):
        engine.state.trajectories.terminate_trajectory()

    @agent.on(Events.EPOCH_COMPLETED)
    def optimize(engine):
        if len(engine.state.trajectories) >= dataset_size:
            sample_elem = engine.state.trajectories[0]
            dataloader = DataLoader(
                dataset=engine.state.trajectories,
                batch_size=batch_size,
                collate_fn=sample_elem.__class__.collate,
                drop_last=True,
            )
            engine.state.trainer.run(dataloader, n_epochs)
            engine.state.trajectories.clear()

    return agent

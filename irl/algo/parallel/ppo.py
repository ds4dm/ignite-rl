# coding: utf-8

"""Policy methods as ignite engine."""

from typing import Optional, Callable
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Events, Engine

import irl.utils as utils
import irl.functional as Firl
from irl.exploration.environment import Environment
from irl.algo.parallel.explorer import ParallelExplorer
from irl.exploration.datasets import Trajectories
import irl.exploration.transforms as transforms


def create_ppo_optimizer(
    actor_critic: nn.Module,
    optimizer: optim.Optimizer,
    ppo_clip: float = .02,
    exploration_loss_coef: float = .001,
    critic_loss_coef: float = 1.,
    critic_loss_function: Callable = F.mse_loss,
    device: Optional[torch.device] = None
) -> Engine:

    def optimize(engine, batch):
        batch
        actor_critic.train()

        action_distribs, critic_values = actor_critic(batch.observation)
        log_probs = action_distribs.log_prob(batch.action)

        loss = Firl.ppo_loss(
            targets=batch.gae.detach(),
            log_probs=log_probs,
            old_log_probs=batch.log_probs.detach(),
            ppo_clip=ppo_clip
        )
        entropy_loss = action_distribs.entropy().mean()
        loss += (exploration_loss_coef * entropy_loss)
        critic_loss = critic_loss_function(critic_values, batch.retrn)
        loss += (critic_loss_coef * critic_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()


def create_ppo(
    env_factory: Callable[[], Environment],
    actor_critic: nn.Module,
    optimizer: optim.Optimizer,
    discount: float = .99,
    lambda_: float = .9,
    ppo_clip: float = .02,
    exploration_loss_coef: float = .001,
    critic_loss_coef: float = 1.,
    normalize_advantages: bool = True,
    n_agents: int = 2,
    max_episode_length: Optional[int] = None,
    n_controllers: int = 2,
    n_epochs: int = 10,
    dataset_size: int = 1024,
    batch_size: int = 32,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None
):
    actor_critic.to(device=device, dtype=dtype)

    def select_action(engine, observation):
        with torch.no_grad():
            actor_critic.eval()
            action_distrib, critic_value = actor_critic(observation)
            action = action_distrib.sample()
            engine.store_transition_members(
                log_prob=action_distrib.log_prob(action),
                entropy=action_distrib.entropy(),
                critic_value=critic_value
            )
            return action

    # Shared objects
    counter = utils.Counter(dataset_size)
    model_lock = utils.RWLock()
    dataset = Trajectories(transforms.compose(
        transforms.WithGAE(discount=discount, lambda_=lambda_,
                           normalize=normalize_advantages),
        transforms.WithReturns(discount=discount, normalize=False),
        lambda trajectory: [t.pin_memory() for t in trajectory]
    ))

    # Creating parallel agents
    agents = []
    for _ in range(n_agents):
        agent = ParallelExplorer(
            env=env_factory(),
            select_action=select_action,
            model_lock=model_lock,
            dtype=dtype,
            device=device
        )

        agent.register_transition_members(
            "log_prob", "entropy", "critic_value")

        @agent.on(Events.STARTED)
        def add_trajectories_to_engine(engine):
            engine.state.trajectories = dataset.new_shared_trajectories()

        @agent.on(Events.ITERATION_COMPLETED)
        def append_transition(engine):
            engine.state.trajectories.append(engine.state.transition)
            nonlocal counter
            counter += 1

        agents.append(agents)

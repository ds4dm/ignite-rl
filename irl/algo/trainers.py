# coding: utf-8

"""Trainers are ignite engine that optimize over a dataset.

The dataset might be updated from more recent experience. The `TRAINER_`
event encapsualte a all iteration done on a same dataset.
"""

from typing import Optional, Callable
import enum

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine

import irl.utils as utils
import irl.functional as Firl


Trainer = Engine


def create_ppo_trainer(
    actor_critic: nn.Module,
    optimizer: optim.Optimizer,
    ppo_clip: float = 0.02,
    exploration_loss_coef: float = 0.001,
    critic_loss_coef: float = 1.0,
    critic_loss_function: Callable = F.mse_loss,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Trainer:
    """Create a trainer to optimize a model with PPO-clip loss.

    Parameters
    ----------
    actor_critic:
        The neural network used to model the policy and critic. Must return a
        tuple (action probalility distribution, critic value).
    optimizer:
        The optimizer used to update the `model` parameters.
    ppo_clip:
        Clip parameter for the PPO loss.
    exploration_loss_coef:
        Multiplier of the entropy bonus for the encouraging exploration.
    dtype:
        Type the observations/model are converted to.
    device:
        Device the observations/model are moved to.

    Returns
    -------
    trainer:
        An ignite engine that optimize an actor-critic over a dataset.

    """
    actor_critic.to(device=device, dtype=dtype)

    def optimize(engine, batch):
        batch = batch.to(device=device, dtype=dtype)
        actor_critic.train()

        action_distribs, critic_values = actor_critic(batch.observation)
        log_probs = action_distribs.log_prob(batch.action)

        loss = Firl.ppo_loss(
            targets=batch.gae.detach(),
            log_probs=log_probs,
            old_log_probs=batch.log_prob.detach(),
            ppo_clip=ppo_clip,
        )
        entropy_loss = action_distribs.entropy().mean()
        loss -= exploration_loss_coef * entropy_loss
        critic_loss = critic_loss_function(critic_values.squeeze(1), batch.retrn)
        loss += critic_loss_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(optimize)
    return trainer


class QLearningMode(utils.NameEnum):
    """Q-learning modes.

    Simple is normal q learning, target uses a target network for making and
    evaluating target V values, double use a target network to evaluate the
    action of the DQN.
    """

    SIMPLE = enum.auto()
    TARGET = enum.auto()
    DOUBLE = enum.auto()


def create_qlearning_trainer(
    dqn: nn.Module,  # Callable[[Observation], QValues]
    target_dqn: Optional[nn.Module],  # Callable[[Observation], QValues]
    optimizer: optim.Optimizer,
    discount: float = 0.99,
    evaluation_mode: QLearningMode = QLearningMode.DOUBLE,
    loss_func: Callable = F.mse_loss,
    clip_grad_norm: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Trainer:
    """Optimize a dataset a DQN on a dataset of transition.

    Optimization is done using TD(0) deep q learning, with memory replay.
    Double and tartget evaluation are also available.

    Parameters
    ----------
    dqn:
        The neural network estimating qvalues tht is being optimized.
    target_dqn:
        Optional second network for target or double q-learning evaluation.
    optimizer:
        The optimizer used to update the `dqn` parameters.
    discount:
        Dicount factor of the rfuture rewards.
    evaluation_mode:
        Change the way targets are evaluated, either with a target network, or
        using double q-learning.
    loss_func:
        The loss function used between the -values of the action taken and the
        new target q-values.
    clip_grad_norm:
        Optionally clip the norm of the `dqn` gradient before applying them.
    dtype:
        Type the observations/model are converted to.
    device:
        Device the observations/model are moved to.

    Returns
    -------
    trainer:
        An ignite engine that optimize an deep Q network over a dataset.

    """
    dqn.to(dtype=dtype, device=device)
    if target_dqn is not None:
        target_dqn.to(dtype=dtype, device=device)

    def optimize(engine, batch):
        batch = batch.to(device=device, dtype=dtype)

        with torch.no_grad():
            if evaluation_mode == QLearningMode.SIMPLE:
                # Actions selected and evaluated by model network.
                dqn.eval()
                next_vvals = dqn(batch.next_observation).vvalue()
            elif evaluation_mode == QLearningMode.TARGET:
                # Actions selected and evaluated by target network.
                target_dqn.eval()
                next_vvals = target_dqn(batch.next_observation).vvalue()
            elif evaluation_mode == QLearningMode.DOUBLE:
                # Actions selected by model evaluated by target network.
                target_dqn.eval()
                actions = dqn(batch.next_observation).greedy()
                next_vvals = target_dqn(batch.next_observation).get(actions)
            next_vvals[batch.done.byte()] = 0
            targets = batch.reward.float() + discount * next_vvals

        dqn.train()
        qvals = dqn(batch.observation).get(batch.action)
        loss = loss_func(qvals, targets)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad_norm is not None:
            nn.utils.clip_grad_norm_(dqn.parameters(), clip_grad_norm)
        optimizer.step()

    return Engine(optimize)

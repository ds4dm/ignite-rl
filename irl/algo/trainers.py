# coding: utf-8

"""Trainers are ignite engine that optimize over a dataset.

The dataset might be updated from more recent experience. The `TRAINER_`
event encapsualte a all iteration done on a same dataset.
"""

from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ignite.engine import Engine

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
        loss += exploration_loss_coef * entropy_loss
        critic_loss = critic_loss_function(critic_values.squeeze(1), batch.retrn)
        loss += critic_loss_coef * critic_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    trainer = Engine(optimize)
    return trainer

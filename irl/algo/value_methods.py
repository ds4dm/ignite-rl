# coding: utf-8

"""Collection of value learning algorithms."""

import abc
import copy
from typing import Optional, Callable, Union, Generic

import attr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ignite.engine import Events

import irl.utils as utils
from irl.exploration.explorer import Explorer
from irl.exploration.datasets import MemoryReplay
import irl.algo.trainers as trainers
import irl.exploration.transforms as T
import irl.numbers as num
from irl.exploration.environment import Observation, Action


Qval = torch.Tensor


class Qvalues(abc.ABC, Generic[Action]):
    """Represent the Q-values of all action in a given state."""

    def vvalue(self) -> Qval:
        """Return the V values, i.e. the maximum Q-value over actions.

        Reimplement for a more optimized version.
        """
        return self.get(self.greedy())

    @abc.abstractmethod
    def greedy(self) -> Action:
        """Return the greedy action for these q-values.

        The greedy action is the one associated with with larget Q-value.
        """
        return NotImplemented

    @abc.abstractmethod
    def get(self, action: Action) -> Qval:
        """Get the Q-value of a given action."""
        return NotImplemented


@attr.s(auto_attribs=True, slots=True, frozen=True)
class TensorQValues(Qvalues[torch.Tensor]):
    """A QValues implementation for fixed sized discrete action spaces."""

    q_values: torch.Tensor

    @staticmethod
    def with_batch_dim(x: torch.Tensor) -> torch.Tensor:
        """Add the batch dimaension as the first one if it does not exists."""
        if x.dim() == 1:
            return x.unsqueeze(0)
        else:
            return x

    def vvalue(self) -> Qval:
        """Return the V values, i.e. the maximum Q-value over actions.

        Reimplement for a more optimized version.
        """
        return self.with_batch_dim(self.q_values).max(1).values

    def greedy(self) -> torch.Tensor:
        """Return the greedy action for these q-values.

        The greedy action is the one associated with with larget Q-value.
        """
        return self.with_batch_dim(self.q_values).argmax(1).squeeze()

    def get(self, action: Action) -> Qval:
        """Get the Q-value of a given action."""
        b_q_values = self.with_batch_dim(self.q_values)
        return b_q_values.gather(1, action.long().view(-1, 1)).squeeze(1)


def create_memory_qlearner(
    dqn: nn.Module,  # Callable[[Observation], QValues]
    random_action: Callable[[Observation], Action],
    optimizer: optim.Optimizer,
    discount: float = 0.99,
    epsilon: Union[float, num.Stepable] = 0.05,
    evaluation_mode: trainers.QLearningMode = trainers.QLearningMode.DOUBLE,
    optimizing_steps: int = 4,
    double_target_weight_copy_steps: int = 1000,
    memory_capacity: int = 10000,
    batch_size: int = 32,
    clip_grad_norm: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Explorer:
    """Create a Q-learner.

    Optimization is done using TD(0) deep q learning, with memory replay.
    Double and tartget evaluation are also available.

    Parameters
    ----------
    dqn:
        The neural network estimating qvalues tht is being optimized.
    random_action:
        A function to make random actions.
    optimizer:
        The optimizer used to update the `dqn` parameters.
    discount:
        Dicount factor of the rfuture rewards.
    epsilon:
        Probability of amking a random action. The value is `step` if possible.
    evaluation_mode:
        Change the way targets are evaluated, either with a target network, or
        using double q-learning.
    optimizing_steps:
        Number of steps between optimization over the replay memory is
        performed.
    double_target_weight_copy_steps:
        Number of steps between the target/double newtorks weights are updated
        (when applicable).
    memory_capacity:
        Size of the replay memory (dataset).
    batch_size:
        Batch size when optimizing over the replay memeory.
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
    # Enable converting from string
    evaluation_mode = trainers.QLearningMode(evaluation_mode)
    dqn.to(device=device, dtype=dtype)
    if evaluation_mode == trainers.QLearningMode.SIMPLE:
        target_dqn = None
    else:
        target_dqn = copy.deepcopy(dqn)

    def select_action(engine, observation):
        """Epsilon greedy action selection."""
        with torch.no_grad():
            dqn.eval()
            if torch.rand(1).item() < epsilon:
                return random_action(observation)
            else:
                return dqn(observation).greedy()

    agent = Explorer(select_action=select_action, dtype=dtype, device=device)
    trainer = trainers.create_qlearning_trainer(
        dqn=dqn,
        target_dqn=target_dqn,
        optimizer=optimizer,
        discount=discount,
        evaluation_mode=evaluation_mode,
        clip_grad_norm=clip_grad_norm,
        dtype=dtype,
        device=device,
    )

    @agent.on(Events.STARTED)
    def add_memory_and_trainer_to_agent(engine):
        engine.state.memory = MemoryReplay(
            T.PinIfCuda(device=device), capacity=memory_capacity
        )
        engine.state.trainer = trainer

    @agent.on(Events.ITERATION_COMPLETED)
    def append_transition_and_step_epsilon(engine):
        engine.state.memory.append(engine.state.transition.cpu())
        if isinstance(epsilon, num.Stepable):
            epsilon.step()

    @agent.on(Events.ITERATION_COMPLETED)
    @utils.every(optimizing_steps)
    def optimize(engine):
        sample_elem = engine.state.memory[0]
        dataloader = DataLoader(
            dataset=engine.state.memory,
            batch_size=batch_size,
            collate_fn=sample_elem.__class__.collate,
            drop_last=True,
        )
        engine.state.trainer.run(dataloader)

    @agent.on(Events.ITERATION_COMPLETED)
    @utils.every(double_target_weight_copy_steps)
    def copy_weights(engine):
        if target_dqn is not None:
            dqn.zero_grad()  # Avoid copying the gradients
            target_dqn.load_state_dict(copy.deepcopy(dqn.state_dict()))

    return agent

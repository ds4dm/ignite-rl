# coding: utf-8

"""The Environement base class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from collections.abc import Sequence, Mapping
from numbers import Number
from typing import TypeVar, Generic, Tuple, Dict, Any

import torch


Observation = TypeVar(
    "Observation", torch.Tensor, Sequence, Mapping)
BatchedObservations = TypeVar(
    "BatchedObservations", torch.Tensor, Sequence, Mapping)
Action = TypeVar(
    "Action", Number, torch.Tensor, Sequence, Mapping)
BatchedActions = TypeVar(
    "BatchedActions", Number, torch.Tensor, Sequence, Mapping)


class Environment(Generic[Action, Observation], ABC):
    """Reinforcement learning environement.

    A Markov decision process where an agent can interact by taking actions.
    An implementation of this class (or at least with similar API) is required
    to use the algorithms.
    """

    @abstractmethod
    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Take an action in the environement.

        Parameters
        ----------
        action:
            Acxtion to taken by the agent.

        Returns
        -------
        observation:
            The new obseraction of the environement.
        reward:
            Reward recieved for having taken the action.
        done:
            Whether the episode is finished.
        infos:
            Other information about the environement.

        """
        return NotImplemented

    @abstractmethod
    def reset(self) -> Observation:
        """Reset the environement.

        Reset the environement ot a clean start, e.g. the begining of the
        episode in episodic casses.

        Returns
        -------
        observation:
            The first observation of the episode.

        """
        return NotImplemented

    @abstractmethod
    def close(self) -> None:
        """Close environement."""
        return NotImplemented

    @abstractmethod
    def seed(self, seed=None) -> None:
        """Set random seed for the environement."""
        return NotImplemented

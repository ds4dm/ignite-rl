# coding: utf-8

"""The Environement base class for reinforcement learning algorithms."""

from abc import ABC, abstractmethod
from numbers import Number
from typing import TypeVar, Generic, Tuple, Dict, Any

import torch
import attr

from irl.utils import from_numpy_sparse, apply_to_tensor
from .data import Data

Observation = TypeVar("Observation", torch.Tensor, Data)
Action = TypeVar("Action", Number, torch.Tensor, Data)


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


@attr.s(auto_attribs=True, frozen=True)
class TensorEnv(Environment, Generic[Action, Observation]):
    """Wraps a numpy environement so it returns Tensors objects."""

    base_env: Environment

    def __getattr__(self, name):
        """All other attributes are gotten from base_env."""
        return getattr(self.base_env, name)

    def step(
        self, action: Action
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        """Take an action in the environement."""
        action = apply_to_tensor(action, lambda t: t.cpu().numpy())
        obs, reward, done, info = self.base_env.step(action)
        return from_numpy_sparse(obs), reward, done, info

    def reset(self) -> Observation:
        """Reset the environement."""
        return from_numpy_sparse(self.base_env.reset())

    def close(self) -> None:
        """Close environement."""
        self.base_env.close()

    def seed(self, seed=None) -> None:
        """Set random seed for the environement."""
        self.base_env.seed(seed)

# coding: utf-8

"""Dataset transforms."""

from functools import reduce, partial
from typing import Callable, List, Any

import attr
import torch

import irl.functional as Firl


def compose(*transforms: Callable) -> Callable:
    """Retrun the composed of all the given functions."""
    return partial(reduce, lambda x, f: f(x), transforms)


@attr.s(auto_attribs=True)
class WithReturns:
    """Transform class for transitions to add a the return to every item."""

    discount: float = 0.9
    normalize: bool = True

    _Transition: type = attr.ib(init=False)

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Add the return to every item in the trajectory."""
        rewards = torch.tensor([t.reward for t in trajectory], dtype=torch.float32)
        returns = Firl.returns(rewards, self.discount)
        if self.normalize:
            returns = Firl.normalize_1d(returns)

        if not hasattr(self, "_Transition"):
            self._Transition = attr.make_class(
                trajectory[0].__class__.__name__,
                ["retrn"],
                bases=(trajectory[0].__class__,),
                frozen=True,
                slots=True,
            )

        return [
            self._Transition(**attr.asdict(t, recurse=False), retrn=r)
            for t, r in zip(trajectory, returns.tolist())
        ]


@attr.s(auto_attribs=True)
class WithGAE:
    """Generalized Advantage Estimation.

    Transform class for transitions to add a the generalized advantage
    estimation to every item.
    """

    discount: float = 0.99
    lambda_: float = 0.9
    normalize: bool = True
    dtype: torch.dtype = torch.float32

    _Transition: type = attr.ib(init=False)

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Add the return to every item in the trajectory."""
        rewards = [t.reward for t in trajectory]
        values = [t.critic_value for t in trajectory]
        if trajectory[-1].done:
            # Value of terminal state is zero
            values.append(0.0)
        else:
            # Missing the critic value for the next observation
            # We shorten the trajectory by one
            del rewards[-1]

        with torch.no_grad():
            rewards = torch.tensor(rewards, dtype=self.dtype)
            values = torch.tensor(values, dtype=self.dtype)

            gae = Firl.generalize_advatange_estimation(
                rewards=rewards,
                values=values,
                discount=self.discount,
                lambda_=self.lambda_,
                normalize=self.normalize,
            )

        if not hasattr(self, "_Transition"):
            self._Transition = attr.make_class(
                trajectory[0].__class__.__name__,
                ["gae"],
                bases=(trajectory[0].__class__,),
                frozen=True,
                slots=True,
            )

        # trunctation done by zip if reward is smaller than transitions
        return [
            self._Transition(**attr.asdict(t, recurse=False), gae=g)
            for t, g in zip(trajectory, gae.tolist())
        ]


@attr.s(auto_attribs=True)
class PinIfCuda:
    """Pin the data if the device is of type Cuda."""

    device: torch.device = attr.ib(converter=torch.device)

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Return the list of pined transitions, if the device is Cuda."""
        if self.device.type == "cuda":
            return [t.pin_memory() for t in trajectory]
        else:
            return trajectory

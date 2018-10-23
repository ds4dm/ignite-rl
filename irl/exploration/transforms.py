# coding: utf-8

from functools import reduce, partial
from typing import Callable, List, Any

import attr
import torch

import irl.functional as Firl


def compose(*transforms: Callable) -> Callable:
    """Retrun the composed of all the given functions."""
    return partial(reduce, lambda x, f: f(x), transforms)


class WithReturns:
    """Transform class to add a the return to every item."""

    def __init__(self, discount: float, normalize: bool = True) -> None:
        """Initialise object."""
        self.discount = discount
        self.normalize = normalize
        self.__Transition = None

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Add the return to every item in the trajectory."""
        rewards = torch.tensor(
            [t.reward for t in trajectory], dtype=torch.float32)
        returns = Firl.returns(rewards, self.discount)
        if self.normalize:
            returns = Firl.normalize_1d(returns)

        if self.__Transition is None:
            self.__Transition = attr.make_class(
                "Transition",
                ["retrn"],
                bases=(trajectory[0].__class__, ),
                frozen=True
            )

        return [self.__Transition(**attr.asdict(x), retrn=r)
                for x, r in zip(trajectory, returns)]

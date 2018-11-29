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
    """Transform class to add a the return to every item."""

    discount: float = .9
    normalize: bool = True

    _Transition: type = attr.ib(init=False)

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Add the return to every item in the trajectory."""
        rewards = torch.tensor(
            [t.reward for t in trajectory], dtype=torch.float32)
        returns = Firl.returns(rewards, self.discount)
        if self.normalize:
            returns = Firl.normalize_1d(returns)

        if not hasattr(self, "_Transition"):
            self._Transition = attr.make_class(
                trajectory[0].__class__.__name__,
                ["retrn"],
                bases=(trajectory[0].__class__, ),
                frozen=True,
                slots=True
            )

        return [self._Transition(**attr.asdict(x), retrn=r)
                for x, r in zip(trajectory, returns.tolist())]

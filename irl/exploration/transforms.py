# coding: utf-8

"""Dataset transforms."""

import logging
from functools import reduce, partial
from typing import Callable, List, Any, Dict, Hashable

import attr
import torch
import torch.nn.functional as F

import irl.functional as Firl


logger = logging.getLogger(__name__)


@attr.s(auto_attribs=True)
class _MultiTaskNormalizer:
    """Normalize and keep track of running averages.

    Running averages are kept by task, being given an id.
    Do not use for contextual bandit as the return is set to zero in some
    edge case to avoid unscaled gradients.
    """

    running_means: Dict[Hashable, torch.Tensor] = attr.Factory(dict)
    running_vars: Dict[Hashable, torch.Tensor] = attr.Factory(dict)
    epsilon: float = 1e-5

    def normalize(self, x: torch.Tensor, task_id: Hashable = None) -> torch.Tensor:
        """Normalize input and store running avgs."""
        can_norm = len(x) > 1
        # Warm starting the scaling factors
        if task_id not in self.running_means:
            if can_norm:
                self.running_means[task_id] = x.mean().unsqueeze(0)
                self.running_vars[task_id] = x.var(unbiased=False).unsqueeze(0)
            else:
                # Single reward on unknown task: avoid unknown gradients
                logger.warning(
                    "Trajectory with length 1 for unknown task. Normalization is 0."
                )
                return x.new_zeros(1)

        # Running avgs updated if we can_norm, otherwise used to normalize.
        return F.batch_norm(
            x.unsqueeze(1),
            running_mean=self.running_means[task_id],
            running_var=self.running_vars[task_id],
            training=can_norm,
            eps=self.epsilon,
        ).squeeze(1)

    def denormalize(self, y: torch.Tensor, task_id: Hashable = None) -> torch.Tensor:
        """Reverse the opertion of normalization using the running avgs.

        If no running avgs are found, the input is return as is.
        """
        if task_id in self.running_means:
            mean = self.running_means[task_id]
            var = self.running_vars[task_id]
            return y * torch.sqrt(var + self.epsilon) + mean
        else:
            logger.warning("Denormalizing unknown task (no-opt).")
            return y


def compose(*transforms: Callable) -> Callable:
    """Retrun the composed of all the given functions."""
    return partial(reduce, lambda x, f: f(x), transforms)


class WithReturns:
    """Transform class for transitions to add a the return to every item.

    When nomralizing, the running mean and variance are computed (but not used
    for the normalization itself).
    """

    def __init__(self, discount: float = 0.9, normalize: bool = True) -> None:
        """Initialize the normalizer if necessary."""
        self.discount = discount
        self._Transition = None
        if normalize:
            self.normalizer = _MultiTaskNormalizer()
        else:
            self.normalizer = None

    def __call__(self, trajectory: List[Any]) -> List[Any]:
        """Add the return to every item in the trajectory."""
        rewards = torch.tensor([t.reward for t in trajectory], dtype=torch.float32)
        returns = Firl.returns(rewards, self.discount)
        if self.normalizer is not None:
            returns = self.normalizer.normalize(returns)

        if self._Transition is None:
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

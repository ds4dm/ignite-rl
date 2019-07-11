# coding: utf-8

"""Dataset transforms."""

import logging
from functools import reduce, partial
from typing import Callable, List, Dict, Hashable

import attr
import torch
import torch.nn.functional as F

import irl.functional as Firl
from irl.exploration.explorer import Transition


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
        convert = y if isinstance(y, torch.Tensor) else None
        if task_id in self.running_means:
            mean = self.running_means[task_id].to(convert)
            var = self.running_vars[task_id].to(convert)
            return y * torch.sqrt(var + self.epsilon) + mean
        else:
            logger.warning("Denormalizing unknown task (no-opt).")
            return y


def compose(*transforms: Callable) -> Callable:
    """Retrun the composed of all the given functions."""
    return partial(reduce, lambda x, f: f(x), transforms)


class WithReturns:
    """Transform class for transitions to add a the return to every item.

    If a trajectory is not terminated, and a critic is available, it is used
    to bootstrap the returns. When normalizing, the running mean and variance
    are computed (but not used for the normalization itself) and kept per task
    (use `task_id` in the transition to keep different values). They are used
    to scale back the critic when bootstrapping.
    """

    def __init__(
        self,
        discount: float = 0.9,
        norm_returns: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the transform.

        Parameters
        ----------
        discount:
            The discount rate applied to compute the returns from the rewards.
        norm_returns:
            Whether to normalize the returns. Running averages are kept per
            task (use `task_id` in the transition to keep different values)
            and used to scale the critic back when bootstrapping.
        dtype:
            The type in which the returns are computed.

        """
        self.discount = discount
        self.dtype = dtype
        self._Transition = None
        if norm_returns:
            self.normalizer = _MultiTaskNormalizer()
        else:
            self.normalizer = None

    def returns(
        self, trajectory: List[Transition], rewards: torch.Tensor
    ) -> torch.Tensor:
        """Compute the returns, possibly bootstrp and scale from critic."""
        full_trajectory = trajectory[-1].done
        task_id = getattr(trajectory[0], "task_id", None)

        # Bootstrp from critic if possible
        if not full_trajectory:
            if hasattr(trajectory[-1], "critic_value"):
                last_critic_val = trajectory[-1].critic_value
                if self.normalizer is not None:
                    last_critic_val = self.normalizer.denormalize(
                        last_critic_val, task_id=task_id
                    )
                rewards[-1] = last_critic_val
            else:
                logger.warning(
                    "Computing returns for uncomplete trajectory without a critic."
                )
        returns = Firl.discounted_sum(rewards, self.discount)
        # Scale the returns if necessary and store running averages
        if self.normalizer is not None:
            returns = self.normalizer.normalize(returns, task_id=task_id)

        return returns

    def __call__(self, trajectory: List[Transition]) -> List[Transition]:
        """Add the return to every item in the trajectory."""
        rewards = torch.tensor([t.reward for t in trajectory], dtype=torch.float32)
        with torch.no_grad():
            returns = self.returns(trajectory, rewards=rewards)

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


class WithGAEs(WithReturns):
    """Generalized Advantage Estimation.

    Transform class for transitions to add a the generalized advantage
    estimation to every item.
    The class also compute the returns wioth bootstrapping. If scaling of
    the returns is used, the running average (per task) are used to scale
    back the critic estimate to compute the GAEs.
    """

    def __init__(
        self,
        discount: float = 0.99,
        lambda_: float = 0.9,
        norm_returns: bool = True,
        norm_gaes: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialize the transform.

        Parameters
        ----------
        discount:
            The discount rate applied to compute the returns from the rewards.
        lambda_:
            Second discount factor for generalized advantages estimation.
        norm_returns:
            Whether to normalize the returns. Running averages are kept per
            task (use `task_id` in the transition to keep different values)
            and used to scale the critic back when bootstrapping.
        norm_gaes:
            Whether to normalize the GAEs computed.
        dtype:
            The type in which the returns are computed.

        """
        super().__init__(discount=discount, norm_returns=norm_returns, dtype=dtype)
        self.lambda_ = lambda_
        self.norm_gaes = norm_gaes

    def gaes(self, trajectory: List[Transition], rewards: torch.Tensor) -> torch.Tensor:
        """Return the generalized adgantages estimation."""
        full_trajectory = trajectory[-1].done
        task_id = getattr(trajectory[0], "task_id", None)

        values = [t.critic_value for t in trajectory]
        if full_trajectory:
            values.append(0.0)  # Value of terminal state is zero
        else:
            # Missing the critic value for the next observation
            # We shorten the trajectory by one
            rewards = rewards[:-1]
        values = torch.tensor(values, dtype=self.dtype)
        if self.normalizer is not None:
            values = self.normalizer.denormalize(values, task_id=task_id)

        return Firl.generalize_advatange_estimation(
            rewards=rewards,
            values=values,
            discount=self.discount,
            lambda_=self.lambda_,
            normalize=self.norm_gaes,
        )

    def __call__(self, trajectory: List[Transition]) -> List[Transition]:
        """Add the return and gae to every item in the trajectory."""
        rewards = torch.tensor([t.reward for t in trajectory], dtype=torch.float32)
        with torch.no_grad():
            returns = self.returns(trajectory, rewards=rewards)
            gaes = self.gaes(trajectory, rewards=rewards)

        if self._Transition is None:
            self._Transition = attr.make_class(
                trajectory[0].__class__.__name__,
                ["retrn", "gae"],
                bases=(trajectory[0].__class__,),
                frozen=True,
                slots=True,
            )

        # trunctation done by zip if reward is smaller than transitions
        return [
            self._Transition(**attr.asdict(t, recurse=False), retrn=r, gae=g)
            for t, r, g in zip(trajectory, returns.tolist(), gaes.tolist())
        ]


@attr.s(auto_attribs=True)
class PinIfCuda:
    """Pin the data if the device is of type Cuda."""

    device: torch.device = attr.ib(converter=torch.device)

    def __call__(self, transition: Transition) -> Transition:
        """Return the list of pined transitions, if the device is Cuda."""
        if self.device.type == "cuda":
            return transition.pin_memory()
        else:
            return transition

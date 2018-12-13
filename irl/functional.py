# coding: utf-8

"""A collection of auxilliary functions for reinforcement learning."""

from functools import singledispatch

import numpy as np
import torch
import torch.nn.functional as F


def value_td_residuals(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount: float
) -> torch.Tensor:
    """Compute TD residual of state value function.

    All tensors must be one dimensional. This is valid only for one trajectory.

    Parameters
    ----------
    rewards:
        The one step reward.
    values:
        The estimated values at the current step. Note that the last test is
        terminal, the associated value should be zero.
    next_values:
        The estimated values at the next step.
    discount:
        The discount rate.

    """
    return rewards + (discount * next_values) - values


@singledispatch
def discounted_sum(
    X: list, discount: float, last: float = 0.
) -> torch.Tensor:
    """Compute a discounted sum for every element.

    Given a one dimension input `x` of size `n`, the output tensor has size `n`
    where every output is given by:
        out[j] = sum_{0 <= i < n-j} x[i+j] * discount**j
    """
    outputs = []
    d = last
    for x in X[::-1]:
        d = discount * d + x
        outputs.append(d)
    return outputs[::-1]


@discounted_sum.register
def _(
    X: np.ndarray, discount: float, last: float = 0.
) -> torch.Tensor:
    output = discounted_sum(X.tolist(), discount=discount, last=last)
    return np.array(output, dtype=np.float32)


@discounted_sum.register
def _(
    X: torch.Tensor, discount: float, last: float = 0.
) -> torch.Tensor:
    output = discounted_sum(X.cpu().numpy(), discount=discount, last=last)
    return torch.from_numpy(output).to(dtype=torch.float, device=X.device)


def returns(
    rewards: torch.Tensor, discount: float, last: float = 0.
) -> torch.Tensor:
    """Compute the disounted returns given the rewards.

    This is equivalent to `discoutned_sum`. This is valid only for one
    trajectory.
    """
    return discounted_sum(rewards, discount=discount, last=last)


def normalize_1d(x: torch.Tensor) -> torch.Tensor:
    """Normalize a 1 dimnesional tensor."""
    return F.batch_norm(x.unsqueeze(1), None, None, training=True).squeeze(1)


def generalize_advatange_estimation(
    rewards: torch.Tensor,
    values: torch.Tensor,
    discount: float = .99,
    lambda_: float = .9,
    normalize: bool = True
) -> torch.Tensor:
    """Compute generalize advatange estimation.

    The discounted sum of TD residuals as described in the generalized advatage
    estimation paper: http://arxiv.org/abs/1506.02438
    This is valid only for one trajectory.
    """
    v_td_residuals = value_td_residuals(
        rewards=rewards,
        values=values[:-1],
        next_values=values[1:],
        discount=discount
    )
    if normalize:
        v_td_residuals = normalize_1d(v_td_residuals)
    return discounted_sum(v_td_residuals, discount*lambda_)


def ppo_loss(
    targets: torch.Tensor,
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    ppo_clip: float = .02
) -> torch.Tensor:
    """Compute the PPO loss.

    As defined in the proximal policy optimization paper:
    http://arxiv.org/abs/1707.06347

    Parameters
    ----------
    targets:
        Often returns or advantages. What the probabilities ratio is multiplied
        by. One dimensional.
    log_probs:
        Current log probabilities of the actions taken (under current policy).
        One dimensional.
    old_log_probs:
        Old log probabilities of the same actions (under previous policy). One
        dimensional.
    ppo_clip:
        Clip parameters for the PPO loss.

    """
    _targets = targets.detach()
    prob_ratio = torch.exp(log_probs - old_log_probs.detach())
    loss1 = prob_ratio * _targets
    loss2 = torch.clamp(prob_ratio, 1.0 - ppo_clip, 1.0 + ppo_clip) * _targets
    return - torch.min(loss1, loss2).mean()

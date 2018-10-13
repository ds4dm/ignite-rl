# coding: utf-8

"""A collection of auxilliary functions for reinforcement learning."""

import torch
import torch.nn.functional as F


def value_td_residuals(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount: float
) -> torch.Tensor:
    """Compute TD residual of state value function.

    All tensors must be one dimensional.
    """
    return rewards + (discount * next_values) - values


def discounted_sum(x: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute a discounted sum for every element.

    Given a one dimension input `x` of size `n`, the output tensor has size `n`
    where every output is given by:
        out[j] = sum_{0 <= i < n-j} x[i+j] * discount**j
    """
    n = len(x)
    powers = torch.arange(n, device=x.device, dtype=x.dtype)
    kernel = float(discount) ** powers
    returns = F.conv1d(x.view(1, 1, -1), kernel.view(1, 1, -1), padding=n-1)
    return returns.squeeze()[n-1:]


def returns(rewards: torch.Tensor, discount: float) -> torch.Tensor:
    """Compute the disounted returns given the rewards."""
    return discounted_sum(rewards, discount)


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
    """
    v_td_residuals = value_td_residuals(rewards, values, discount)
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
        Often returns or advantages. What the probilities ration is multiplied
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

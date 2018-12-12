# coding:utf-8

import torch
import numpy as np

import irl.functional as Firl


def test_value_td_residuals(device):
    values = torch.rand(11, device=device)
    reward = torch.rand(10, device=device)
    done = torch.rand(10, device=device) > .6
    Firl.value_td_residuals(reward, values[:-1], values[1:], done, .9)
    Firl.value_td_residuals(reward, values[:-1], values[1:], done.float(), .9)


def test_discounted_sum(device):
    X = torch.arange(4, dtype=torch.float, device=device)
    expected = torch.tensor(
        [0.1234, 1.234, 2.34, 3.4], device=device, dtype=torch.float)
    result = Firl.discounted_sum(X, 0.1, last=4)
    assert result.device == device
    assert (result == expected).all().item()


def test_normalize_1d(device):
    result = Firl.normalize_1d(2 * torch.rand(10, device=device) + 3)
    assert result.device == device
    assert np.isclose(result.cpu().mean(), 0, atol=1e-4)
    assert np.isclose(result.cpu().std(), 1, atol=1e-1)


def test_generalize_advatange_estimation(device):
    Firl.generalize_advatange_estimation(
        torch.rand(10, device=device),
        torch.rand(11, device=device),
        dones=torch.rand(10, device=device) > .6
    )


def test_ppo_loss(device):
    targets = torch.rand(10, device=device, requires_grad=True)
    log_probs = torch.rand(10, device=device, requires_grad=True)
    old_log_probs = torch.rand(10, device=device, requires_grad=True)
    loss = Firl.ppo_loss(targets, -log_probs, -old_log_probs)
    assert loss.device == device
    assert loss.shape == torch.Size([])

    loss.backward()
    assert targets.grad is None
    assert log_probs.grad is not None
    assert old_log_probs.grad is None

# coding: utf-8

"""Pytest fixtures."""

import pytest
import torch


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> torch.device:
    """Device to run code on."""
    _device = torch.device(request.param)
    if _device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip()
    return _device

# coding: utf-8

import attr
import torch

import irl.exploration.transforms as T
import irl.exploration.data as data


@attr.s(auto_attribs=True)
class Obs:
    x: torch.Tensor = torch.rand(3)


def test_compose():
    f = T.compose(lambda x: x + 2, lambda y: y ** 2)
    assert f(0) == 4


def test_WithReturns():
    Transition = attr.make_class("Transition", ("reward", "obs"))
    trajectory = [Transition(i, Obs()) for i in range(4)]
    transform = T.WithReturns(discount=0.1, normalize=True)

    transformed = transform(trajectory)
    assert isinstance(transformed, list)
    assert len(transformed) == len(trajectory)
    assert all(hasattr(t, "retrn") for t in transformed)
    assert all(isinstance(t.retrn, float) for t in transformed)
    assert all(hasattr(t, "obs") for t in transformed)
    assert all(isinstance(t.obs, Obs) for t in transformed)


def test_WithGAE():
    Transition = attr.make_class("T", ("reward", "critic_value", "done", "obs"))
    trajectory = [Transition(i, 3.0, False, Obs()) for i in range(4)]
    transform = T.WithGAE(discount=0.1, normalize=True, lambda_=0.9)

    transformed = transform(trajectory)
    assert isinstance(transformed, list)
    assert len(transformed) == len(trajectory) - 1
    assert all(hasattr(t, "gae") for t in transformed)
    assert all(isinstance(t.gae, float) for t in transformed)
    assert all(hasattr(t, "obs") for t in transformed)
    assert all(isinstance(t.obs, Obs) for t in transformed)

    trajectory.append(Transition(9, 3.0, True, Obs()))
    transformed = transform(trajectory)
    assert len(transformed) == len(trajectory)


def test_PinIfCuda(device):
    Transition = attr.make_class("Transition", ("x",), bases=(data.Data,))
    trajectory = [Transition(torch.rand(5, device=device)) for _ in range(4)]
    transform = T.PinIfCuda(device=device)

    transformed = transform(trajectory)
    assert isinstance(transformed, list)
    assert len(transformed) == len(trajectory)
    assert all(isinstance(t.x, torch.Tensor) for t in transformed)
    if device.type == "cuda":
        assert all(t.x.is_pinned() for t in transformed)

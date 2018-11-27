# coding: utf-8

import attr
import torch

import irl.exploration.transforms as T


def test_compose():
    f = T.compose(
        lambda x: x+2,
        lambda y: y**2
    )
    assert f(0) == 4


def test_WithReturns():
    Transition = attr.make_class("Transition", ("obs", "reward"))
    trajectory = [Transition(torch.rand(10), i) for i in range(4)]
    transform = T.WithReturns(discount=.1, normalize=True)

    transformed = transform(trajectory)
    assert isinstance(transformed, list)
    assert len(transformed) == len(trajectory)
    assert hasattr(transformed[0], "retrn")

# coding: utf-8

import attr
import torch

import irl.exploration.transforms as T
import irl.exploration.data as data


@attr.s(auto_attribs=True)
class Obs:
    x: torch.Tensor = torch.rand(3)


def approx_eq(x: torch.Tensor, y: torch.Tensor, epsilon: float = 1e-5) -> bool:
    return torch.all(torch.abs(x - y) < epsilon).item()


def test_compose():
    f = T.compose(lambda x: x + 2, lambda y: y ** 2)
    assert f(0) == 4


def test_MultiTaskNormalizer_normalize():
    normalizer = T._MultiTaskNormalizer()

    in1 = torch.rand(1)
    out1 = normalizer.normalize(in1)
    assert (out1 == 0.0).item()
    assert len(normalizer.running_means) == len(normalizer.running_vars) == 0

    in2 = torch.rand(10)
    out2 = normalizer.normalize(in2, task_id=1)
    assert approx_eq(out2.mean(), 0.0)
    assert approx_eq(normalizer.running_means[1], in2.mean())

    in3 = 3 * torch.rand(10)
    out3 = normalizer.normalize(in3, task_id=1)
    assert approx_eq(out3.mean(), 0.0)
    assert not approx_eq(normalizer.running_means[1], in3.mean())

    out11 = normalizer.normalize(in1, task_id=1)
    assert not approx_eq(out11, 0.0)


def test_MultiTaskNormalizer_denormalize():
    normalizer = T._MultiTaskNormalizer()
    in1 = torch.rand(10, dtype=torch.float64)
    out1 = normalizer.normalize(in1, task_id=1)
    assert approx_eq(normalizer.denormalize(out1, task_id=1), in1, epsilon=1e-2)
    assert normalizer.denormalize(out1) is out1


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
    assert abs(sum(t.retrn for t in transformed)) < 1e-5

    running_mean = transform.normalizer.running_means[None]
    assert isinstance(running_mean, torch.Tensor)
    assert running_mean.shape == (1,)


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

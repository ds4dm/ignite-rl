# coding: utf-8

import attr
import torch
import mock

from irl.exploration.data import Data, AttribMeta


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Something(Data):
    x: bool = True
    y: torch.Tensor = torch.rand(3)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Example(Data):
    a: torch.Tensor = torch.rand(10)
    b: int = 0
    c: Something = Something()
    d: list = attr.ib(default="d", metadata={AttribMeta.COLLATE: " ".join})


def test_data_move(device):
    d = Example()
    new_d = d.to(device=device)
    assert new_d.a.device == device
    assert new_d.c.y.device == device


def test_apply():
    d = Example()
    m = mock.MagicMock()
    d.apply(m)
    assert m.call_count == 2


def test_pin_memory(device):
    if device.type == "cuda":
        d = attr.evolve(Example(), a=torch.sparse.FloatTensor())
        d.pin_memory()


def test_get_batched_class():
    BatchClass = Example.get_batch_class()
    assert len(attr.fields(BatchClass)) == 4


def test_collate():
    d = Example()
    batch = Example.collate([d, d])
    assert isinstance(batch, Example.get_batch_class())
    assert isinstance(batch.a, torch.Tensor)
    assert batch.a.shape == (2, 10)
    assert isinstance(batch.b, torch.Tensor)
    assert batch.b.shape == (2,)
    assert isinstance(batch.c, Something.get_batch_class())
    assert isinstance(batch.c.x, torch.Tensor)
    assert batch.c.x.shape == (2,)
    assert batch.c.x.dtype == torch.uint8
    assert isinstance(batch.c.y, torch.Tensor)
    assert batch.c.y.shape == (2, 3)
    assert batch.d == "d d"


def test_from_dict():
    e = Example()
    d = attr.asdict(e)
    assert isinstance(d, dict)
    assert isinstance(d["c"], dict)
    f = Example.from_dict(d)
    assert isinstance(f, Example)
    assert isinstance(f.c, Something)

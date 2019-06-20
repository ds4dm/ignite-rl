# coding: utf-8

import threading

import pytest
import mock
import numpy as np
import scipy.sparse as sp
import torch

import irl.utils as utils


def test_every():
    m = mock.MagicMock()
    f = utils.every(3, start=True)(m)
    for i in range(2):
        f(i)
    m.assert_called_once()
    m.assert_called_with(0)
    for i in range(2, 4):
        f(i)
    assert m.call_count == 2
    m.assert_called_with(3)


def test_apply_to_type():
    t = torch.rand(3)
    t_applied = utils.apply_to_tensor(t, lambda x: x.byte())
    assert not t_applied.any().item()

    seq = [torch.rand(3) for _ in range(4)]
    seq_applied = utils.apply_to_tensor(seq, lambda x: x.byte())
    assert len(seq) == len(seq_applied)
    assert all(not x.any().item() for x in seq_applied)

    map = {i: torch.rand(3) for i in range(4)}
    map_applied = utils.apply_to_tensor(map, lambda x: x.byte())
    assert len(map) == len(map_applied)
    assert all(not x.any().item() for x in map_applied.values())

    mixed_rec = (t, seq, map)
    mixed_rec_applied = utils.apply_to_tensor(mixed_rec, lambda x: x.byte())
    assert len(mixed_rec_applied) == len(mixed_rec)
    assert isinstance(mixed_rec_applied[0], torch.Tensor)
    assert isinstance(mixed_rec_applied[1], list)
    assert isinstance(mixed_rec_applied[2], dict)


def test_from_numpy_sparse():
    x = np.random.rand(10, 10)
    x[x < 0.7] = 0
    x_t = utils._from_numpy_sparse(x)

    assert isinstance(x_t, torch.Tensor)
    assert x_t.shape == x.shape

    x_sp = sp.csr_matrix(x)
    x_sp_t = utils._from_numpy_sparse(x_sp)
    assert isinstance(x_sp_t, torch.Tensor)
    assert x_sp_t.is_sparse
    assert x_sp_t.shape == x_sp.shape


def test_default_merge(device):
    data = [{"a": 1, "b": torch.rand(2, device=device), 3: (10, 3)}] * 4
    data_merged = utils.default_merge(data)
    assert isinstance(data_merged, dict)
    assert (data_merged["a"] == torch.ones(4).long()).all().item()
    assert data_merged["b"].shape == (4, 2)
    assert data_merged["b"].device == device
    assert isinstance(data_merged[3], tuple)
    assert (data_merged[3][0] == 10 * torch.ones(4).long()).all().item()


@pytest.mark.timeout(1)
def test_RWLock():
    lock = utils.RWLock()

    def dummy_read():
        with lock.reader():
            assert True

    with lock.reader():
        t = threading.Thread(target=dummy_read)
        t.start()
        t.join()


@pytest.mark.timeout(1)
def test_Counter():
    counter = utils.Counter(10)

    def wait_assert():
        counter.wait()
        assert counter.count >= 10

    t = threading.Thread(target=wait_assert)
    t.start()

    for _ in range(10):
        counter += 1

    t.join()
    counter.reset()
    assert counter.count == 0
    assert not counter.event.is_set()

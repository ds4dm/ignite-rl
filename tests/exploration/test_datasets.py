# coding: utf-8

import attr

import irl.exploration.datasets as D


def test_Trajectories():
    Trans = attr.make_class("Transition", ("val", "done"))
    ds = D.Trajectories(lambda traj: [x.val**2 for x in traj])

    ds.concat([Trans(1, False), Trans(2, False), Trans(3, True)])
    assert len(ds) == len(ds.data) == 3
    assert ds[0] == 1

    ds.append(Trans(4, False))
    assert len(ds) == 3
    assert len(ds.partial_trajectory) == 1
    ds.append(Trans(5, True))
    assert len(ds) == 5
    assert len(ds.partial_trajectory) == 0

    ds.append(Trans(6, False))
    ds.clear()
    assert len(ds) == len(ds.partial_trajectory) == 0


def test_MemoryReplay():
    ds = D.MemoryReplay(lambda x: x-1, capacity=3)

    for i in range(3):
        ds.append(i)
        assert ds[0] == -1

    assert len(ds) == len(ds.data) == 3
    ds.append(3)
    assert ds[0] == 0
    assert len(ds) == len(ds.data) == 3

    ds.clear()
    assert len(ds) == 0

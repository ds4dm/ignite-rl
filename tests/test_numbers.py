# coding: utf-8


import irl.numbers as num


def test_LinearDecay():
    x = num.LinearDecay(start=1, end=0.5, horizon=2)
    assert isinstance(x, num.Stepable)
    assert x == 1
    x.step()
    assert x == 0.75
    x.step()
    assert x == 0.5
    x.step()
    assert x == 0.5

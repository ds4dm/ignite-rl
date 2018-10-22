# coding: utf-8

from functools import reduce, partial
from typing import Callable


def compose(*transforms: Callable) -> Callable:
    return partial(reduce, lambda x, f: f(x), transforms)

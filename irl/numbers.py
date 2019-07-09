# coding: utf-8

"""Mutable numbers.

Adapted from Nauka:
https://github.com/obilaniu/Nauka/blob/master/src/nauka/utils/lr.py
"""

import abc
import math
import numbers

import attr


@attr.s(auto_attribs=True, slots=True, cmp=False, hash=False)
class Stepable(numbers.Real, abc.ABC):
    """A mutable float, which value can be updated using a `step` method."""

    steps: int = 0

    def __bool__(self):  # noqa
        return bool(float(self))

    def __abs__(self):  # noqa
        return abs(float(self))

    def __add__(self, x):  # noqa
        return float(self) + x

    def __ceil__(self):  # noqa
        return math.ceil(float(self))

    def __divmod__(self, x):  # noqa
        return divmod(float(self), x)

    def __eq__(self, x):  # noqa
        return float(self) == x

    def __floor__(self):  # noqa
        return math.floor(float(self))

    def __floordiv__(self, x):  # noqa
        return float(self) // x

    def __format__(self, f):  # noqa
        return format(float(self), f)

    def __ge__(self, x):  # noqa
        return float(self) >= x

    def __gt__(self, x):  # noqa
        return float(self) > x

    def __le__(self, x):  # noqa
        return float(self) <= x

    def __lt__(self, x):  # noqa
        return float(self) < x

    def __mod__(self, x):  # noqa
        return float(self) % x

    def __mul__(self, x):  # noqa
        return float(self) * x

    def __neg__(self):  # noqa
        return -float(self)

    def __pos__(self):  # noqa
        return +float(self)

    def __pow__(self, x):  # noqa
        return pow(float(self), x)

    def __radd__(self, x):  # noqa
        return x + float(self)

    def __rdivmod__(self, x):  # noqa
        return divmod(x, float(self))

    def __rfloordiv__(self, x):  # noqa
        return x // float(self)

    def __rmod__(self, x):  # noqa
        return x % float(self)

    def __rmul__(self, x):  # noqa
        return x * float(self)

    def __round__(self, n):  # noqa
        return round(float(self), n)

    def __rpow__(self, x):  # noqa
        return pow(x, float(self))

    def __rsub__(self, x):  # noqa
        return x - float(self)

    def __rtruediv__(self, x):  # noqa
        return x / float(self)

    def __sub__(self, x):  # noqa
        return float(self) - x

    def __truediv__(self, x):  # noqa
        return float(self) / x

    def __trunc__(self):  # noqa
        return math.trunc(float(self))

    def step(self) -> None:
        """Increase steps counter."""
        self.steps += 1

    @abc.abstractmethod
    def __float__(self) -> float:
        """Return the value at the current step."""
        return NotImplemented


@attr.s(auto_attribs=True, slots=True, cmp=False, hash=False)
class LinearDecay(Stepable):
    """A linear decay number with minimum value."""

    start: float = 1.0
    end: float = 0.05
    horizon: float = 10000

    def __float__(self) -> float:
        """Return the value at the current step."""
        val = max(1 - (self.steps / self.horizon), 0)
        val *= self.start - self.end
        val += self.end
        return val

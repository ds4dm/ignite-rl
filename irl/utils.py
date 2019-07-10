# coding: utf-8

"""Utilities."""

import enum
import threading
import contextlib
import collections.abc
from functools import wraps
from numbers import Number
from typing import Callable, Sequence, Mapping, Union, Any, Optional

import attr
import numpy as np
import scipy.sparse as sp
import torch


Decorator = Callable[[Callable], Callable]


class NameEnum(enum.Enum):
    """Enum class to override `auto` behaviour."""

    def _generate_next_value_(name, start, count, last_values):
        """Return the name of the enum attribute for `auto`."""
        return name.lower()


def every(steps: int, start: bool = False) -> Decorator:
    """Retrun decorator.

    Return a decorator that run the decorated function only one every `steps`
    calls. Other calls do not run the decorated function and return `None`.
    Primary use is to decorate Ignite handlers.

    Parameters
    ----------
    steps:
        Number of steps between every decorated function calls.
    start:
        Whether or not to call the decorated function on the first call.

    Returns
    -------
    decorator:
        The decorator to use on a function.

    """

    def decorator(func: Callable) -> Callable:
        step_count = -1 if start else 0

        @wraps(func)
        def new_func(*args, **kwargs):
            nonlocal step_count
            step_count += 1
            if steps is None or steps <= 0:
                return None
            if step_count % steps == 0:
                return func(*args, **kwargs)
            else:
                return None

        return new_func

    return decorator


def apply_to_tensor(
    input: Union[Any, Sequence, Mapping],
    function: Callable[[torch.Tensor], torch.Tensor],
) -> Union[Any, Sequence, Mapping]:
    """Apply a function on a tensor, a sequence, or a mapping of tensors."""
    return apply_to_type(input, torch.Tensor, function)


def apply_to_type(
    input: Union[Any, Sequence, Mapping], in_type, function: Callable
) -> Union[Any, Sequence, Mapping]:
    """Apply a function on elements of a given type.

    Apply a function on an object of `input_type`, a sequence, or a mapping
    of objects of `input_type`.
    """
    if isinstance(input, in_type):
        return function(input)
    elif isinstance(input, collections.abc.Mapping):
        values = list(input.values())
        if len(values) > 0 and isinstance(values[0], in_type):
            return {k: apply_to_type(v, in_type, function) for k, v in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        if len(input) > 0 and isinstance(input[0], in_type):
            return [apply_to_type(sample, in_type, function) for sample in input]
    return input


def _from_numpy_sparse(
    t: Union[np.ndarray, sp.spmatrix], dtype: Optional[torch.dtype] = None
) -> torch.Tensor:
    """Create a Tensor from numpy array or scipy sparse matrix."""
    if isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(dtype)
    elif isinstance(t, sp.spmatrix):
        t_coo = t.tocoo(copy=False)
        return torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack((t_coo.row, t_coo.col))),
            values=t_coo.data,
            size=t_coo.shape,
        ).to(dtype)
    else:
        raise TypeError(
            "Argument of type {t.__class__} is neither a"
            "numpy array not a scipy sparse matrix."
        )


def from_numpy_sparse(
    input: Union[Any, Sequence, Mapping]
) -> Union[Any, Sequence, Mapping]:
    """Convert numpy arrays and scipys sparse matrices to Tensors."""
    return apply_to_type(
        input, in_type=(np.ndarray, sp.spmatrix), function=_from_numpy_sparse
    )


def default_merge(
    seq: Sequence[Union[Number, torch.Tensor, Sequence, Mapping]]
) -> Union[torch.Tensor, Sequence, Mapping]:
    """Stack data points.

    Stack a sequence of data points, where data point are numbers, tensors,
    sequences, or mappings of number/tensors.
    """
    elem = seq[0]
    if isinstance(elem, Number):
        return torch.tensor(seq)
    elif isinstance(elem, torch.Tensor):
        return torch.stack(seq)
    elif isinstance(elem, collections.abc.Mapping):
        return {k: default_merge([e[k] for e in seq]) for k in elem.keys()}
    elif isinstance(elem, collections.abc.Sequence):
        merged = [default_merge([e[i] for e in seq]) for i in range(len(elem))]
        return elem.__class__(merged)
    return seq


class RWLock(object):
    """Reader-writer lock with preference to writers.

    Readers can access a resource simultaneously.
    Writers get an exclusive access.

    API is self-descriptive:
        reader_enters()
        reader_leaves()
        writer_enters()
        writer_leaves()
    """

    def __init__(self):
        """Initialize read-write lock."""
        self.mutex = threading.RLock()
        self.can_read = threading.Semaphore(0)
        self.can_write = threading.Semaphore(0)
        self.active_readers = 0
        self.active_writers = 0
        self.waiting_readers = 0
        self.waiting_writers = 0

    def reader_enters(self):
        """Acquire reader lock."""
        with self.mutex:
            if self.active_writers == 0 and self.waiting_writers == 0:
                self.active_readers += 1
                self.can_read.release()
            else:
                self.waiting_readers += 1
        self.can_read.acquire()

    def reader_leaves(self):
        """Release reader lock."""
        with self.mutex:
            self.active_readers -= 1
            if self.active_readers == 0 and self.waiting_writers != 0:
                self.active_writers += 1
                self.waiting_writers -= 1
                self.can_write.release()

    @contextlib.contextmanager
    def reader(self):
        """Reader lock as context manager."""
        self.reader_enters()
        try:
            yield
        finally:
            self.reader_leaves()

    def writer_enters(self):
        """Acquire writer lock."""
        with self.mutex:
            if (
                self.active_writers == 0
                and self.waiting_writers == 0
                and self.active_readers == 0
            ):
                self.active_writers += 1
                self.can_write.release()
            else:
                self.waiting_writers += 1
        self.can_write.acquire()

    def writer_leaves(self):
        """Release writer lock."""
        with self.mutex:
            self.active_writers -= 1
            if self.waiting_writers != 0:
                self.active_writers += 1
                self.waiting_writers -= 1
                self.can_write.release()
            elif self.waiting_readers != 0:
                t = self.waiting_readers
                self.waiting_readers = 0
                self.active_readers += t
                while t > 0:
                    self.can_read.release()
                    t -= 1

    @contextlib.contextmanager
    def writer(self):
        """Writer lock as context manager."""
        self.writer_enters()
        try:
            yield
        finally:
            self.writer_leaves()


@attr.s(auto_attribs=True)
class Counter:
    """A simple thread safe counter that can be waited on some value.

    Attributes
    ----------
    target:
        When the counter reaches this value, the event is set.
    count:
        The internal counter.
    lock:
        Reentrant lock for accessing the internal count.
    event:
        Signal that the counter reached its target.

    """

    target: int = 1000
    count: int = attr.ib(init=False, default=0)
    lock: threading.RLock = attr.ib(init=False, factory=threading.RLock)
    event: threading.Event = attr.ib(init=False, factory=threading.Event)

    def __iadd__(self, amount: int) -> "Counter":
        """Increase the counter."""
        with self.lock:
            self.count += amount
            if self.count >= self.target:
                self.event.set()
        return self

    def wait(self, timeout: int) -> None:
        """Wait unitl the counter reaches the target value."""
        self.event.wait(timeout)

    def reset(self) -> None:
        """Reset the object."""
        with self.lock:
            self.count = 0
        self.event.clear()

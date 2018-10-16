# coding: utf-8

"""Utilities."""

import collections.abc
from functools import wraps
from numbers import Number
from typing import Callable, Sequence, Mapping, Union, Any, TypeVar

import numpy as np
import scipy.sparse as sp
import torch


Decorator = Callable[[Callable], Callable]
T = TypeVar("T")


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
    input: Union[
        torch.Tensor,
        Sequence[torch.Tensor],
        Mapping[Any, torch.Tensor]],
    function: Callable[[torch.Tensor], torch.Tensor]
) -> Union[torch.Tensor, Sequence[torch.Tensor], Mapping[Any, torch.Tensor]]:
    """Apply a function on a tensor, a sequence, or a mapping of tensors."""
    return apply_to_type(input, torch.Tensor, function)


def apply_to_type(
    input: T,
    in_type,
    function: Union[T, Sequence[T], Mapping[Any, T]]
) -> Union[T, Sequence[T], Mapping[Any, T]]:
    """Apply a function on elements of a given type.

    Apply a function on an object of `input_type`, a sequence, or a mapping
    of objects of `input_type`.
    """
    if isinstance(input, in_type):
        return function(input)
    elif isinstance(input, collections.abc.Mapping):
        values = list(input.values())
        if len(values) > 0 and isinstance(values[0], in_type):
            return {k: apply_to_type(v, in_type, function)
                    for k, v in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        if len(input) > 0 and isinstance(input[0], in_type):
            return [apply_to_type(sample, in_type, function)
                    for sample in input]
    return input


def from_numpy_sparse(t: Union[np.ndarray, sp.spmatrix]) -> torch.Tensor:
    """Create a Tensor from numpy array or scipy sparse matrix."""
    if isinstance(t, np.ndarray):
        return torch.from_numpy(t)
    elif isinstance(t, sp.spmatrix):
        t_coo = t.tocoo(copy=False)
        return torch.sparse_coo_tensor(
            indices=torch.from_numpy(np.vstack((t_coo.row, t_coo.col))),
            values=t_coo.data,
            size=t_coo.shape
        )
    else:
        raise TypeError("Argument of type {t.__class__} is neither a"
                        "numpy array not a scipy sparse matrix.")


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

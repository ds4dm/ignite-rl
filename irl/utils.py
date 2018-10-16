# coding: utf-8

"""Utilities."""

import collections.abc
from functools import wraps
from typing import Callable, Sequence, Mapping, Union, Any, TypeVar

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
    func: Callable[[torch.Tensor], torch.Tensor]
) -> Union[torch.Tensor, Sequence[torch.Tensor], Mapping[Any, torch.Tensor]]:
    """Apply a function on a tensor, a sequence, or a mapping of tensors."""
    return apply_to_type(input, torch.Tensor, func)


def apply_to_type(
    input: T,
    in_type,
    func: Union[T, Sequence[T], Mapping[Any, T]]
) -> Union[T, Sequence[T], Mapping[Any, T]]:
    """Apply a function on elements of a given type.

    Apply a function on an object of `input_type`, a sequence, or a mapping
    of objects of `input_type`.
    """
    if isinstance(input, in_type):
        return func(input)
    elif isinstance(input, collections.abc.Mapping):
        return {k: apply_to_type(v, in_type, func) for k, v in input.items()}
    elif isinstance(input, collections.abc.Sequence):
        return [apply_to_type(sample, in_type, func) for sample in input]
    else:
        raise TypeError(("input must contain {}, dicts or lists; found {}"
                         .format(in_type, type(input))))

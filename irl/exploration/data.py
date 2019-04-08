# coding: utf-8

"""Utilities to create pytorch data classes."""

from typing import Optional, Callable, Sequence

import attr
import torch
from torch.utils.data.dataloader import default_collate


class Data:
    """A base data class to extend with `attr.ib`."""

    def apply(self, func: Callable) -> "Data":
        """Apply a function on all underlying tensors and Data classes."""
        to_evolve = {}
        for name in attr.fields_dict(self.__class__):
            val = getattr(self, name)
            if isinstance(val, (torch.Tensor, Data)):
                to_evolve[name] = func(val)
        return attr.evolve(self, **to_evolve)

    def to(
        self,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        non_blocking: bool = False,
    ) -> "Data":
        """Change type/device of data."""
        return self.apply(
            lambda x: x.to(dtype=dtype, device=device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "Data":
        """Move the underlying tensors to pinned memory."""

        def mapper(t):
            # Cannot pin memory of sparse tensor
            if isinstance(t, torch.Tensor) and t.is_sparse:
                return t
            else:
                return t.pin_memory()

        return self.apply(mapper)

    def share_memory_(self) -> "Data":
        """Move the underlying tensors to shared memory."""
        return self.apply(lambda x: x.share_memory_())

    def cpu(self) -> "Data":
        """Move the underlying tensors to cpu memory."""
        return self.apply(lambda x: x.cpu())

    def numpy(self) -> "Data":
        """Convert the underlying tensors to numpy."""
        return self.apply(lambda x: x.numpy())

    @classmethod
    def get_batch_class(cls):
        """Return a batched version of this class."""
        # Done lazily as we want to make sure this is naturally overriden
        try:
            return cls.__batched_Class
        except AttributeError:
            cls.__batched_Class = attr.make_class(
                name="Batch_" + cls.__name__, attrs=[], bases=(cls,)
            )
            return cls.__batched_Class

    @classmethod
    def collate(cls, points: Sequence["Data"]) -> "Batch_Data":
        """Collate a list of points into a batched class.

        The class used for batching can be found in `cls.batched_Class` and
        contains similar fields as this class.
        """
        batch_dict = {}
        for name in attr.fields_dict(cls):
            member_list = [getattr(p, name) for p in points]
            if isinstance(member_list[0], Data):
                klass = member_list[0].__class__
                batch_dict[name] = klass.collate(member_list)
            else:
                batch_dict[name] = default_collate(member_list)
        return cls.get_batch_class()(**batch_dict)

    @classmethod
    def from_dict(cls, inputs) -> "Data":
        """Build object from dictionnary recursively."""
        parameters = {}
        fields_dict = attr.fields_dict(cls)
        for name, val in inputs.items():
            attrib = fields_dict[name]
            if issubclass(attrib.type, Data):
                parameters[name] = attrib.type.from_dict(val)
            else:
                parameters[name] = val
        return cls(**parameters)

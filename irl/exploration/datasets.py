# coding: utf-8

"""Dataset classes for reinforcement learning.

Data point are related to transition and trajectories.
"""

from typing import List, Union, Callable, Generic, TypeVar, Optional

import attr
from torch.utils.data import Dataset

from irl.exploration.explorer import Transition


Data = TypeVar("Data")
Transform = Callable[[Transition], Data]
TrajectoryTransform = Callable[[List[Transition]], List[Data]]


@attr.s(auto_attribs=True)
class Trajectories(Dataset, Generic[Data]):
    """A dataset concatenating trajectories.

    The dataset can take one (or many using `compose`) trajectory
    transformation function. It is the apropiate place to discard unecessary
    information, or compute values depending on the whole trajectory, such
    as the returns.

    Attributes
    ----------
    trajectory_transform:
        Transformation to apply on new trajectories added.
    data:
        A list of all transformed transitions.

    """

    trajectory_transform: TrajectoryTransform = lambda x: x
    data: List[Data] = attr.ib(factory=list)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Data, "Trajectories"]:
        """Select Transition or sub trajectory."""
        selected = self.transitions[idx]
        if isinstance(selected, list):
            return Trajectories(self.trajectory_transform, selected)
        else:
            return selected

    def __len__(self) -> int:
        """Length of the trajectory."""
        return len(self.data)

    def concat(self, trajectory: List[Transition]) -> None:
        """Add a new trajectory to the dataset."""
        self.data[len(self.data):] = self.trajectory_transform(trajectory)

    def clear(self) -> None:
        """Empty the dataset."""
        self.data.clear()


@attr.s(auto_attribs=True)
class MemoryReplay(Dataset, Generic[Data]):
    """A dataset of transitions.

    Stores transitions after applying a possible transformation. Can also
    limit the capacity of the replay buffer to a certain amount. Oldest
    elements are removed.

    Attributes
    ----------
    transform:
        Transformation to apply on new transition added.
    capacity:
        Optional capacity limit.
    data:
        A list of all transformed transitions.

    """

    transform: Transform = lambda x: x
    capacity: Optional[int] = None
    data: List[Data] = attr.ib(factory=list)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Data, "Trajectories"]:
        """Select Transition or sub trajectory."""
        selected = self.transitions[idx]
        if isinstance(selected, list):
            return MemoryReplay(self.transform, selected)
        else:
            return selected

    def __len__(self) -> int:
        """Length of the trajectory."""
        return len(self.data)

    def append(self, transition: Transition) -> None:
        """Add a transition to the dataset."""
        self.data.append(self.transform(transition))
        if self.capacity is not None and len(self) > self.capacity:
            self.data.pop()

    def clear(self) -> None:
        """Empty the dataset."""
        self.data.clear()

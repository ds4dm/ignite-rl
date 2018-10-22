# coding: utf-8

from typing import List, Union, Callable, Generic, TypeVar, Optional

import attr
from torch.utils.data import Dataset

from irl.exploration.explorer import Transition


Data = TypeVar("Data")
Transform = Callable[[Transition], Data]
TrajectoryTransform = Callable[[List[Transition]], List[Data]]


@attr.s(auto_attribs=True)
class Trajectories(Dataset, Generic[Data]):

    trajectory_transform: Optional[TrajectoryTransform] = None
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
        self.data[len(self.data):] = self.trajectory_transform(trajectory)

    def clear(self) -> None:
        self.data.clear()


@attr.s(auto_attribs=True)
class MemoryReplay(Dataset, Generic[Data]):

    transform: Optional[Transform] = None
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
        self.data.append(self.transform(transition))
        if self.capacity is not None and len(self) > self.capacity:
            self.data.pop()

    def clear(self) -> None:
        self.data.clear()

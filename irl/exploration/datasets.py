# coding: utf-8

"""Dataset classes for reinforcement learning.

Data point are related to transition and trajectories.
"""

from typing import List, Union, Callable, Generic, TypeVar, Optional, Iterator

import attr
from torch.utils.data import Dataset

from irl.exploration.explorer import Transition
import irl.utils as utils


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

    The class is thread safe for simple operations.

    Attributes
    ----------
    trajectory_transform:
        Transformation to apply on new trajectories added.
    data:
        A list of all transformed transitions.
    partial_trajectory:
        A buffer to put transitions, waiting for the episode to terminate.
    data_lock:
        A lock that let multiple readers xor one writter acess the data at the
        time.
    par_traj_lock:
        A lock that let multiple readers xor one writter acess the partial
        trajectory at the time.

    """

    trajectory_transform: TrajectoryTransform = lambda x: x

    data: List[Data] = attr.ib(init=False, factory=list)
    partial_trajectory: List[Transition] = attr.ib(init=False, factory=list)
    data_lock: utils.RWLock = attr.ib(init=False, factory=utils.RWLock)
    par_traj_lock: utils.RWLock = attr.ib(init=False, factory=utils.RWLock)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Data, "Trajectories"]:
        """Select Transition or sub trajectory."""
        with self.data_lock.reader():
            selected = self.data[idx]
            if isinstance(selected, list):
                return Trajectories(self.trajectory_transform, selected)
            else:
                return selected

    def __len__(self) -> int:
        """Length of the trajectory."""
        with self.data_lock.reader():
            return len(self.data)

    def __iter__(self) -> Iterator[Data]:
        """Iterate over the data.

        Method defined for when using a Dataloder is overkill.
        """
        with self.data_lock.reader():
            return iter(self.data)

    def concat(self, trajectory: List[Transition]) -> None:
        """Add a new trajectory to the dataset."""
        with self.data_lock.writer():
            self.data += self.trajectory_transform(trajectory)

    def append(self, transition: Transition) -> None:
        """Append a single transition.

        Stage the transition in the buffer `partial_trajectory`. Use
        `terminate_trajectory` to signal start a new trajectory.
        """
        with self.par_traj_lock.writer():
            self.partial_trajectory.append(transition)

    def terminate_trajectory(self) -> None:
        """Terminate a partial trajectory.

        When the episode is assume to be over, call this method.
        The trajectory is passed to `concat` and the buffer in cleared.
        Note: this is different from `done` in environement because one may
        truncate the episode before a terminal state.
        """
        self.concat(self.partial_trajectory)
        with self.par_traj_lock.writer():
            self.partial_trajectory.clear()

    def clear(self) -> None:
        """Empty the dataset."""
        with self.data_lock.writer():
            self.data.clear()
        with self.par_traj_lock.writer():
            self.partial_trajectory.clear()


@attr.s(auto_attribs=True)
class MemoryReplay(Dataset, Generic[Data]):
    """A dataset of transitions.

    Stores transitions after applying a possible transformation. Can also
    limit the capacity of the replay buffer to a certain amount. Oldest
    elements are removed.

    The class is thread safe for simple operations.

    Attributes
    ----------
    transform:
        Transformation to apply on new transition added.
    capacity:
        Optional capacity limit.
    data:
        A list of all transformed transitions.
    lock:
        A lock that let multiple readers xor one writter acess the data at the
        time.

    """

    transform: Transform = lambda x: x
    capacity: Optional[int] = None

    data: List[Data] = attr.ib(init=False, factory=list)
    lock: utils.RWLock = attr.ib(init=False, factory=utils.RWLock)

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Data, "Trajectories"]:
        """Select Transition or sub trajectory."""
        with self.lock.reader():
            selected = self.data[idx]
            if isinstance(selected, list):
                return MemoryReplay(self.transform, selected)
            else:
                return selected

    def __len__(self) -> int:
        """Length of the trajectory."""
        with self.lock.reader():
            return len(self.data)

    def __iter__(self) -> Iterator[Data]:
        """Iterate over the data.

        Method defined for when using a Dataloder is overkill.
        """
        with self.lock.reader():
            return iter(self.data)

    def append(self, transition: Transition) -> None:
        """Add a transition to the dataset."""
        with self.lock.writer():
            self.data.append(self.transform(transition))
            # not using len(self) because the lock is not reentrant.
            if self.capacity is not None and len(self.data) > self.capacity:
                self.data.pop(0)

    def clear(self) -> None:
        """Empty the dataset."""
        with self.lock.writer():
            self.data.clear()

# coding: utf-8

"""Explorer metrics."""

import abc

import ignite.metrics


class EpisodeAccumulation(ignite.metrics.Metric, abc.ABC):
    """A Metric class that accumulate some measure in an episode.

    Use this Metric with an explorer
    """

    def average(self) -> bool:
        """Return wether to use mean rather than sum."""
        return NotImplemented

    @abc.abstractmethod
    def get_val(self, transition, infos) -> float:
        """Return the measure to be accumulated."""
        return NotImplemented

    def reset(self) -> None:
        """Reset accumulators of the measure."""
        self.val = 0
        self.num = 0

    def update(self, output) -> None:
        """Update the aggregate with new measure value."""
        self.val += self.get_val(*output)
        self.num += 1

    def compute(self) -> float:
        """Return the final aggregated measure."""
        if self.average:
            return self.val / self.num
        else:
            return self.val


class TransitionMetric(EpisodeAccumulation):
    """An episode metric computed using a transition attribute."""

    def __init__(self, name: str, average: bool = False, *args, **kwargs) -> None:
        """Initialize the Metric."""
        super().__init__(*args, **kwargs)
        self.average = average
        self.name = name

    def get_val(self, transition, infos) -> float:
        """Return the transition attribute value."""
        return getattr(transition, self.name)


class InfoMetric(EpisodeAccumulation):
    """An episode metric computed using a the environment infos."""

    def __init__(self, name: str, average: bool = False, *args, **kwargs) -> None:
        """Initialize the Metric."""
        super().__init__(*args, **kwargs)
        self.average = average
        self.name = name

    def get_val(self, transition, infos) -> float:
        """Return the infos attribute value."""
        return infos[self.name]


class Return(TransitionMetric):
    """An Episode metric computing the undiscounted retrun."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize the Metric."""
        super().__init__("reward", False, *args, **kwargs)


class EpisodeLength(EpisodeAccumulation):
    """An Episode metric computing the episode length."""

    average = False

    def get_val(self, transition, infos) -> float:
        """Return one."""
        return 1

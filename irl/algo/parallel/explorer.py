# coding: utf-8

"""A parallel explorer."""

import threading
from typing import Callable, Optional

import torch
from ignite.engine import Engine

from irl.exploration.explorer import Explorer
from irl.exploration.environment import Observation, Action, Environment
from irl.utils import RWLock, Range


class ParallelExplorer(Explorer):
    """A parallel explorer.

    A threaded explorer that holds a read-write lock before calling the
    `select_action`. Be careful that the objects passed to it may also need to
    be duplicated between other explorer (e.g. the environment).
    """

    def __init__(
        self,
        env: Environment,
        select_action: Callable[[Engine, Observation], Action],
        model_lock: RWLock,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Engine:
        """Create an explorer using a read-write lock.

        Parameters
        ----------
        env:
            The environment to explore (be careful no to share it with another
            explorer).
        select_action:
            A function used to select an action. Has access to the engine and
            the iteration number. The current observation is stored under
            `engine.state.observation`. Takes as input the ingine and the
            iteration number, returns the action passed to the environement,
            along with a dictionary (possibly empty) of other variable to
            remember.
        model_lock:
            A read-write lock to be shared with all agents using a same model.
        dtype:
            Type to cast observations in.
        device:
            Device to move observations to before passing it to the
            `select_action` function.

        """
        def __select_action(engine, obs):
            with model_lock.reader():
                return select_action(engine, obs)

        super().__init__(
            env=env,
            select_action=__select_action,
            dtype=dtype,
            device=device
        )

    def run(
        self,
        max_episode_length: Optional[int] = None,
        max_epochs: Optional[int] = None
    ) -> None:
        """Asynchronously run the agent.

        Agent is run in its own thread. Use `None` as a value to specify
        unlimited.
        """
        max_epoch_or_inf = float("inf") if max_epochs is None else max_epochs
        self.thread = threading.Thread(
            target=super().run,
            args=(Range(max_episode_length), max_epoch_or_inf)
        )
        self.thread.start()

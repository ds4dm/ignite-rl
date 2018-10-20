# coding: utf-8

"""Ignite exploration engine.

An engine that explore the environement according to a user define policy.
The explorer can store `Transition`s and `Trajectorie`s during its interaction
with the environment, so that it may latter be use for learning. The explorer
doesn't execute any learning but is the basis for many learning algorithms.
Because it is an `ignite.Engine`, the user can fully interact with it through
handles.

We use `TypeVar` and `Generic` to make the code more readable as we do not
assume any beahviour as for the type of actions and observations: the user is
free to define their own types. For these particluar cases, a function can be
passed to `Trajectory` method to merge observations or actions.
"""

from typing import List, Union, Callable, Optional, Generic

import attr
import numpy as np
import scipy.sparse as sp
import torch
from ignite.engine import Engine, Events

import irl.functional as func
import irl.utils as utils
from irl.environment import Observation, BatchedObservations
from irl.environment import Action, BatchedActions
from irl.environment import Environment


@attr.s(auto_attribs=True, frozen=True)
class Transition(
    Generic[Observation, BatchedObservations, Action, BatchedActions]
):
    """An observed Transition in the environement."""

    observation: Observation
    action: Action
    next_observation: Observation
    reward: float
    done: bool


@attr.s(auto_attribs=True)
class Trajectory(
    Generic[Observation, BatchedObservations, Action, BatchedActions]
):
    """A list of sucessive Transitions."""

    transitions: List[Transition] = attr.ib(factory=list)

    def append(
        self,
        observation: Observation,
        action: Action,
        next_observation: Observation,
        reward: float,
        done: bool
    ) -> None:
        """Save a transition."""
        self.transitions.append(Transition(
            observation=observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done
        ))

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Union[Transition, "Trajectory"]:
        """Select Transition or sub trajectory."""
        selected = self.transitions[idx]
        if isinstance(selected, list):
            return Trajectory(selected)
        else:
            return selected

    def __len__(self) -> int:
        """Length of the trajectory."""
        return len(self.transitions)

    def observations(
        self,
        merge: Callable[
            [List[Observation]], BatchedObservations
        ] = utils.default_merge
    ) -> BatchedObservations:
        """All observations acted upon stacked together."""
        return merge([t.observation for t in self.transitions])

    def actions(
        self,
        merge: Callable[[List[Action]], BatchedActions] = utils.default_merge
    ) -> BatchedActions:
        """All actions stacked together."""
        return merge([t.action for t in self.transitions])

    def next_observations(
        self,
        merge: Callable[
            [List[Observation]], BatchedObservations
        ] = utils.default_merge
    ) -> BatchedObservations:
        """All next observations stacked together."""
        return merge([t.next_observation for t in self.transitions])

    def all_observations(
        self,
        merge: Callable[
            [List[Observation]], BatchedObservations
        ] = utils.default_merge
    ) -> BatchedObservations:
        """All len(self) + 1 observations stacked together.

        When there are memory constraints, this fuctions avoid duplicating
        observations between `observations` and `next_observations`.
        """
        all_obs = [t.observation for t in self.transitions]
        all_obs.append(self.transitions[-1].next_observation)
        return merge(all_obs)

    def rewards(self) -> torch.Tensor:
        """All rewards stacked together."""
        return torch.tensor([t.reward for t in self.transitions])

    def dones(self) -> torch.Tensor:
        """All the terminaison criterion."""
        return torch.tensor([t.done for t in self.transitions])

    def returns(
        self, discount: float = 1., normalize: bool = True
    ) -> torch.Tensor:
        """Compute the discounted returns of the trajectory."""
        raw_returns = func.returns(self.rewards(), discount)
        if normalize:
            return func.normalize_1d(raw_returns)
        else:
            return raw_returns


def create_explorer(
    env: Environment,
    select_action: Callable[[Engine, int], Action],
    store_trajectory: bool = False,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Engine:
    """Create an ignite engine to explore the environement.

    Parameters
    ----------
    env:
        The environement to explore.
    select_action:
        A function used to select an action. Has acess to the engine and the
        iteration number. The current observation is stored under
        `engine.state.observation`.
    store_trajectory:
        Whether or not to collect a `Trajectory` object every episode. This
        trajectory is not stored between episodes.
    dtype:
        Type to cast observations in.
    device:
        Device to move observations to.

    Returns
    -------
    explorer:
        An ignite Engine that will explore the environement according to the
        given policy. Handlers can be added by the user to perform different
        algorithms of reinforcement learning. Run with
        `engine.run(range(max_episode_length), n_episodes)`

    """
    def _process_func(engine, timestep):
        # Select action.
        action = select_action(engine, timestep)

        # Make action.
        next_observation, reward, done, infos = env.step(action)
        next_observation = utils.apply_to_type(
            next_observation,
            (np.ndarray, sp.spmatrix),
            utils.from_numpy_sparse
        ).to(dtype=dtype)

        # We create the transition object and store it.
        engine.state.transition = Transition(
            observation=engine.state.observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done
        )

        # Store timestep and info
        engine.state.episode_timestep = timestep
        engine.state.environement_info = infos

        # Store trajectory if necessary.
        if store_trajectory:
            engine.state.trajectory.append(
                **attr.asdict(engine.state.transition)
            )

        # Save for next move
        if device is not None and torch.device(device).type == "cuda":
            engine.state.observation = utils.apply_to_tensor(
                next_observation, lambda t: t.pin_memory())
        else:
            engine.state.observation = next_observation

        if done:  # Iteration events still fired.
            engine.terminate_epoch()

    explorer = Engine(_process_func)

    @explorer.on(Events.ITERATION_STARTED)
    def _move_to_device(engine):
        engine.state.observation = utils.apply_to_tensor(
            engine.state.observation,
            lambda t: t.to(device, non_blocking=True))

    @explorer.on(Events.EPOCH_STARTED)
    def _init_episode(engine):
        engine.state.observation = utils.apply_to_type(
            env.reset(),
            (np.ndarray, sp.spmatrix),
            utils.from_numpy_sparse
        ).to(dtype=dtype)

    @explorer.on(Events.COMPLETED)
    def _close(engine):
        env.close()

    if store_trajectory:
        @explorer.on(Events.EPOCH_STARTED)
        def _strore_trajectory(engine):
            engine.state.trajectory = Trajectory()

    return explorer

# coding: utf-8

"""Ignite exploration engine.

An engine that explore the environment according to a user define policy.
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

from typing import Callable, Optional, Generic

import attr
import torch
from ignite.engine import Engine, Events

from .environment import Observation, Action, Environment
from .data import Data


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Transition(
    Data, Generic[Observation, Action]
):
    """An observed Transition in the environment."""

    observation: Observation
    action: Action
    next_observation: Observation
    reward: float
    done: bool


class Explorer(Engine):
    """Environement Explorer.

    An ignite Engine that will explore the environment according to the
    given policy. Handlers can be added by the user to perform
    different algorithms of reinforcement learning. Run with
    `engine.run(range(max_episode_length), n_episodes)`.

    Exposes the following attributes in the state of the engine:
    episode_timestep:
        The time step in the current episode.
    transition:
        An object with all the information necessary for learning.
    environment_info:
        The extra information passed by the environment.
    observation:
        The current observation, moved accross devices.
    """

    def __init__(
        self,
        env: Environment,
        select_action: Callable[[Engine, Observation], Action],
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Engine:
        """Initialize an ignite engine to explore the environment.

        Parameters
        ----------
        env:
            The environment to explore.
        select_action:
            A function used to select an action. Has access to the engine and
            the iteration number. The current observation is stored under
            `engine.state.observation`. Takes as input the ingine and the
            iteration number, returns the action passed to the environement,
            along with a dictionary (possibly empty) of other variable to
            remember.
        dtype:
            Type to cast observations in.
        device:
            Device to move observations to before passing it to the
            `select_action` function.

        """
        def _maybe_pin(state):
            """Pin observation if necessary.

            Set the attribute `observation_dev` with a pinned version
            of observation if necessary.
            """
            if device is not None and torch.device(device).type == "cuda":
                state.observation_dev = state.observation.pin_memory()
            else:
                state.observation_dev = state.observation

        def _process_func(engine, timestep):
            """Take action on each iteeration."""
            # Store timestep for user
            engine.state.episode_timestep = timestep

            # Select action.
            action = select_action(engine, engine.state.observation_dev)

            # Make action.
            next_observation, reward, done, infos = env.step(action)
            next_observation = next_observation.to(dtype=dtype)

            # We create the transition object and store it.
            engine.state.transition = engine.state.TransitionClass(
                observation=engine.state.observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=done,
                **getattr(engine.state, "extra_transition_members", {})
            )
            # Cleaning to avoid exposing unecessary information
            if hasattr(engine.state, "extra_transition_members"):
                del engine.state.extra_transition_members

            # Store info for user
            engine.state.environment_info = infos

            # Save for next move
            # Observation on cpu (untouched)
            engine.state.observation = next_observation
            _maybe_pin(engine.state)

            if done:  # Iteration events still fired.
                engine.terminate_epoch()

        super().__init__(_process_func)

        @self.on(Events.STARTED)
        def _store_TransitionClass(engine):
            engine.state.TransitionClass = Transition

        @self.on(Events.ITERATION_STARTED)
        def _move_to_device(engine):
            engine.state.observation_dev = engine.state.observation_dev.to(
                device=device, non_blocking=True)

        @self.on(Events.EPOCH_STARTED)
        def _init_episode(engine):
            engine.state.observation = env.reset().to(dtype=dtype)
            _maybe_pin(engine.state)

        @self.on(Events.COMPLETED)
        def _close(engine):
            env.close()

    def register_transition_members(
        engine,
        *names: str,
        **name_attribs
    ) -> None:
        """Register extra members to be stored in the transition object.

        At every step, the method `store_transition_members` must be called
        to store the extra members.

        Parameters
        ----------
        names:
            Names of the parameters to register.
        name_attribs:
            Named parameters along with an `attr.ib` object (to have specific
            behaviour such as converters).

        """
        @engine.on(Events.STARTED)
        def _(engine):
            engine.state.TransitionClass = attr.make_class(
                "Transition",
                {**{n: attr.ib() for n in names}, **name_attribs},
                bases=(engine.state.TransitionClass, ),
                frozen=True,
                slots=True,
            )

    def store_transition_members(engine, **members) -> None:
        """Store transition members.

        Store the value of the transition members (registered with
        `register_transition_members`) to be added in the next transition.

        Parameters
        ----------
        members:
            Named parameters with the name of the register members, and its
            value.

        """
        engine.state.extra_transition_members = members

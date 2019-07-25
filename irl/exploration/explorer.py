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

from typing import Callable, Optional, Generic, Dict
import itertools

import attr
import torch
import ignite.metrics
from ignite.engine import Engine, Events, State

from .environment import Observation, Action, Environment
from .data import Data


@attr.s(auto_attribs=True, frozen=True, slots=True)
class Transition(Data, Generic[Observation, Action]):
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

    @staticmethod
    def _maybe_pin(obs: Observation, device: torch.device) -> Observation:
        """Return a pinned observation if necessary."""
        if device is not None and torch.device(device).type == "cuda":
            return obs.pin_memory()
        else:
            return obs

    def __init__(
        self,
        select_action: Callable[[Engine, Observation], Action],
        metrics: Optional[Dict[str, ignite.metrics.Metric]] = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> Engine:
        """Initialize an ignite engine to explore the environment.

        Parameters
        ----------
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

        def _process_func(engine, episode_timestep):
            """Take action on each iteration."""
            self.state.episode_timestep = episode_timestep
            # Select action.
            action = select_action(engine, engine.state.observation_dev)

            # Make action.
            next_observation, reward, done, infos = engine.state.env.step(action)
            next_observation = next_observation.to(dtype=dtype)

            # We create the transition object and store it.
            required_fields = [
                name
                for name, attrib in attr.fields_dict(
                    engine.state.TransitionClass
                ).items()
                if attrib.init
            ]
            engine.state.transition = engine.state.TransitionClass(
                observation=engine.state.observation,
                action=action,
                next_observation=next_observation,
                reward=reward,
                done=done,
                **getattr(engine.state, "extra_transition_members", {}),
                **{k: v for k, v in infos.items() if k in required_fields},
            )
            # Cleaning to avoid exposing unecessary information
            if hasattr(engine.state, "extra_transition_members"):
                del engine.state.extra_transition_members

            # Store info for user
            engine.state.environment_info = infos

            # Save for next move
            # Observation on cpu (untouched)
            engine.state.observation = next_observation
            # observation on device fo passing to select_action
            engine.state.observation_dev = self._maybe_pin(next_observation, device)

            if done:  # Iteration events still fired.
                engine.terminate_epoch()

            return engine.state.transition, engine.state.environment_info

        super().__init__(_process_func)

        @self.on(Events.STARTED)
        def _store_TransitionClass(engine):
            engine.state.TransitionClass = Transition

        @self.on(Events.ITERATION_STARTED)
        def _move_to_device(engine):
            engine.state.observation_dev = engine.state.observation_dev.to(
                device=device, non_blocking=True
            )

        @self.on(Events.EPOCH_STARTED)
        def _init_episode(engine):
            obs = engine.state.env.reset().to(dtype=dtype)
            engine.state.observation = obs
            engine.state.observation_dev = obs.to(device=device, non_blocking=True)

        if metrics is not None:
            for name, metric in metrics.items():
                metric.attach(self, name)

    def register_transition_members(engine, *names: str, **name_attribs) -> None:
        """Register extra members to be stored in the transition object.

        At every step, either the method `store_transition_members` must be
        called to store the extra members, or the member must be present in
        the environment info.

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
                bases=(engine.state.TransitionClass,),
                frozen=True,
                slots=True,
            )

    def store_transition_members(engine, **members) -> None:
        """Store transition members.

        Store the value of the transition members (registered with
        `register_transition_members`) to be added in the next transition.
        It is safe to call this function multiple times. The members stored are
        deleted after each iteration/timestep.

        Parameters
        ----------
        members:
            Named parameters with the name of the register members, and its
            value.

        """
        if hasattr(engine.state, "extra_transition_members"):
            engine.state.extra_transition_members.update(members)
        else:
            engine.state.extra_transition_members = members

    def run(
        self, env: Environment[Action, Observation], max_episodes: Optional[int] = None
    ) -> State:
        """Run the explorer.

        Parameters
        ----------
        env:
            The RL environement in which the agent interacts.
        max_episodes:
            Number of episode to run for.

        Return
        ------
        state:
            The state of the engine.

        """

        @self.on(Events.STARTED)
        def _save_env(engine):
            engine.state.env = env

        return super().run(
            itertools.count(),
            max_epochs=(float("inf") if max_episodes is None else max_episodes),
        )

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

from typing import Callable, Optional, Generic, Dict, Tuple

import attr
import torch
from ignite.engine import Engine, Events

import irl.utils as utils
from .environment import Observation, Action, Environment


@attr.s(auto_attribs=True, frozen=True)
class Transition(
    Generic[Observation, Action]
):
    """An observed Transition in the environment."""

    observation: Observation
    action: Action
    next_observation: Observation
    reward: float
    done: bool


def create_explorer(
    env: Environment,
    select_action: Callable[[Engine, int], Tuple[Action, Dict]],
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> Engine:
    """Create an ignite engine to explore the environment.

    Parameters
    ----------
    env:
        The environment to explore.
    select_action:
        A function used to select an action. Has access to the engine and the
        iteration number. The current observation is stored under
        `engine.state.observation`. Takes as input the ingine and the iteration
        number, returns the action passed to the environement, along with a
        dictionary (possibly empty) of other variable to remember.
    dtype:
        Type to cast observations in.
    device:
        Device to move observations to before passing it to the `select_action`
        function.

    Returns
    -------
    explorer:
        An ignite Engine that will explore the environment according to the
        given policy. Handlers can be added by the user to perform different
        algorithms of reinforcement learning. Run with
        `engine.run(range(max_episode_length), n_episodes)`

    """
    __Transition = None

    def _process_func(engine, timestep):
        # Select action.
        action, others = select_action(engine, timestep)

        # update transition class to contain others
        nonlocal __Transition
        if __Transition is None:
            __Transition = attr.make_class(
                "Transition",
                list(others.keys()),
                bases=(Transition, ),
                frozen=True
            )

        # Make action.
        next_observation, reward, done, infos = env.step(action)
        next_observation = utils.apply_to_tensor(
            next_observation, lambda t: t.to(dtype=dtype))

        # We create the transition object and store it.
        engine.state.transition = __Transition(
            observation=engine.state.observation,
            action=action,
            next_observation=next_observation,
            reward=reward,
            done=done,
            **others
        )

        # Store timestep and info
        engine.state.episode_timestep = timestep
        engine.state.environment_info = infos

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
            lambda t: t.to(device=device, non_blocking=True))

    @explorer.on(Events.EPOCH_STARTED)
    def _init_episode(engine):
        engine.state.observation = utils.apply_to_tensor(
            env.reset(), lambda t: t.to(dtype=dtype))

    @explorer.on(Events.COMPLETED)
    def _close(engine):
        env.close()

    return explorer

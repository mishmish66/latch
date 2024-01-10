import jax_dataclasses as jdc

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod

from typing import Tuple


@jdc.pytree_dataclass(kw_only=True)
class Env(ABC):
    action_bounds: jax.Array
    state_dim: jdc.Static[int]
    action_dim: jdc.Static[int]
    dt: jdc.Static[float]
    substep: jdc.Static[int]

    def step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Steps the environment forward one step.

        Args:
            state (jax.Array): a (s,) array of the current state.
            action (jax.Array): an (a,) array of the action to take.

        Returns:
            jax.Array: The next state.
        """

        return self.dense_step(state, action)[-1]

    def random_action(self, key: jax.Array) -> jax.Array:
        """Samples a random action from the action space.

        Args:
            key (jax.Array): A random key.

        Returns:
            jax.Array: An (a,) array of the action.
        """

        rng, key = jax.random.split(key)
        rands = jax.random.uniform(rng, shape=(self.action_dim,))
        action_range_sizes = self.action_bounds[:, 1] - self.action_bounds[:, 0]
        action_mins = self.action_bounds[:, 0]
        random_actions = rands * action_range_sizes + action_mins

        return random_actions

    @abstractmethod
    def dense_step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Steps the environment forward one step and returns the intermediate states.

        Args:
            state (jax.Array): a (s,) array of the current state.
            action (jax.Array): an (a,) array of the action to take.

        Returns:
            jax.Array: An (substep x s) array of the intermediate states.
        """
        pass

    @abstractmethod
    def reset(self) -> jax.Array:
        """Initializes the environment.

        Returns:
            array: An (s,) array of the initial state.
        """
        pass

    @abstractmethod
    def send_wandb_video(self, name, states, step, dense=True):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            states (array): A (t x s) array of the states.
            step (int): The current training step.
            dense (bool, optional): Whether the states passed are dense or substep strided states. Defaults to True.
        """
        pass

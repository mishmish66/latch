from env.env_config import EnvConfig

import jax
import jax.numpy as jnp

from abc import ABC, abstractmethod


class Env(ABC):
    @classmethod
    @abstractmethod
    def step(cls, state, action, env_config=None):
        """Steps the environment forward one step.

        Args:
            state (array): a (s,) array of the current state.
            action (array): an (a,) array of the action to take.
            env_config (EnvConfig): The config to use to step the environment.

        Returns:
            (array, array): A tuple of an (s,) array of the next state and a (u x s) array containing the unstrided dense states.
        """
        pass

    @classmethod
    @abstractmethod
    def init(cls):
        """Initializes the environment.

        Returns:
            Env: The initialized environment.
        """
        pass

    @classmethod
    @abstractmethod
    def get_config(cls):
        """Gets the default configuration of the environment.

        Returns:
            EnvConfig: The default configuration of the environment.
        """
        pass

    @classmethod
    @abstractmethod
    def send_wandb_video(cls, name, states, env_config: EnvConfig, step, dense=True):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            states (array): A (t x s) array of the states.
            env_config (EnvConfig): The configuration of the environment.
            step (int): The current training step.
            dense (bool, optional): Whether the states passed are dense or substep strided states. Defaults to True.
        """
        pass

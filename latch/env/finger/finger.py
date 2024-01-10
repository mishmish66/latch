from latch.rendering import JAXRenderer

from ..env import Env

import jax_dataclasses as jdc

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jnp
from jax.experimental import io_callback
from jax.experimental.host_callback import id_tap
from jax.tree_util import Partial, register_pytree_node_class

from overrides import override

import wandb

import numpy as np

from pathlib import Path

from typing import Optional, Any


@jdc.pytree_dataclass(kw_only=True)
class Finger(Env):
    """The finger environment."""

    _host_model: jdc.Static[mujoco.MjModel]  # type: ignore
    _model: mjx.Model
    _renderer: jdc.Static[JAXRenderer]

    @classmethod
    def init(
        cls,
        action_bounds: Optional[jax.Array] = None,
        dt: Optional[float] = None,  # type: ignore
        substep: int = 32,
    ):
        # Find the xml file
        this_file_path = Path(__file__).resolve()
        xml_path = this_file_path.parent / "assets" / "scene.xml"

        # Load the model and renderer
        host_model = mujoco.MjModel.from_xml_path(str(xml_path))  # type: ignore

        # Compute the configuration
        if action_bounds is None:
            action_bounds = jnp.array(host_model.actuator_ctrlrange)

        state_dim = host_model.nq + host_model.nv

        action_dim = host_model.nu

        if dt is None:
            dt = host_model.opt.timestep * substep

        # Update model
        substep_dt = dt / substep  # type: ignore
        host_model.opt.timestep = substep_dt
        host_model = host_model
        model = mjx.put_model(host_model)
        renderer = JAXRenderer(host_model, 512, 512)

        return cls(
            action_bounds=action_bounds,
            state_dim=state_dim,
            action_dim=action_dim,
            dt=dt,  # type: ignore
            substep=substep,
            _host_model=host_model,
            _model=model,
            _renderer=renderer,
        )

    @override
    def dense_step(self, state: jax.Array, action: jax.Array) -> jax.Array:
        """Steps the environment forward one step.

        Args:
            state (array): An (s,) array of the current state.
            action (array): An (a,) array of the action to take.

        Returns:
            (array): A (substep x s) array of the states between substeps.
        """

        # Make the data
        data = mjx.make_data(self._model)

        # TODO: See if we still need this
        # Filter out nans in the action
        action_is_nan = jnp.isnan(action)
        ctrl = jnp.where(action_is_nan, data.ctrl, action)

        # Set the model state
        qpos = state[: self._model.nq]  # type: ignore
        qvel = state[self._model.nq :]  # type: ignore

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        def scanf(data, _):
            data = data.replace(ctrl=ctrl)
            next_data = mjx.step(self._model, data)
            next_state = self._data_to_obs(data)
            return next_data, next_state

        _, states = jax.lax.scan(
            scanf,
            data,
            xs=None,
            length=self.substep,
        )

        return states

    @staticmethod
    def _data_to_obs(data):
        """Makes a state from the MJ(X)Data.

        Args:
            data (MJXData): _description_

        Returns:
            array: An (s,) array containing the observation.
        """
        return jnp.concatenate([data.qpos, data.qvel], dtype=jnp.float32)

    @override
    def reset(self) -> jax.Array:
        """Initializes the environment state array.

        Returns:
            array: An (s,) array of the state.
        """
        data = mjx.make_data(self._model)
        data = data.replace(qpos=jnp.array([0, 0, 0]))

        return self._data_to_obs(data)

    def render(self, state):
        data = mjx.make_data(self._model)

        nq = self._model.nq

        data = data.replace(qpos=state[:nq])
        data = data.replace(qvel=state[nq:])

        data = mjx.forward(self._model, data)

        img = self._renderer.render(data, "topdown").transpose(2, 0, 1)

        return img

    @Partial(jax.jit, static_argnames=["fps", "dense"])
    def make_video(self, states: jax.Array, fps: int = 24, dense=True):
        """Makes a video from the states.

        Args:
            states (array): A (t x s) array of the states.
            fps (int): The approximate fps to decide which states to render of the video.
            dense (bool, optional): Whether the states passed are dense or substep strided states. Defaults to True.

        Returns:
            array: A (t x 3 x 512 x 512) array of the video.
        """

        # Calculate the state array stride taking into account the substeps
        states_dt = self.dt / self.substep if dense else self.dt
        stride = 1 / fps / states_dt

        # Approximate the fps
        if stride < 1:
            # If the stride is less than 1, then we will raise it to 1 and set the fps as high as possible
            stride = 1
        else:
            # Otherwise, we will round the stride to the nearest integer and set the fps to that
            stride = int(stride)

        frame_states = states[::stride]

        video_frames = jax.vmap(self.render)(frame_states)
        return video_frames

    def host_send_video_array_to_wandb(
        self,
        name: str,
        frames: np.ndarray,
        step: int,
        fps: int,
    ):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            frames (array): An (f x 3 x width x height) array containing the video frames.
            step (int): The current training step.
            fps (int): The fps of the video.
        """

        print(f"Sending 🕊️ \"{name}\" for step {step}")  # fmt: skip

        wandb.log({name: wandb.Video(frames, fps=fps)}, step=step)

    # Jit decorator here is necessary to avoid the string being interpreted as a jax type
    # and jax can't handle the string type so that causes a crash.
    @Partial(jax.jit, static_argnames=["name", "fps", "dense"])
    def send_wandb_video(
        self,
        name: str,
        states: jax.Array,
        step: int,
        fps: int = 24,
        dense: bool = True,
    ):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            states (array): A (t x s) array of the states.
            env_config (EnvConfig): The configuration of the environment.
            step (int): The current training step.
            dense (bool, optional): Whether the states passed are dense or substep strided states. Defaults to True.
        """

        # Render the video
        video_array = self.make_video(states=states, dense=dense)

        # This wrapper is necessary to avoid the string being interpreted as a jax type
        def host_send_video_wrapper(frames: np.ndarray, step):
            self.host_send_video_array_to_wandb(
                name=name, frames=frames, step=step, fps=fps
            )

        def send_wandb_video_for_tap(tap_pack, transforms):
            frames, step = tap_pack
            host_send_video_wrapper(frames=frames, step=step)

        tap_pack = (video_array, step)
        id_tap(send_wandb_video_for_tap, tap_pack)

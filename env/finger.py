# Before anything else is imported, set the environment variable to use the right rendering backend
# EGL works on a headless server but it crashes when called from a JAX callback.
# OSMESA uses the CPU to render which is slow but it works in a JAX callback and since we use id_tap it doesn't block anyway.
import os

os.environ["MUJOCO_GL"] = "OSMESA"

from learning.train_state import TrainState
from env.env_config import EnvConfig
from env.env import Env

import mujoco
from mujoco import mjx

import jax
from jax import numpy as jnp
from jax.experimental import io_callback
from jax.experimental.host_callback import id_tap
from jax.tree_util import Partial

import wandb

import numpy as np

from dataclasses import dataclass


@dataclass
class Finger(Env):
    @classmethod
    def class_init(cls):
        cls.host_model = mujoco.MjModel.from_xml_path("assets/finger/scene.xml")
        cls.model = mjx.device_put(cls.host_model)
        cls.renderer = mujoco.Renderer(cls.host_model, 512, 512)

    @classmethod
    def step(cls, state, action, env_config: EnvConfig = None):
        """Steps the environment forward one step.

        Args:
            state (array): An (s,) array of the current state.
            action (array): An (a,) array of the action to take.
            env_config (EnvConfig, optional): The configuration of the environment. Defaults to None.

        Returns:
            (array, array): The next state and a (substep x s) array of the states between substeps.
        """

        if env_config is None:
            env_config = cls.get_config()

        # Configure the model to the env_config
        substep_dt = env_config.dt / env_config.substep
        model = cls.model
        temp_opt = model.opt.replace(timestep=substep_dt)
        model = model.replace(opt=temp_opt)

        # Make the data
        data = mjx.make_data(model)

        # Filter out nans in the action
        nan_action_elems = jnp.isnan(action)
        ctrl = jnp.where(nan_action_elems, data.ctrl, action)

        # Set the model state
        qpos = state[: model.nq]
        qvel = state[model.nq :]

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=ctrl)

        def scanf(data, _):
            data = data.replace(ctrl=ctrl)
            next_data = mjx.step(model, data)
            next_state = cls.make_state(data)
            return next_data, next_state

        _, states = jax.lax.scan(
            scanf,
            data,
            xs=None,
            length=env_config.substep,
        )

        return states[-1], states

    @classmethod
    def make_state(cls, data):
        """Makes a state from the MJ(X)Data.

        Args:
            data (MJXData): _description_

        Returns:
            array: An (s,) array of the state.
        """
        return jnp.concatenate([data.qpos, data.qvel], dtype=jnp.float32)

    @classmethod
    def init(cls):
        """Initializes the environment state array.

        Returns:
            array: An (s,) array of the state.
        """
        data = mjx.make_data(cls.model)
        data = data.replace(qpos=jnp.array([0, 0, 0]))

        return cls.make_state(data)

    @classmethod
    def get_config(cls, substep=32):
        """Makes a configuration of the environment.

        Args:
            substep (int, optional): How many substeps to use between states and actions. Defaults to 32.

        Returns:
            EnvConfig: The generated configuration of the environment.
        """

        return EnvConfig(
            action_bounds=jnp.array(cls.model.actuator_ctrlrange),
            state_dim=cls.model.nq + cls.model.nv,
            act_dim=cls.model.nu,
            dt=cls.model.opt.timestep * substep,
            substep=substep,
        )

    @classmethod
    def host_render_frame(cls, state):
        host_data = mujoco.MjData(cls.host_model)

        nq = int(cls.host_model.nq)

        host_data.qpos[:] = state[:nq]
        host_data.qvel[:] = state[nq:]

        mujoco.mj_forward(cls.host_model, host_data)

        cls.renderer.update_scene(host_data, "topdown")
        img = cls.renderer.render()

        return img

    @classmethod
    def host_make_video(cls, states, env_config: EnvConfig, fps=24, dense=True):
        """Makes a video from the states.

        Args:
            states (array): A (t x s) array of the states.
            env_config (EnvConfig): The configuration of the environment.
            fps (int, optional): The fps of the video. Defaults to 24.
            dense (bool, optional): Whether the states passed are dense or substep struded states. Defaults to True.

        Returns:
            array: A (t x 3 x 512 x 512) array of the video.
        """

        # Calculate the state array stride taking into account the substeps
        states_dt = env_config.dt / env_config.substep if dense else env_config.dt
        stride = 1 / fps / states_dt

        # Approximate the fps
        if stride < 1:
            # If the stride is less than 1, then we will raise it to 1 and set the fps as high as possible
            stride = 1
            fps = 1 / states_dt
        else:
            # Otherwise, we will round the stride to the nearest integer and set the fps to that
            stride = int(stride)
            fps = 1 / states_dt / stride

        frames = []
        next_state_i = 0
        while next_state_i < states.shape[0]:
            frames.append(cls.host_render_frame(states[next_state_i]))
            next_state_i += stride

        vid_arr = np.stack(frames).transpose(0, 3, 1, 2)
        return vid_arr

    @classmethod
    def host_send_wandb_video(
        cls, name, states, env_config: EnvConfig, step, dense=True
    ):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            states (array): A (t x s) array of the states.
            env_config (EnvConfig): The configuration of the environment.
            step (int): The current training step.
            dense (bool, optional): Whether the states passed are dense or substep struded states. Defaults to True.
        """

        print(f"Sending 📨 \"{name}\" for step {step}")  # fmt: skip

        fps = 24
        video_array = cls.host_make_video(states, env_config, fps, dense)

        wandb.log({name: wandb.Video(video_array, fps=fps)}, step=step)

        # print(f"Sent 🕊️ \"{name}\"")  # fmt: skip

    # Jit decorator here is necessary to avoid the string being interpreted as a jax type
    # and jax can't handle the string type so that causes a crash.
    @Partial(jax.jit, static_argnames=["name", "dense"])
    def send_wandb_video(name, states, env_config: EnvConfig, step, dense=True):
        """Sends a video to wandb.

        Args:
            name (str): The name of the video.
            states (array): A (t x s) array of the states.
            env_config (EnvConfig): The configuration of the environment.
            step (int): The current training step.
            dense (bool, optional): Whether the states passed are dense or substep strided states. Defaults to True.
        """

        # This wrapper is necessary to avoid the string being interpreted as a jax type
        def host_send_video_wrapper(states, env_config, step, dense):
            Finger.host_send_wandb_video(
                name=name, states=states, env_config=env_config, step=step, dense=dense
            )

        def send_wandb_video_for_tap(tap_pack, transforms):
            states, env_config, step, dense = tap_pack
            host_send_video_wrapper(
                states=states,
                env_config=env_config,
                step=step,
                dense=dense,
            )

        tap_pack = (states, env_config, step, dense)
        id_tap(send_wandb_video_for_tap, tap_pack)


Finger.class_init()

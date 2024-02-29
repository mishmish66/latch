import os
import shutil
from latch.config import TrainConfig
from pathlib import Path

from .train_rollout import train_rollout

import hydra
import jax
import jax_dataclasses as jdc
import orbax.checkpoint as ocp
import wandb
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap
from jax.tree_util import Partial, register_pytree_node_class
from omegaconf import DictConfig, OmegaConf

from latch import Infos, LatchState
from latch.policy import ActorPolicy
from latch.rollout import eval_actor

from dataclasses import dataclass


class Trainer:
    def __init__(self, train_config: TrainConfig):
        self.checkpoint_dir = Path(train_config.checkpoint_dir)
        self.checkpoint_count = train_config.checkpoint_count
        self.save_every = train_config.save_every
        self.eval_every = train_config.eval_every
        self.use_wandb = train_config.use_wandb
        self.checkpointer = ocp.PyTreeCheckpointer()
        self.checkpoint_paths = []

        wandb.init(
            project="latch",
            # name="latch",
            config=OmegaConf.to_container(train_config),  # type: ignore
            dir=self.checkpoint_dir,
            mode="disabled" if not self.use_wandb else "online",
        )

    def host_save_model(self, train_state):
        """This is a callback that runs on the host and saves the model to disk."""
        print("Saving 💾 Network")
        checkpoint_path = (
            self.checkpoint_dir
            / f"checkpoint_r{train_state.rollout}_s{train_state.step}"
        )
        self.checkpointer.save(
            checkpoint_path.absolute(),
            train_state,
        )
        # Save it as a zip file with {checkpoint_path}.zip
        shutil.make_archive(str(checkpoint_path), "zip", checkpoint_path)
        # Delete the orbax folder
        shutil.rmtree(checkpoint_path)
        # Overwrite the latest checkpoint
        latest_checkpoint_path = self.checkpoint_dir / "checkpoint_latest.zip"
        shutil.copyfile(
            f"{checkpoint_path}.zip",
            latest_checkpoint_path,
        )
        # Update the file in wandb
        wandb.save(str(latest_checkpoint_path))

        # Queue the new checkpoint path at the front of the list
        self.checkpoint_paths.insert(0, checkpoint_path)
        # Drop the oldest checkpoint if there are more than checkpoint_count
        if len(self.checkpoint_paths) > self.checkpoint_count:
            # Get the oldest checkpoint
            oldest_checkpoint_path = self.checkpoint_paths.pop()
            # Delete the oldest checkpoint
            os.remove(f"{oldest_checkpoint_path}.zip")

    @Partial(jax.jit, static_argnames=["self"])
    def save_model(self, train_state: LatchState):
        """This is the jax wrapper for the checkpointing callback."""

        def save_model_for_tap(train_state: LatchState, transforms):
            self.host_save_model(train_state)

        id_tap(save_model_for_tap, train_state)

    @Partial(jax.jit, static_argnames=["self"])
    def eval_model(self, train_state: LatchState):
        """This evaluates the model and logs the results to wandb."""
        jax.debug.print("Evaluating 🧐 Network")

        state_target = jnp.zeros(train_state.config.state_dim)
        state_weights = jnp.zeros_like(state_target)

        state_target = state_target.at[0].set(jnp.pi * 2)
        state_weights = state_weights.at[0].set(1.0)
        policy = ActorPolicy(state_target=state_target, state_weights=state_weights)

        key, train_state = train_state.split_key()

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 32)
        _, eval_infos, dense_states = jax.vmap(
            Partial(
                eval_actor,
                start_state=train_state.config.env.reset(),
                train_state=train_state,
                policy=policy,
            )
        )(key=rngs)

        eval_infos: Infos = eval_infos.condense(method="mean") # TODO: Change this back to unstack
        infos = Infos().add_info("eval_infos", eval_infos)
        infos.dump_to_wandb(train_state.step)
        train_state.config.env.send_wandb_video(
            name="Actor Video",
            states=dense_states[0],
            step=train_state.step,
        )

    @Partial(jax.jit, static_argnames=["self"])
    def train_loop_body(self, train_state: LatchState):
        """Trains for a single rollout and applies callbacks"""
        rollout_index = train_state.rollout

        def print_rollout_msg_for_tap(tap_pack, transforms):
            """Dumps a fun message to the console."""
            i = tap_pack
            print(f"Rollout 🛺 {i}")

        id_tap(print_rollout_msg_for_tap, rollout_index)

        save_this_time = rollout_index % self.save_every == 0
        jax.lax.cond(
            save_this_time,
            self.save_model,
            lambda train_state: None,
            train_state,
        )

        train_state, eval_state = train_state.split_state()

        eval_this_time = rollout_index % self.eval_every == 0
        jax.lax.cond(
            eval_this_time,
            self.eval_model,
            lambda train_state: None,
            eval_state,
        )

        train_state = train_rollout(train_state)

        return train_state

    def train(self, train_state: LatchState):
        """Runs the training loop."""

        # Log that we're starting training
        print("Starting Training Loop 🤓")

        trained_state = jax.lax.while_loop(
            lambda train_state: ~train_state.is_done(),
            self.train_loop_body,
            train_state,
        )

        return trained_state

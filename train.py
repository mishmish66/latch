from env.finger import Finger

from learning.eval_actor import eval_batch_actor

from learning.train_state import TrainConfig, TrainState

from learning.training.train_rollout import train_rollout

from nets.nets import (
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
)

import orbax.checkpoint as ocp
import optax

import jax
from jax import numpy as jnp
from jax.experimental.host_callback import id_tap

import wandb

import shutil
from pathlib import Path
import os
import sys
import time

import argparse

seed = 0

# Generate random key
key = jax.random.PRNGKey(seed)

checkpointer = ocp.PyTreeCheckpointer()

# Save a list of the most recent checkpoints
checkpoint_paths = []
checkpoint_count = 3

# Set up the training config
learning_rate = float(5e-3)
every_k = 1

# Set the environment class
env_cls = Finger

# Grab the default environment config from the env class
env_config = env_cls.get_config()

# Set the latent state and action dimensions (here I've just set them the same as the state and action dims)
latent_state_dim = 6
latent_action_dim = 2

# Actually create the training config
train_config = TrainConfig.init(
    learning_rate=learning_rate,
    # Make the optimizer
    optimizer=optax.chain(
        optax.zero_nans(),
        optax.adamw(
            learning_rate=optax.cosine_onecycle_schedule(
                transition_steps=8192,
                peak_value=learning_rate,
                pct_start=0.3,
                div_factor=25.0,
                final_div_factor=1.0,
            ),
        ),
    ),
    # Instantiate all of the networks
    state_encoder=StateEncoder(latent_state_dim=latent_state_dim),
    action_encoder=ActionEncoder(latent_action_dim=latent_action_dim),
    transition_model=TransitionModel(
        latent_state_dim=latent_state_dim, n_layers=8, latent_dim=64, heads=4
    ),
    state_decoder=StateDecoder(state_dim=env_config.state_dim),
    action_decoder=ActionDecoder(act_dim=env_config.act_dim),
    latent_state_dim=latent_state_dim,
    latent_action_dim=latent_action_dim,
    env_config=env_config,
    env_cls=env_cls,
    seed=seed,
    target_net_tau=0.05,
    transition_factor=10.0,
    rollouts=256,
    epochs=64,
    batch_size=64,
    every_k=every_k,
    traj_per_rollout=1024,
    rollout_length=64,
    state_radius=1.6,
    action_radius=2.0,
    reconstruction_weight=1.0,
    forward_weight=1.0,
    smoothness_weight=1.0,
    condensation_weight=1.0,
    dispersion_weight=10.0,
    forward_gate_sharpness=1,
    smoothness_gate_sharpness=1,
    dispersion_gate_sharpness=1,
    condensation_gate_sharpness=1,
    forward_gate_center=-6,
    smoothness_gate_center=-3,
    dispersion_gate_center=-3,
    condensation_gate_center=-6,
)

# Create the train state that contains all of the network and optimizer parameters
rng, key = jax.random.split(key)
train_state = TrainState.init(rng, train_config)

# Handle resume arg
parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument(
    "--resume",
    action="store_true",
    default=False,
    help="Resume training from the latest checkpoint",
)
args = parser.parse_args()

# Initialize wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="Latch",
    config=train_config.make_dict(),
    resume=True if args.resume else "allow",
)

# Set a dir based on the wandb run id
checkpoint_dir = Path(f"checkpoints/checkpoints_{wandb.run.id}")


def host_save_model(key, train_state, i):
    """This is a callback that runs on the host and saves the model to disk."""
    print("Saving 💾 Network")
    checkpoint_path = checkpoint_dir / f"checkpoint_r{i}_s{train_state.step}"
    checkpointer.save(
        checkpoint_path.absolute(),
        train_state,
    )
    # Save it as a zip file with {checkpoint_path}.zip
    shutil.make_archive(checkpoint_path, "zip", checkpoint_path)
    # Delete the orbax folder
    shutil.rmtree(checkpoint_path)
    # Overwrite the latest checkpoint
    latest_checkpoint_path = checkpoint_dir / "checkpoint_latest.zip"
    shutil.copyfile(
        f"{checkpoint_path}.zip",
        latest_checkpoint_path,
    )
    # Update the file in wandb
    wandb.save(str(latest_checkpoint_path))

    # Queue the new checkpoint path at the front of the list
    checkpoint_paths.insert(0, checkpoint_path)
    # Drop the oldest checkpoint if there are more than checkpoint_count
    if len(checkpoint_paths) > checkpoint_count:
        # Get the oldest checkpoint
        oldest_checkpoint_path = checkpoint_paths.pop()
        # Delete the oldest checkpoint
        os.remove(f"{oldest_checkpoint_path}.zip")


def save_model(key, train_state, i):
    """This is the jax wrapper for the checkpointing callback."""

    def save_model_for_tap(tap_pack, transforms):
        key, train_state, i = tap_pack
        host_save_model(key, train_state, i)

    id_tap(save_model_for_tap, (key, train_state, i))


def eval_model(key, train_state, i):
    """This evaluates the model and logs the results to wandb."""
    jax.debug.print("Evaluating 🧐 Network")
    rng, key = jax.random.split(key)
    _, infos, dense_states = eval_batch_actor(
        key=rng,
        start_state=env_cls.init(),
        net_state=train_state.target_net_state,
        train_config=train_state.train_config,
    )

    infos.dump_to_wandb(train_state)
    env_cls.send_wandb_video(
        name="Actor Video",
        states=dense_states[0],
        env_config=env_config,
        step=train_state.step,
    )


def print_rollout_msg_for_tap(tap_pack, transforms):
    """Dumps a fun message to the console."""
    i = tap_pack
    print(f"Rollout 🛺 {i}")


# Here we restore the latest checkpoint if we're resuming
if wandb.run.resumed:
    checkpoint = wandb.restore(
        str(checkpoint_dir / "checkpoint_latest.zip"),
        replace=True,
        root=checkpoint_dir,
    )
    shutil.unpack_archive(
        checkpoint.name,
        checkpoint_dir / "checkpoint_latest",
    )

    train_state = checkpointer.restore(
        (checkpoint_dir / "checkpoint_latest").absolute(),
        item=train_state,
    )

# Log that we're starting training
print("Starting Training Loop 🤓")

# Set the save and eval intervals
save_every = 1
eval_every = 1


# Define the body of the training loop
def train_loop(train_state):
    """Trains for a single rollout and applies callbacks"""
    i = train_state.rollout
    key, train_state = train_state.split_key()

    id_tap(print_rollout_msg_for_tap, i)

    save_this_time = i % save_every == 0
    jax.lax.cond(
        save_this_time,
        save_model,
        lambda key, train_state, i: None,
        *(key, train_state, i),
    )

    eval_this_time = i % eval_every == 0
    jax.lax.cond(
        eval_this_time,
        eval_model,
        lambda key, train_state, i: None,
        *(key, train_state, i),
    )

    train_state = train_rollout(train_state)

    return train_state


def train_cond(train_state: TrainState):
    """Checks if we're done training"""
    # TODO: actually implement every_k
    return ~train_state.is_done()


# Not sure if this is actually a good way to do this
# I feel like it seems crazy to use a jax.lax.while_loop
# for the whole training loop, but it's really fast
# It takes a while to compile though (especially on
# the unity nodes with their really slow server CPUs)

"""Launch the jax lax while loop that trains the networks"""
final_train_state = jax.lax.while_loop(train_cond, train_loop, train_state)

print("Finished Training 🎉")

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

seed = 0

# Generate random key
key = jax.random.PRNGKey(seed)
checkpoint_dir = Path("checkpoints")

checkpointer = ocp.PyTreeCheckpointer()

learning_rate = float(1e-4)
every_k = 1

env_cls = Finger

env_config = env_cls.get_config()

latent_state_dim = 6
latent_action_dim = 2

train_config = TrainConfig.init(
    learning_rate=learning_rate,
    optimizer=optax.chain(
        optax.zero_nans(),
        optax.lion(
            learning_rate=optax.cosine_onecycle_schedule(
                transition_steps=8192,
                peak_value=learning_rate,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=0.25,
            )
        ),
    ),
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
    rollouts=256,
    epochs=256,
    batch_size=32,
    every_k=every_k,
    traj_per_rollout=256,
    rollout_length=64,
    state_radius=1.25,
    action_radius=2.0,
    reconstruction_weight=1.0,
    forward_weight=1.0,
    smoothness_weight=1.0,
    condensation_weight=10.0,
    dispersion_weight=10.0,
    forward_gate_sharpness=1,
    smoothness_gate_sharpness=1,
    dispersion_gate_sharpness=1,
    condensation_gate_sharpness=1,
    forward_gate_center=-16,
    smoothness_gate_center=-9,
    dispersion_gate_center=-9,
    condensation_gate_center=-16,
)

rng, key = jax.random.split(key)
train_state = TrainState.init(rng, train_config)

wandb.init(
    # set the wandb project where this run will be logged
    project="Latch",
    config=train_config.make_dict(),
)

# Check if dir exists
if os.path.exists(checkpoint_dir):
    # If it exists wait 3 seconds and then delete it (iterate counter in console for 3 seconds)
    for i in [3, 2, 1, 0]:
        print(f"⏲️ Preparing to delete old checkpoints in {i} second(s)...", end="\r")
        sys.stdout.flush()
        time.sleep(1)
    print("\n🧹 Clearing old checkpoints...")

    shutil.rmtree(checkpoint_dir)


def host_save_model(key, train_state, i):
    checkpoint_path = checkpoint_dir / f"checkpoint_r{i}_s{train_state.step}"
    checkpointer.save(
        checkpoint_path.absolute(),
        train_state,
    )
    wandb.save(str(checkpoint_dir / "checkpoint_r*"), base_path=str(checkpoint_dir))


def save_model(key, train_state, i):
    def save_model_for_tap(tap_pack, transforms):
        key, train_state, i = tap_pack
        host_save_model(key, train_state, i)

    id_tap(save_model_for_tap, (key, train_state, i))


def eval_model(key, train_state, i):
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


def save_and_eval_model(key, train_state, i):
    """Saves the model and evaluates it."""
    jax.debug.print("Saving 💾 and Evaluating 🧐 Network")

    save_model(key, train_state, i)
    eval_model(key, train_state, i)


def print_rollout_msg_for_tap(tap_pack, transforms):
    i = tap_pack
    print(f"Rollout 🛺 {i}")


print("Starting Training Loop 🤓")

save_and_eval_every = 4


# @profile
def train_loop(train_state, x_pack):
    i, key = x_pack

    id_tap(print_rollout_msg_for_tap, i)

    is_every = i % save_and_eval_every == 0
    jax.lax.cond(
        is_every,
        save_and_eval_model,
        lambda key, train_state, i: None,
        *(key, train_state, i),
    )

    rng, key = jax.random.split(key)
    train_state = train_rollout(
        key=key,
        train_state=train_state,
    )

    return train_state, None


xs = (jnp.arange(train_config.rollouts), jax.random.split(key, train_config.rollouts))

final_train_state, _ = jax.lax.scan(train_loop, train_state, xs)

print("Finished Training 🎉")

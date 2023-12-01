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

import wandb

import shutil
from pathlib import Path
import os
import time

# # Profiling stuff
# from pympler import muppy
# from memory_profiler import profile

# ##############################

seed = 0

# Generate random key
key = jax.random.PRNGKey(seed)
checkpoint_dir = Path("checkpoints")

checkpointer = ocp.PyTreeCheckpointer()

learning_rate = float(1.0e-4)
every_k = 1

env_cls = Finger

env_config = env_cls.get_config()

schedule = learning_rate  # optax.cosine_onecycle_schedule(
#     transition_steps=4096,
#     peak_value=learning_rate,
#     pct_start=0.125,
#     div_factor=5.0,
#     final_div_factor=5.0,
# )

latent_state_dim = 6
latent_action_dim = 2

train_config = TrainConfig.init(
    learning_rate=learning_rate,
    optimizer=optax.MultiSteps(
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1e12),
            optax.lion(learning_rate=learning_rate),
        ),
        every_k_schedule=every_k,
    ),
    state_encoder=StateEncoder(latent_state_dim=latent_state_dim),
    action_encoder=ActionEncoder(latent_action_dim=latent_action_dim),
    transition_model=TransitionModel(
        latent_state_dim=latent_state_dim, n_layers=3, latent_dim=64, heads=4
    ),
    state_decoder=StateDecoder(state_dim=env_config.state_dim),
    action_decoder=ActionDecoder(act_dim=env_config.act_dim),
    latent_state_dim=latent_state_dim,
    latent_action_dim=latent_action_dim,
    env_config=env_config,
    env_cls=env_cls,
    seed=seed,
    rollouts=64,
    epochs=128,
    batch_size=32,
    every_k=every_k,
    traj_per_rollout=256,
    rollout_length=64,
    state_radius=1.375,
    action_radius=2.0,
    reconstruction_weight=1.0,
    forward_weight=1.0,
    smoothness_weight=1.0,
    condensation_weight=1.0,
    dispersion_weight=1.0,
    forward_gate_sharpness=256,
    smoothness_gate_sharpness=256,
    dispersion_gate_sharpness=1,
    condensation_gate_sharpness=1,
    forward_gate_center=0,
    smoothness_gate_center=0,
    dispersion_gate_center=-9,
    condensation_gate_center=-9,
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
    for i in range(3, 0, -1):
        print(f"🚀 Preparing to delete old checkpoints in {i} second(s)...", end=None)
        time.sleep(1)
        print("\r", end=None)
    print("\n🧹 Clearing old checkpoints...")

    shutil.rmtree(checkpoint_dir)


def save_and_eval(key, train_state, i):
    checkpoint_path = checkpoint_dir / f"checkpoint_r{i}_s{train_state.step}"
    checkpointer.save(
        checkpoint_path.absolute(),
        train_state,
    )
    wandb.save(str(checkpoint_dir / "checkpoint_r*"), base_path=str(checkpoint_dir))

    _, infos, dense_states = eval_batch_actor(
        key=key,
        start_state=env_cls.init(),
        train_state=train_state,
    )

    infos.dump_to_wandb(train_state)
    env_cls.send_wandb_video(
        name="Actor Video",
        states=dense_states[0],
        env_config=env_config,
        step=train_state.step,
    )


print("Starting Train Loop 🤓")

save_and_eval_every = 4


@profile
def train_loop(train_state, x_pack):
    muppy.print_summary()

    i, key = x_pack

    if i % save_and_eval_every == 0:
        print("Saving 💾 and Evaluating 🧐 Network")
        # save_and_eval(key, train_state, i)

    rng, key = jax.random.split(key)
    train_state = train_rollout(
        key=key,
        train_state=train_state,
    )

    return train_state, None


xs = (jnp.arange(train_config.rollouts), jax.random.split(key, train_config.rollouts))

for x_pack in zip(*xs):
    train_state, _ = train_loop(train_state, x_pack)

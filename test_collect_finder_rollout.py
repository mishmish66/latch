from env.finger import Finger

from learning.eval_actor import eval_batch_actor

from env.rollout import collect_rollout_batch, collect_single_rollout

from policy.finder_policy import FinderPolicy

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
from jax.tree_util import Partial

import wandb

import shutil

from pathlib import Path

seed = 0

# Generate random key
key = jax.random.PRNGKey(seed)
checkpoint_dir = Path("checkpoints")

checkpointer = ocp.PyTreeCheckpointer()

learning_rate = float(1.0e-5)
every_k = 1

env_cls = Finger

env_config = env_cls.get_config()

schedule = optax.cosine_onecycle_schedule(
    transition_steps=2048,
    peak_value=learning_rate,
    pct_start=0.125,
    div_factor=5.0,
    final_div_factor=1.0,
)

latent_state_dim = 6
latent_action_dim = 2

train_config = TrainConfig.init(
    learning_rate=learning_rate,
    optimizer=optax.MultiSteps(
        optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(500),
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
    rollouts=4,
    epochs=128,
    batch_size=64,
    every_k=every_k,
    traj_per_rollout=256,
    rollout_length=64,
    state_radius=1.375,
    action_radius=2.0,
    reconstruction_weight=10.0,
    forward_weight=10.0,
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
    mode="dryrun",
)

# Let's try just a single trajectory

policy = FinderPolicy.init()
aux = FinderPolicy.make_aux(jnp.zeros(6))

rng, key = jax.random.split(key)
(states, actions), infos, dense_states = collect_single_rollout(
    rng, env_cls.init(), policy, aux, train_state
)

print(f"single traj states_mean = {jnp.mean(states):.3f}")

# Let's try a batch of trajectories
policy = FinderPolicy.init()
policy_aux = jax.vmap(FinderPolicy.make_aux)(
    target_state=jnp.zeros([train_state.train_config.traj_per_rollout, 6])
)
rng, key = jax.random.split(key)
collect_rollout_batch(
    rng,
    env_cls.init(),
    policy,
    policy_aux,
    train_state,
    train_state.train_config.traj_per_rollout,
)

# rngs = jax.random.split(rng, train_state.train_config.traj_per_rollout)
# (states, actions), infos, dense_states = jax.vmap(
# ((states, actions),) = jax.vmap(
#     Partial(
#         collect_single_rollout,
#         start_state=env_cls.init(),
#         policy=policy,
#         train_state=train_state,
#     )
# )(
#     key=rngs,
#     policy_aux=policy_aux,
# )

print(f"batch states_mean = {jnp.mean(states):.3f}")

pass

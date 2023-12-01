from env.env import Env
from env.rollout import collect_rollout_batch

from policy.policy_noise_wrapper import PolicyNoiseWrapper
from policy.finder_policy import FinderPolicy
from policy.random_policy import RandomPolicy

from learning.training.train_epoch import train_epoch
from learning.train_state import TrainState

from nets.inference import encode_state

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

# # Profiling stuff
# from memory_profiler import profile


# @profile
def train_rollout(key, train_state: TrainState):
    """Trains the model for a single rollout.

    Args:
        key (PRNGKey): Random seed for the rollout.
        train_state (TrainState): The current training state.
    """

    # Collect rollout data
    rng, key = jax.random.split(key)

    random_vectors = jax.random.normal(
        key=rng,
        shape=[
            train_state.train_config.traj_per_rollout,
            train_state.train_config.latent_state_dim,
        ],
    )
    random_vector_norms = jnp.linalg.norm(random_vectors, ord=1, axis=-1)
    unit_norm_samples = random_vectors / random_vector_norms[..., None]

    target_states = unit_norm_samples * train_state.train_config.state_radius * 1.25

    policy = PolicyNoiseWrapper(FinderPolicy.init())
    policy_auxes = jax.vmap(
        Partial(
            policy.make_aux,
            variances=jnp.ones(2) * 0.1,
        )
    )(target_state=target_states)

    rng, key = jax.random.split(key)
    start_state = train_state.train_config.env_cls.init()
    (states, actions), rollout_infos, dense_states = collect_rollout_batch(
        key=rng,
        start_state=start_state,
        policy=policy,
        policy_auxs=policy_auxes,
        train_state=train_state,
        batch_size=train_state.train_config.traj_per_rollout,
    )

    rollout_infos = rollout_infos.condense(method="mean")

    final_states = states[..., -1, :]

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(final_states))
    latent_final_states = jax.vmap(
        jax.tree_util.Partial(
            encode_state,
            train_state=train_state,
        )
    )(rngs, final_states)

    final_latent_diffs = latent_final_states - target_states
    final_latent_diff_norms = jnp.linalg.norm(final_latent_diffs, ord=1, axis=-1)
    final_latent_diff_norms_mean = jnp.mean(final_latent_diff_norms)

    rollout_infos = rollout_infos.add_plain_info(
        "mean_rollout_costs", final_latent_diff_norms_mean
    )
    rollout_infos.dump_to_wandb(train_state, prefix="Rollout")

    traj_states_for_render = dense_states[0]
    train_state.train_config.env_cls.send_wandb_video(
        name="Rollout Video",
        states=traj_states_for_render,
        env_config=train_state.train_config.env_config,
        step=train_state.step,
    )

    # Comment this out to not clog up the actor info
    # rollout_infos.dump_to_wandb(train_state)

    # Train the model for a bunch of epochs over the rollout data
    def train_epoch_for_scan(train_state, key):
        new_train_state = train_epoch(key, states, actions, train_state)
        return new_train_state, new_train_state

    rng, key = jax.random.split(rng)
    rngs = jax.random.split(key, train_state.train_config.epochs)
    train_state, _ = jax.lax.scan(train_epoch_for_scan, train_state, rngs)

    return train_state

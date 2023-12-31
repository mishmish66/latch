from latch.env import Env
from latch.learning import collect_rollout

from latch.policy import PolicyNoiseWrapper, FinderPolicy, RandomPolicy

from learning.training.train_epoch import train_epoch
from latch.latch_state import TrainState

from latch import Infos

from nets.inference import encode_state

import jax
import jax.numpy as jnp
from jax.tree_util import Partial

from typing import Tuple


def train_rollout(train_state: TrainState) -> TrainState:
    """Trains the model for a single rollout.

    Args:
        train_state (TrainState): The current training state.
    """

    # Fork out a key from the train_state
    key, train_state = train_state.split_key()

    # Collect rollout data
    rng, key = jax.random.split(key)

    # TODO: consider switching away from shell targets
    random_vectors = jax.random.normal(
        key=rng,
        shape=[
            train_state.train_config.traj_per_rollout,
            train_state.train_config.latent_state_dim,
        ],
    )
    random_vector_norms = jnp.linalg.norm(random_vectors, ord=1, axis=-1)
    unit_norm_samples = random_vectors / random_vector_norms[..., None]

    target_states = (
        unit_norm_samples * train_state.train_config.latent_state_radius * 1.1
    )

    # Define a function that makes a policy for a given target
    def make_policy(target: jax.Array):
        finder = FinderPolicy(latent_target=target)
        noisy = PolicyNoiseWrapper(wrapped_policy=finder, variances=jnp.ones(2) * 0.1)
        return noisy

    # Make the policy for each target
    policies = jax.vmap(make_policy)(target_states)

    # Use the target network to run the policy
    start_state = train_state.train_config.env.reset()

    # Collect a rollout of physics data
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, train_state.train_config.traj_per_rollout)
    rollout_result: Tuple[Tuple[jax.Array, jax.Array], Infos, jax.Array] = jax.vmap(
        Partial(
            collect_rollout,
            start_state=start_state,
            train_state=train_state,
            return_dense_states=True,
        )
    )(key=rngs, policy=policies)

    (states, actions), rollout_infos, dense_states = rollout_result

    # Squash the infos
    rollout_infos = rollout_infos.condense(method="mean")

    # Grab the final states reached during the rollouts
    final_states = states[..., -1, :]

    # Compute the mean distance between the final states and the targets
    latent_final_states = jax.vmap(
        jax.tree_util.Partial(
            encode_state,
            net_state=train_state.target_net_state,
            train_config=train_state.train_config,
        )
    )(final_states)
    final_latent_diffs = latent_final_states - target_states
    final_latent_diff_norms = jnp.linalg.norm(final_latent_diffs, ord=1, axis=-1)
    final_latent_diff_norms_mean = jnp.mean(final_latent_diff_norms)

    # Add more infos
    rollout_infos = rollout_infos.add_info(
        "mean_rollout_costs", final_latent_diff_norms_mean
    )
    infos = Infos()
    infos = infos.add_info("rollout_infos", rollout_infos)

    # Render one of the rollouts
    render_states = dense_states[0]
    train_state.train_config.env.send_wandb_video(
        name="Rollout Video",
        states=render_states,
        step=train_state.step,
        dense=True,
    )

    # Train the model for a bunch of epochs over the rollout data
    def train_epoch_for_scan(train_state, _):
        new_train_state = train_epoch(states, actions, train_state)
        return new_train_state, None

    train_state, _ = jax.lax.scan(
        train_epoch_for_scan,
        train_state,
        None,
        length=train_state.train_config.epochs,
    )

    # Update the target network
    train_state = train_state.pull_target()

    # Increment the rollout counter
    train_state = train_state.replace(rollout=train_state.rollout + 1)  # type: ignore

    return train_state

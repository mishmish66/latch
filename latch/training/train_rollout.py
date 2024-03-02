from typing import Tuple

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
from jax.tree_util import Partial

from latch import Infos, LatchState
from latch.env import Env
from latch.policy import FinderPolicy, PolicyNoiseWrapper
from latch.rollout import collect_rollout

from .train_epoch import train_epoch


def train_rollout(train_state: LatchState) -> LatchState:
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
            train_state.config.traj_per_rollout,
            train_state.config.latent_state_dim,
        ],
    )
    random_vector_norms = jnp.linalg.norm(random_vectors, ord=1, axis=-1)
    unit_norm_samples = random_vectors / random_vector_norms[..., None]

    target_states = unit_norm_samples * train_state.config.latent_state_radius * 1.1

    # Define a function that makes a policy for a given target
    def make_policy(target: jax.Array):
        finder = FinderPolicy(latent_target=target)
        noisy = PolicyNoiseWrapper(wrapped_policy=finder, variances=jnp.ones(2) * 0.025)
        return noisy

    # Make the policy for each target
    policies = jax.vmap(make_policy)(target_states)

    # Use the target network to run the policy
    start_state = train_state.config.env.reset()

    # Collect a rollout of physics data
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, train_state.config.traj_per_rollout)
    rollout_result: Tuple[Tuple[jax.Array, jax.Array], Infos, jax.Array] = jax.vmap(
        Partial(
            collect_rollout,
            start_state=start_state,
            train_state=train_state,
            return_dense_states=True,
        )
    )(key=rngs, policy=policies)

    (states, actions), rollout_infos, dense_states = rollout_result

    # Grab the final states reached during the rollouts
    final_states = states[..., -1, :]

    # Compute the mean distance between the final states and the targets
    latent_final_states = jax.vmap(train_state.target_models.encode_state)(final_states)
    final_latent_diffs = latent_final_states - target_states
    final_latent_diff_norms = jnp.linalg.norm(final_latent_diffs, ord=1, axis=-1)

    # Add more infos
    rollout_infos = rollout_infos.add_info(
        "mean_rollout_costs", final_latent_diff_norms
    )
    # Squash the infos
    rollout_info_hists = rollout_infos.condense(method="unstack").condense(
        method="unstack"
    )
    rollout_infos = rollout_infos.condense(method="mean").condense(method="mean")
    rollout_infos = rollout_infos.add_info("hists", rollout_info_hists)

    infos = Infos()
    infos = infos.add_info("rollout_infos", rollout_infos)

    # Log the infos
    infos.dump_to_wandb(train_state.step)

    # Render one of the rollouts
    render_states = dense_states[0]
    train_state.config.env.send_wandb_video(
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
        length=train_state.config.epochs,
    )

    # Update the target network
    train_state = train_state.pull_target()

    # Increment the rollout counter
    with jdc.copy_and_mutate(train_state) as train_state:
        train_state.rollout += 1

    return train_state

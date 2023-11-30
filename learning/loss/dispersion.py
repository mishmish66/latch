from learning.train_state import TrainState

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    sample_gaussian,
    get_neighborhood_states,
    get_neighborhood_actions,
)

import jax
from jax import numpy as jnp


def loss_dispersion(key, states, actions, train_state: TrainState, num_samples=8):
    """Computes the dispersion loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (n x s) array of n states with dim s
        actions (array): An (n x a) array of n actions with dim a
        train_state (TrainState): The current training state.
        num_samples (int, optional): The number of states to sample from the batch to compute the loss for pairwise. Defaults to 8.

    Returns:
        (scalar, Info): A tuple containing the loss value and associated info object.
    """

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(states))
    latent_states = jax.vmap(
        jax.tree_util.Partial(
            encode_state,
            train_state=train_state,
        )
    )(
        key=rngs,
        state=states,
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(actions))
    latent_actions = jax.vmap(
        jax.tree_util.Partial(
            encode_action,
            train_state=train_state,
        )
    )(
        key=rngs,
        action=actions,
        latent_state=latent_states,
    )

    rng_0, rng_1, key = jax.random.split(key, 3)
    sampled_latent_states = jax.random.choice(
        rng_0, latent_states, shape=(num_samples,)
    )
    sampled_latent_actions = jax.random.choice(
        rng_1, latent_actions, shape=(num_samples,)
    )

    pairwise_latent_state_diffs = (
        sampled_latent_states[..., None, :] - sampled_latent_states[..., None, :, :]
    )
    pairwise_latent_action_diffs = (
        sampled_latent_actions[..., None, :] - sampled_latent_actions[..., None, :, :]
    )

    pairwise_latent_state_diffs_norm = jnp.linalg.norm(
        pairwise_latent_state_diffs, ord=1, axis=-1
    )
    pairwise_latent_action_diffs_norm = jnp.linalg.norm(
        pairwise_latent_action_diffs, ord=1, axis=-1
    )

    state_dispersion_loss = jnp.mean(pairwise_latent_state_diffs_norm)
    action_dispersion_loss = jnp.mean(pairwise_latent_action_diffs_norm)
    total_loss = state_dispersion_loss + action_dispersion_loss

    infos = Infos.init()
    infos = infos.add_plain_info("state_dispersion_loss", state_dispersion_loss)
    infos = infos.add_plain_info("action_dispersion_loss", action_dispersion_loss)

    return total_loss, infos

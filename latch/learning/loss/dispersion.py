from learning.train_state import TrainState, TrainConfig, NetState

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    get_neighborhood_states,
    get_neighborhood_actions,
)

import jax
from jax import numpy as jnp


def loss_dispersion(
    key, states, actions, net_state: NetState, train_config: TrainConfig, num_samples=8
):
    """Computes the dispersion loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (n x s) array of n states with dim s
        actions (array): An (n x a) array of n actions with dim a
        net_state (NetState): The network weights to use.
        train_config (TrainState): The training config.
        num_samples (int, optional): The number of states to sample from the batch to compute the loss for pairwise. Defaults to 8.

    Returns:
        (scalar, Info): A tuple containing the loss value and associated info object.
    """

    latent_states = jax.vmap(
        jax.tree_util.Partial(
            encode_state,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        state=states,
    )
    latent_actions = jax.vmap(
        jax.tree_util.Partial(
            encode_action,
            net_state=net_state,
            train_config=train_config,
        )
    )(
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

    state_dispersion_loss = -jnp.mean(jnp.log(pairwise_latent_state_diffs_norm + 1.0))
    action_dispersion_loss = -jnp.mean(jnp.log(pairwise_latent_action_diffs_norm + 1.0))
    total_loss = state_dispersion_loss + action_dispersion_loss

    infos = Infos.init()
    infos = infos.add_plain_info("state_dispersion_loss", state_dispersion_loss)
    infos = infos.add_plain_info("action_dispersion_loss", action_dispersion_loss)

    return total_loss, infos

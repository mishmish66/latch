from ..train_state import TrainState, NetState, TrainConfig

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    sample_gaussian,
)

import jax
from jax import numpy as jnp


def loss_reconstruction(
    key, states, actions, net_state: NetState, train_config: TrainConfig
):
    """This calculates the reconstruction loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (n x s) array of n states with dim s
        actions (array): An (n x a) array of n actions with dim a
        net_state (NetState): The network weights to use.
        train_config (TrainConfig): The training config.

    Returns:
        (scalar, Info): A tuple containing the loss value and associated info object.
    """

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(states))
    latent_states = jax.vmap(
        jax.tree_util.Partial(
            encode_state,
            net_state=net_state,
            train_config=train_config,
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
            net_state=net_state,
            train_config=train_config,
        )
    )(
        key=rngs,
        action=actions,
        latent_state=latent_states,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(actions))
    reconstructed_states = jax.vmap(
        jax.tree_util.Partial(
            decode_state,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        key=rngs,
        latent_state=latent_states,
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(actions))
    reconstructed_actions = jax.vmap(
        jax.tree_util.Partial(
            decode_action,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        key=rngs,
        latent_state=latent_states,
        latent_action=latent_actions,
    )

    state_mae = jnp.mean(jnp.abs(states - reconstructed_states))
    action_mae = jnp.mean(jnp.abs(actions - reconstructed_actions))
    reconstruction_loss = state_mae + action_mae

    infos = Infos.init()
    infos = infos.add_plain_info("state_reconstruction_loss", state_mae)
    infos = infos.add_plain_info("action_reconstruction_loss", action_mae)

    return reconstruction_loss, infos

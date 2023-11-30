from ..train_state import TrainState

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


def loss_reconstruction(key, states, actions, train_state: TrainState):
    """This calculates the reconstruction loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (n x s) array of n states with dim s
        actions (array): An (n x a) array of n actions with dim a
        train_state (TrainState): The current training state.

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

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(actions))
    reconstructed_states = jax.vmap(
        jax.tree_util.Partial(
            decode_state,
            train_state=train_state,
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
            train_state=train_state,
        )
    )(
        key=rngs,
        latent_state=latent_states,
        latent_action=latent_actions,
    )

    state_msle = jnp.mean(jnp.log(jnp.square(states - reconstructed_states) + 1))
    action_msle = jnp.mean(jnp.log(jnp.square(actions - reconstructed_actions) + 1))
    reconstruction_loss = state_msle + action_msle

    infos = Infos.init()
    infos = infos.add_plain_info("state_reconstruction_loss", state_msle)
    infos = infos.add_plain_info("action_reconstruction_loss", action_msle)

    return reconstruction_loss, infos

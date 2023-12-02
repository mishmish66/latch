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

    # I'm defining this weird loss function to keep a gradient near 0 but also have a large gradient when the error is large.
    def mean_max_sq_log_error(result, gt):
        errors = jnp.abs(result - gt)
        sq_errors = jnp.square(errors)
        log_errors = jnp.log(errors + 1.0)
        maxxed = jnp.maximum(sq_errors, log_errors)
        return jnp.mean(maxxed)

    state_loss = mean_max_sq_log_error(reconstructed_states, states)
    action_loss = mean_max_sq_log_error(reconstructed_actions, actions)

    reconstruction_loss = state_loss + action_loss

    infos = Infos.init()
    infos = infos.add_plain_info("state_reconstruction_loss", state_loss)
    infos = infos.add_plain_info("action_reconstruction_loss", action_loss)

    return reconstruction_loss, infos

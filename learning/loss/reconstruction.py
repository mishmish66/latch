from ..train_state import TrainState, NetState, TrainConfig

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    infer_states,
    decode_state,
    decode_action,
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
        ((scalar, scalar), Info): A tuple containing a tuple of loss values for states and actions, and associated info object.
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
        action=actions,
        latent_state=latent_states,
    )

    reconstructed_states = jax.vmap(
        jax.tree_util.Partial(
            decode_state,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        latent_state=latent_states,
    )

    reconstructed_actions = jax.vmap(
        jax.tree_util.Partial(
            decode_action,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        latent_action=latent_actions,
        latent_state=latent_states,
    )

    def loss_fn(x, y):
        err = jnp.abs(x - y)
        err_sq = jnp.square(err)
        err_ln = jnp.log(err + 1e-8)
        return jnp.mean(err_sq + err_ln)

    state_loss = loss_fn(states - reconstructed_states)
    action_loss = loss_fn(actions - reconstructed_actions)

    return (state_loss, action_loss), Infos.init()

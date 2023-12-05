from ..train_state import TrainState, NetState, TrainConfig

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    eval_log_gaussian,
    get_state_space_gaussian,
    get_action_space_gaussian,
    infer_states,
    sample_gaussian,
)

import jax
from jax import numpy as jnp
from jax.scipy.stats.norm import pdf as gaussian_pdf


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

    state_space_gaussian = jax.vmap(
        jax.tree_util.Partial(
            get_state_space_gaussian,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        latent_state=latent_states,
    )
    action_space_gaussian = jax.vmap(
        jax.tree_util.Partial(
            get_action_space_gaussian,
            net_state=net_state,
            train_config=train_config,
        )
    )(
        latent_state=latent_states,
        latent_action=latent_actions,
    )

    # Add to the variance to prevent numerical issues
    state_space_gaussian = state_space_gaussian.at[
        ..., train_config.latent_state_dim :
    ].add(1e-6)
    action_space_gaussian = action_space_gaussian.at[
        ..., train_config.latent_action_dim :
    ].add(1e-6)

    state_probs = jax.vmap(
        eval_log_gaussian,
    )(
        gaussian=state_space_gaussian,
        point=states,
    )
    action_probs = jax.vmap(
        eval_log_gaussian,
    )(
        gaussian=action_space_gaussian,
        point=actions,
    )

    state_loss = -jnp.mean(state_probs)
    action_loss = -jnp.mean(action_probs)

    return (state_loss, action_loss), Infos.init()

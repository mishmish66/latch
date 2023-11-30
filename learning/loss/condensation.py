from ..train_state import TrainState

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


def loss_condensation(key, states, actions, train_state: TrainState):
    """Comutes the condensation loss for a set of states and actions.

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

    state_radii = jnp.linalg.norm(latent_states, axis=-1)
    action_radii = jnp.linalg.norm(latent_actions, axis=-1)

    state_radius_violations = jnp.maximum(
        0.0, state_radii - train_state.train_config.state_radius
    )
    action_radius_violations = jnp.maximum(
        0.0, action_radii - train_state.train_config.action_radius
    )

    state_radius_violation_log = jnp.log(state_radius_violations + 1e-6)
    action_radius_violation_log = jnp.log(action_radius_violations + 1e-6)

    state_radius_violation_loss = jnp.mean(state_radius_violation_log)
    action_radius_violation_loss = jnp.mean(action_radius_violation_log)

    total_loss = state_radius_violation_loss + action_radius_violation_loss

    infos = Infos.init()
    infos = infos.add_plain_info(
        "state_radius_violation_loss", state_radius_violation_loss
    )
    infos = infos.add_plain_info(
        "action_radius_violation_loss", action_radius_violation_loss
    )

    return total_loss, infos

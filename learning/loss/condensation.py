from ..train_state import TrainConfig, NetState

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


def loss_condensation(
    key,
    states,
    actions,
    net_state: NetState,
    train_config: TrainConfig,
):
    """Comutes the condensation loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (n x s) array of n states with dim s
        actions (array): An (n x a) array of n actions with dim a
        net_state (NetState): The network weights to use.
        train_config (TrainConfig): The training config.

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

    state_radii = jnp.linalg.norm(latent_states, ord=1, axis=-1)
    action_radii = jnp.linalg.norm(latent_actions, ord=1, axis=-1)

    state_radius_violations = jnp.maximum(0.0, state_radii - train_config.state_radius)
    action_radius_violations = jnp.maximum(
        0.0, action_radii - train_config.action_radius
    )

    def sq_p_log(x):
        return jnp.square(x) + jnp.log(x + 1e-6)

    state_radius_violation_log = sq_p_log(state_radius_violations)
    action_radius_violation_log = sq_p_log(action_radius_violations)

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

    return jnp.zeros_like(total_loss), infos

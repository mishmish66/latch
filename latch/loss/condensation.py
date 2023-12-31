from .loss import SigmoidGatedLoss

from latch.nets import NetState

from infos import Infos

import jax
from jax import numpy as jnp

from einops import rearrange

from typing import override, Tuple


class StateCondensationLoss(SigmoidGatedLoss):
    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ) -> Tuple[float, Infos]:
        """Computes the condensation loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): An (b x t-1 x a) array of n actions with dim a
            net_state (NetState): The network weights to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = rearrange(states, "b t s -> (b t) s")

        latent_states = jax.vmap(net_state.encode_state)(state=states)

        state_radii = jnp.linalg.norm(latent_states, ord=1, axis=-1)

        state_radius_violations = jnp.maximum(
            0.0, state_radii - net_state.latent_state_radius
        )

        state_radius_violation_square = jnp.square(state_radius_violations)
        state_radius_violation_log = jnp.log(state_radius_violations + 1e-6)

        losses = state_radius_violation_square + state_radius_violation_log
        loss = jnp.mean(losses).item()

        infos = Infos()
        infos = infos.add_info("loss", loss)

        return loss, infos


class ActionCondensationLoss(SigmoidGatedLoss):
    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ) -> Tuple[float, Infos]:
        """Computes the condensation loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): An (b x t-1 x a) array of n actions with dim a
            net_state (NetState): The network weights to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = states[..., :-1, :]
        states = rearrange(states, "b t s -> (b t) s")
        actions = rearrange(actions, "b t a -> (b t) a")

        latent_states = jax.vmap(
            net_state.encode_state,
        )(
            state=states,
        )
        latent_actions = jax.vmap(
            net_state.encode_action,
        )(
            action=actions,
            latent_state=latent_states,
        )

        action_radii = jnp.linalg.norm(latent_actions, ord=1, axis=-1)

        action_radius_violations = jnp.maximum(
            0.0, action_radii - net_state.latent_action_radius
        )

        action_radius_violation_square = jnp.square(action_radius_violations)
        action_radius_violation_log = jnp.log(action_radius_violations + 1e-6)

        losses = action_radius_violation_square + action_radius_violation_log
        loss = jnp.mean(losses).item()

        infos = Infos()
        infos = infos.add_info("loss", loss)

        return loss, infos

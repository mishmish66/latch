from typing import Tuple

import jax
import jax_dataclasses as jdc
from einops import rearrange
from jax import numpy as jnp
from overrides import override

from latch import Infos
from latch.models import ModelState

from .loss_func import WeightedLossFunc


@jdc.pytree_dataclass(kw_only=True)
class StateCondensationLoss(WeightedLossFunc):
    # class StateCondensationLoss(SpikeGatedLoss):
    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the condensation loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): An (b x t-1 x a) array of n actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = rearrange(states, "b t s -> (b t) s")

        latent_states = jax.vmap(models.encode_state)(state=states)

        state_radii = jnp.linalg.norm(latent_states, ord=1, axis=-1)

        state_radius_violations = jnp.maximum(
            0.0, state_radii - models.latent_state_radius
        )

        state_radius_violation_log = jnp.log(state_radius_violations + 1.0)

        losses = state_radius_violation_log
        loss = jnp.mean(losses)

        infos = Infos()
        infos = infos.add_info("raw", loss)

        return loss, infos


@jdc.pytree_dataclass(kw_only=True)
class ActionCondensationLoss(WeightedLossFunc):
    # class ActionCondensationLoss(SpikeGatedLoss):
    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the condensation loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): An (b x t-1 x a) array of n actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = states[..., :-1, :]
        states = rearrange(states, "b t s -> (b t) s")
        actions = rearrange(actions, "b t a -> (b t) a")

        latent_states = jax.vmap(
            models.encode_state,
        )(
            state=states,
        )
        latent_actions = jax.vmap(
            models.encode_action,
        )(
            action=actions,
            latent_state=latent_states,
        )

        action_radii = jnp.linalg.norm(latent_actions, ord=1, axis=-1)

        action_radius_violations = jnp.maximum(
            0.0, action_radii - models.latent_action_radius
        )

        action_radius_violation_log = jnp.log(action_radius_violations + 1.0)

        losses = action_radius_violation_log
        loss = jnp.mean(losses)

        infos = Infos()
        infos = infos.add_info("raw", loss)

        return loss, infos

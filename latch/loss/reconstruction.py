from dataclasses import dataclass
from typing import Tuple

import jax
import jax_dataclasses as jdc
from einops import rearrange
from hydra.core.config_store import ConfigStore
from jax import numpy as jnp
from jax.tree_util import Partial
from overrides import override

from latch import Infos
from latch.loss.loss_func import LossFunc
from latch.models import ModelState

from .loss_func import WeightedLossFunc
from .loss_registry import register_loss

cs = ConfigStore.instance()


@register_loss("state_reconstruction")
@jdc.pytree_dataclass(kw_only=True)
class StateReconstructionLoss(WeightedLossFunc):
    """Computes the action reconstruction loss for a set of states and actions."""

    @override
    def compute_raw(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the reconstruction loss for a set of states and actions.

        Args:The network weights
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): A (b x t-1 x a) array of n actions with dim a
            models (ModelState): The models to use.

        Returns:
            ((scalar, scalar), Info): A tuple containing a tuple of loss values for states and actions, and associated info object.
        """

        # flatten the states since we don't care about time here
        states = rearrange(states, "b l s -> (b l) s")

        latent_states = jax.vmap(
            models.encode_state,
        )(states)
        reconstructed_states = jax.vmap(
            models.decode_state,
        )(latent_state=latent_states)

        error = jnp.abs(states - reconstructed_states)
        error_square = jnp.square(error)
        error_log = jnp.log(error + 1e-8)
        losses = error_square  # + error_log
        loss = jnp.mean(losses)

        return loss, Infos()

    @dataclass
    class Config(WeightedLossFunc.Config):
        loss_type: str = "state_reconstruction"


cs.store(group="loss", name="state_reconstruction", node=StateReconstructionLoss.Config)


@register_loss("action_reconstruction")
@jdc.pytree_dataclass(kw_only=True)
class ActionReconstructionLoss(WeightedLossFunc):
    """Computes the reconstruction loss for a set of states and actions."""

    name: str = "action_reconstruction"

    @override
    def compute_raw(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the reconstruction loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (n x s) array of n states with dim s
            actions (array): An (n x a) array of n actions with dim a
            models (ModelState): The network weights to use.

        Returns:
            ((scalar, scalar), Info): A tuple containing a tuple of loss values for states and actions, and associated info object.
        """

        # Drop the last state to make the states and actions the same length
        states = states[..., :-1, :]
        # flatten the states since we don't care about time here
        states = rearrange(states, "b l s -> (b l) s")
        actions = rearrange(actions, "b l a -> (b l) a")

        latent_states = jax.vmap(
            models.encode_state,
        )(states)
        latent_actions = jax.vmap(
            models.encode_action,
        )(action=actions, latent_state=latent_states)

        reconstructed_actions = jax.vmap(models.decode_action)(
            latent_action=latent_actions, latent_state=latent_states
        )

        error = jnp.abs(actions - reconstructed_actions)
        error_square = jnp.square(error)
        error_log = jnp.log(error + 1e-8)
        losses = error_square  # + error_log
        loss = jnp.mean(losses)

        return loss, Infos()

    @dataclass
    class Config(WeightedLossFunc.Config):
        loss_type: str = "action_reconstruction"


cs.store(
    group="loss", name="action_reconstruction", node=ActionReconstructionLoss.Config
)

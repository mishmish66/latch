from .loss import WeightedLoss

from latch.nets import NetState

from infos import Infos


import jax
from jax.tree_util import Partial
from jax import numpy as jnp

from einops import rearrange

from typing import override


class StateReconstructionLoss(WeightedLoss):
    """Computes the action reconstruction loss for a set of states and actions."""

    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ):
        """Computes the reconstruction loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): A (b x t-1 x a) array of n actions with dim a
            net_state (NetState): The network weights to use.

        Returns:
            ((scalar, scalar), Info): A tuple containing a tuple of loss values for states and actions, and associated info object.
        """

        # flatten the states since we don't care about time here
        states = rearrange(states, "b l s -> (b l) s")

        latent_states = jax.vmap(
            net_state.encode_state,
        )(states)
        reconstructed_states = jax.vmap(
            net_state.decode_state,
        )(latent_state=latent_states)

        error = jnp.abs(states - reconstructed_states)
        error_square = jnp.square(error)
        error_log = jnp.log(error + 1e-8)
        losses = error_square + error_log
        loss: float = jnp.mean(losses).item()

        infos = Infos()
        infos = infos.add_info("loss", loss)

        return loss, infos


class ActionReconstructionLoss(WeightedLoss):
    """Computes the reconstruction loss for a set of states and actions."""

    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ):
        """Computes the reconstruction loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (n x s) array of n states with dim s
            actions (array): An (n x a) array of n actions with dim a
            net_state (NetState): The network weights to use.

        Returns:
            ((scalar, scalar), Info): A tuple containing a tuple of loss values for states and actions, and associated info object.
        """

        # Drop the last state to make the states and actions the same length
        states = states[..., :-1, :]
        # flatten the states since we don't care about time here
        states = rearrange(states, "b l s -> (b l) s")
        actions = rearrange(actions, "b l a -> (b l) a")

        latent_states = jax.vmap(
            net_state.encode_state,
        )(states)
        latent_actions = jax.vmap(
            net_state.encode_action,
        )(action=actions, latent_state=latent_states)

        reconstructed_actions = jax.vmap(net_state.decode_action)(
            latent_action=latent_actions, latent_state=latent_states
        )

        error = jnp.abs(actions - reconstructed_actions)
        error_square = jnp.square(error)
        error_log = jnp.log(error + 1e-8)
        losses = error_square + error_log
        loss: float = jnp.mean(losses).item()

        infos = Infos()
        infos = infos.add_info("loss", loss)

        return loss, infos

from latch.nets import NetState

from infos import Infos

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from einops import rearrange

from typing import Any, List, Tuple, override
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ) -> Tuple[float, Infos]:
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x t x s) array of b trajectories of l states with dim s
            actions (array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            net_state (NetState): The networks to train.

        Returns:
            (float, Infos): A tuple containing the loss and associated infos object.
        """
        pass

    def __call__(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ):
        loss, infos = self.compute(key, states, actions, net_state)

        return loss, infos


@jdc.pytree_dataclass(kw_only=True)
class WeightedLoss(Loss):
    """Abstract base class for losses."""

    weight: float = 1.0

    def __call__(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        net_state: NetState,
    ):
        super_loss, infos = super().__call__(
            key=key,
            states=states,
            actions=actions,
            net_state=net_state,
        )

        weighted_loss = super_loss * self.weight
        infos = infos.add_info("weighted_loss", weighted_loss)

        return weighted_loss, infos


@jdc.pytree_dataclass(kw_only=True)
class SigmoidGatedLoss(WeightedLoss):
    sharpness: float = 1.0
    center: float = 0.0

    def __call__(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        gate_in: float,
        net_state: NetState,
    ):
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x t x s) array of b trajectories of l states with dim s
            actions (array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            net_state (NetState): The networks to train.

        Returns:
            (float, Infos): A tuple containing the loss and associated infos object.
        """
        sg_gate_in = jax.lax.stop_gradient(gate_in)

        gate_value: float = (
            1 + jnp.exp(self.sharpness * (sg_gate_in - self.center)).item()
        ) ** -1

        super_loss, infos = super().__call__(
            key=key,
            states=states,
            actions=actions,
            net_state=net_state,
        )

        gated_loss = gate_value * super_loss
        infos = infos.add_info("gate_value", gate_value)
        infos = infos.add_info("gated_loss", gated_loss)

        return gated_loss, infos


def make_log_gate_value(x, sharpness, center):
    sgx = jax.lax.stop_gradient(x)
    base = sgx / center
    den = 1 + base**sharpness
    return 1 / den

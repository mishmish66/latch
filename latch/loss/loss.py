from latch.models import ModelState

from latch import Infos

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from overrides import override

from einops import rearrange

from typing import Any, List, Tuple
from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x t x s) array of b trajectories of l states with dim s
            actions (array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            models (ModelState): The models to use.

        Returns:
            (float, Infos): A tuple containing the loss and associated infos object.
        """
        pass

    def __call__(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ):
        loss, infos = self.compute(key, states, actions, models)

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
        models: ModelState,
    ):
        super_loss, infos = super().__call__(
            key=key,
            states=states,
            actions=actions,
            models=models,
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
        models: ModelState,
    ):
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x t x s) array of b trajectories of l states with dim s
            actions (array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            models (ModelState): The models to use.

        Returns:
            (float, Infos): A tuple containing the loss and associated infos object.
        """
        sg_gate_in = jax.lax.stop_gradient(gate_in)

        gate_value: jax.Array = (
            1 + jnp.exp(self.sharpness * (sg_gate_in - self.center))
        ) ** -1

        super_loss, infos = super().__call__(
            key=key,
            states=states,
            actions=actions,
            models=models,
        )

        gated_loss = gate_value * super_loss
        infos = infos.add_info("gate_value", gate_value)
        infos = infos.add_info("gated_loss", gated_loss)

        return gated_loss, infos


# def make_log_gate_value(x, sharpness, center):
#     sgx = jax.lax.stop_gradient(x)
#     base = sgx / center
#     den = 1 + base**sharpness
#     return 1 / den

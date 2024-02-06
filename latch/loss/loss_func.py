from latch.models import ModelState

from latch import Infos

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from overrides import override

from einops import rearrange

from typing import Any, List, Tuple
from abc import ABC, abstractmethod


class LossFunc(ABC):
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
class WeightedLossFunc(LossFunc):
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

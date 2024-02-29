import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List, Tuple

import jax
import jax_dataclasses as jdc
from einops import rearrange
from jax import numpy as jnp
from omegaconf import OmegaConf
from overrides import override

from latch import Infos
from latch.models import ModelState


@jdc.pytree_dataclass(kw_only=True)
class LossFunc(ABC):
    name: str

    @abstractmethod
    def compute_raw(
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

    def postprocess(
        self,
        gated_loss: jax.Array,
        infos: Infos,
    ) -> Tuple[jax.Array, Infos]:
        return gated_loss, infos

    @dataclass
    class Config:
        name: str
        loss_type: str

    @classmethod
    def configure(cls, config: "LossFunc.Config") -> "LossFunc":
        arg_dict: dict = OmegaConf.to_container(config)  # type: ignore
        arg_dict.pop("loss_type")
        return cls(**arg_dict)


@jdc.pytree_dataclass(kw_only=True)
class WeightedLossFunc(LossFunc):
    """Abstract base class for losses."""

    weight: float = 1.0

    def postprocess(
        self, gated_loss: jax.Array, infos: Infos
    ) -> Tuple[jax.Array, Infos]:
        super_loss, infos = super().postprocess(gated_loss, infos)

        weighted_loss = super_loss * self.weight
        infos = infos.add_info("weighted_loss", weighted_loss)

        return weighted_loss, infos

    @dataclass
    class Config(LossFunc.Config):
        weight: float = 1.0

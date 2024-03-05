from abc import ABC, abstractmethod
from typing import Generic, Tuple, TypeVar

import jax
import jax_dataclasses as jdc

from latch.infos import Infos
from latch.latch_state import LatchState

from .policy import Policy


@jdc.pytree_dataclass(kw_only=True)
class PresetPolicy(Policy[None]):
    """Policy that plays back preset actions."""

    preset_actions: jax.Array

    def make_init_carry(
        self,
        key: jax.Array,
        start_state: jax.Array,
        train_state: LatchState,
    ) -> Tuple[None, Infos]:
        return None, Infos()

    def __call__(
        self,
        key: jax.Array,
        state: jax.Array,
        i: int,
        carry: None,
        train_state: LatchState,
    ) -> Tuple[jax.Array, None, Infos]:

        latent_action = self.preset_actions[i]
        latent_state = train_state.target_models.encode_state(state)
        action = train_state.target_models.decode_action(
            latent_action=latent_action, latent_state=latent_state
        )

        return action, None, Infos()

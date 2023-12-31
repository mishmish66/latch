from latch.learning import TrainState
from .policy import Policy

from latch.infos import Infos

import jax
from jax import numpy as jnp

import jax_dataclasses as jdc

from typing import override


@jdc.pytree_dataclass
class RandomPolicy(Policy[None]):
    @override
    def make_init_carry(
        self,
        key,
        start_state,
        train_state: TrainState,
    ):
        return None, Infos()

    @override
    def __call__(
        self,
        key,
        state,
        i,
        carry,
        train_state: TrainState,
    ):
        rng, key = jax.random.split(key)
        random_action = train_state.train_config.env.random_action(rng)

        return (
            random_action,
            None,
            Infos(),
        )

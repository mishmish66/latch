from .policy import Policy
from latch import LatchState, Infos

import jax
from jax import numpy as jnp

import jax_dataclasses as jdc

from overrides import override


@jdc.pytree_dataclass
class RandomPolicy(Policy[None]):
    @override
    def make_init_carry(
        self,
        key: jax.Array,
        start_state: jax.Array,
        train_state: LatchState,
    ):
        return None, Infos()

    @override
    def __call__(
        self,
        key: jax.Array,
        state: jax.Array,
        i: int,
        carry: None,
        train_state: LatchState,
    ):
        rng, key = jax.random.split(key)
        random_action = train_state.config.env.random_action(rng)

        return (
            random_action,
            None,
            Infos(),
        )

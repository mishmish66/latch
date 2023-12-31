from latch.learning import TrainState

from .policy import Policy

from latch.infos import Infos

import jax
from jax import numpy as jnp

import jax_dataclasses as jdc

from typing import TypeVar, Tuple, Callable, override


@jdc.pytree_dataclass(kw_only=True)
class PolicyNoiseWrapper[C](Policy[C]):
    wrapped_policy: Policy[C]
    variances: jax.Array

    @override
    def make_init_carry(
        self,
        key,
        start_state,
        train_state: TrainState,
    ) -> Tuple[C, Infos]:
        return self.wrapped_policy.make_init_carry(
            key=key,
            start_state=start_state,
            train_state=train_state,
        )

    def __call__(
        self,
        key,
        state,
        i,
        carry,
        train_state: TrainState,
    ) -> Tuple[jax.Array, C, Infos]:
        rng, key = jax.random.split(key)
        no_noise_act, wrapped_carry, wrapped_infos = self.wrapped_policy(
            key=rng,
            state=state,
            i=i,
            carry=carry,
            train_state=train_state,
        )

        rng, key = jax.random.split(key)
        unit_noise = jax.random.normal(rng, shape=no_noise_act.shape)
        scaled_noise = unit_noise * jnp.sqrt(self.variances)

        noisy_act = no_noise_act + scaled_noise

        return noisy_act, wrapped_carry, wrapped_infos

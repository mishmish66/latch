from learning.train_state import NetState, TrainConfig

import jax
from jax import numpy as jnp

from dataclasses import dataclass


@dataclass
class PolicyNoiseWrapper:
    wrapped_policy: any

    def make_aux(self, variances, *args, **kwargs):
        wrapped_aux = self.wrapped_policy.make_aux(*args, **kwargs)
        return (variances, wrapped_aux)

    def make_init_carry(
        self,
        key,
        start_state,
        aux,
        net_state: NetState,
        train_config: TrainConfig,
    ):
        variances, wrapped_aux = aux
        init_carry, info = self.wrapped_policy.make_init_carry(
            key=key,
            start_state=start_state,
            aux=wrapped_aux,
            net_state=net_state,
            train_config=train_config,
        )

        return (init_carry, variances), info

    def __call__(
        self,
        key,
        state,
        i,
        carry,
        net_state: NetState,
        train_config: TrainConfig,
    ):
        carry, variances = carry
        rng, key = jax.random.split(key)
        no_noise_act, wrapped_carry, wrapped_infos = self.wrapped_policy(
            key=rng,
            state=state,
            i=i,
            carry=carry,
            net_state=net_state,
            train_config=train_config,
        )

        rng, key = jax.random.split(key)
        noise = jax.random.normal(rng, shape=no_noise_act.shape) * jnp.sqrt(variances)

        noised_act = no_noise_act + noise

        return noised_act, (wrapped_carry, variances), wrapped_infos

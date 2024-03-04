from .nets import Nets

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from typing import Optional, Any, Dict, Callable


@jdc.pytree_dataclass(kw_only=True)
class NetParams:
    state_encoder_params: Dict[str, Any]
    action_encoder_params: Dict[str, Any]
    transition_model_params: Dict[str, Any]
    state_decoder_params: Dict[str, Any]
    action_decoder_params: Dict[str, Any]


@jdc.pytree_dataclass(kw_only=True)
class ModelState:
    net_params: NetParams
    nets: Nets

    def encode_state(self, state: jax.Array) -> jax.Array:
        latent_state: jax.Array = self.nets.state_encoder.apply(  # type: ignore
            self.net_params.state_encoder_params,
            state,
        )
        return latent_state

    def encode_action(self, action: jax.Array, latent_state: jax.Array) -> jax.Array:
        latent_action: jax.Array = self.nets.action_encoder.apply(  # type: ignore
            self.net_params.action_encoder_params,
            action,
            latent_state,
        )

        return latent_action

    def decode_state(self, latent_state):
        reconstructed_state: jax.Array = self.nets.state_decoder.apply(  # type: ignore
            self.net_params.state_decoder_params,
            latent_state,
        )
        return reconstructed_state

    def decode_action(self, latent_action, latent_state):
        action: jax.Array = self.nets.action_decoder.apply(  # type: ignore
            self.net_params.action_decoder_params,
            latent_action,
            latent_state,
        )
        return action

    def infer_states(self, latent_start_state, latent_actions, current_action_i=0):
        latent_states_prime: jax.Array = self.nets.transition_model.apply(  # type: ignore
            self.net_params.transition_model_params,
            latent_start_state,
            latent_actions,
            current_action_i,
        )

        return latent_states_prime

    @property
    def latent_state_radius(self):
        return self.nets.latent_state_radius

    @property
    def latent_action_radius(self):
        return self.nets.latent_action_radius

    @property
    def latent_state_dim(self):
        return self.nets.latent_state_dim

    @property
    def latent_action_dim(self):
        return self.nets.latent_action_dim

    @property
    def gamma(self):
        return self.nets.gamma

    def get_neighborhood_states(
        self,
        key: jax.Array,
        latent_state: jax.Array,
        count: int = 1,
    ) -> jax.Array:

        def sampler(key):
            ball_sample = jax.random.ball(
                key, d=self.latent_state_dim, p=1, dtype=latent_state.dtype
            )
            return ball_sample + latent_state

        def validity_check(sample):
            inside_space = jnp.linalg.norm(sample) <= self.latent_state_radius
            sample_dist = jnp.linalg.norm(sample - latent_state, p=1)
            inside_neighborhood = sample_dist <= self.latent_state_radius

            return jnp.logical_and(inside_space, inside_neighborhood)

        return generate_passing_samples(
            key=key,
            sampler=sampler,
            validity_check=validity_check,
            count=count,
        )

    def get_neighborhood_actions(
        self,
        key: jax.Array,
        latent_action: jax.Array,
        count: int = 1,
    ):

        def sampler(key):
            ball_sample = jax.random.ball(
                key, d=self.latent_action_dim, p=1, dtype=latent_action.dtype
            )
            return ball_sample + latent_action

        def validity_check(sample):
            inside_space = jnp.linalg.norm(sample) <= self.latent_action_radius
            sample_dist = jnp.linalg.norm(sample - latent_action, ord=1)
            inside_neighborhood = sample_dist <= self.latent_action_radius

            return jnp.logical_and(inside_space, inside_neighborhood)

        return generate_passing_samples(
            key=key,
            sampler=sampler,
            validity_check=validity_check,
            count=count,
        )


def generate_passing_samples(
    key: jax.Array,
    sampler: Callable[[jax.Array], jax.Array],
    validity_check: Callable[[jax.Array], jax.Array],
    count: int = 1,
    max_loops: int = 16,
) -> jax.Array:

    def cond_fun(while_pack):
        sample, key, count = while_pack
        invalid = ~validity_check(sample)
        under_max_loops = count < max_loops
        return jnp.logical_and(under_max_loops, invalid)

    def body_fun(while_pack):
        sample, key, count = while_pack
        rng, key = jax.random.split(key)
        return sampler(rng), key, count + 1

    def generate_sample(key):
        rng, key = jax.random.split(key)
        init_val = (sampler(rng), key, 0)
        sample, key, count = jax.lax.while_loop(
            cond_fun=cond_fun,
            body_fun=body_fun,
            init_val=init_val,
        )
        return sample

    neighborhood_samples = jax.vmap(generate_sample)(jax.random.split(key, count))

    return neighborhood_samples

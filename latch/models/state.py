from .nets import Nets

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from typing import Optional, Any, Dict


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

    def get_neighborhood_states(
        self,
        key: jax.Array,
        latent_state: jax.Array,
        count: int = 1,
    ):
        rng, key = jax.random.split(key)
        ball_sample = jax.random.ball(
            rng,
            self.latent_state_dim,
            p=1,
            shape=(count,),
        )

        return latent_state + ball_sample

    def get_neighborhood_actions(
        self,
        key: jax.Array,
        latent_action: jax.Array,
        count: int = 1,
    ):
        rng, key = jax.random.split(key)
        ball_sample = jax.random.ball(
            rng,
            d=self.latent_action_dim,
            p=1,
            shape=(count,),
        )

        return latent_action + ball_sample

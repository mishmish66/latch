from latch.latch_state import TrainState, NetState
from latch.latch_config import TrainConfig

from nets.nets import make_inds

import jax
from jax import numpy as jnp
from jax.scipy.stats.norm import logpdf as norm_pdf


def encode_state(
    state,
    net_state: NetState,
    train_config: TrainConfig,
):
    latent_state = train_config.state_encoder.apply(
        net_state.state_encoder_params,
        state,
    )
    return latent_state


def get_neighborhood_states(
    key,
    latent_state,
    train_config: TrainConfig,
    count=1,
):
    rng, key = jax.random.split(key)
    ball_sample = jax.random.ball(
        rng, train_config.latent_state_dim, p=1, shape=[count]
    )

    return latent_state + ball_sample


def get_neighborhood_actions(key, latent_action, train_config: TrainConfig, count=1):
    rng, key = jax.random.split(key)
    ball_sample = jax.random.ball(
        rng, d=train_config.latent_action_dim, p=1, shape=[count]
    )

    return latent_action + ball_sample


def encode_action(
    action,
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    latent_action = train_config.action_encoder.apply(
        net_state.action_encoder_params,
        action,
        latent_state,
    )

    return latent_action


def decode_state(
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    state = state = train_config.state_decoder.apply(
        net_state.state_decoder_params,
        latent_state,
    )
    return state


def decode_action(
    latent_action,
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    action = train_config.action_decoder.apply(
        net_state.action_decoder_params,
        latent_action,
        latent_state,
    )
    return action


def infer_states(
    latent_start_state,
    latent_actions,
    net_state: NetState,
    train_config: TrainConfig,
    current_action_i=0,
):
    latent_states_prime = train_config.transition_model.apply(
        net_state.transition_model_params,
        latent_start_state,
        latent_actions,
        jnp.arange(latent_actions.shape[0]) * train_config.env_config.dt,
        current_action_i,
    )

    return latent_states_prime


def make_mask(mask_len, first_known_i):
    inds = make_inds(mask_len, first_known_i)
    mask = inds >= 0
    return mask

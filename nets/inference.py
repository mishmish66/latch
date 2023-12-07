from learning.train_state import TrainState, NetState
from learning.train_config import TrainConfig

from nets.nets import make_inds

import jax
from jax import numpy as jnp
from jax.scipy.stats.norm import pdf as norm_pdf


def eval_log_gaussian(gaussian, point):
    dim = gaussian.shape[-1] // 2
    mean = gaussian[..., :dim]
    variance = gaussian[..., dim:]

    logits = norm_pdf(point, mean, variance)
    return jnp.mean(logits, axis=-1)


def sample_gaussian(key, gaussian):
    dim = gaussian.shape[-1] // 2

    old_pre_shape = gaussian.shape[:-1]

    flat_gaussians = jnp.reshape(gaussian, (-1, gaussian.shape[-1]))

    flat_mean = flat_gaussians[:, :dim]
    flat_variance_vectors = flat_gaussians[:, dim:]

    rng, key = jax.random.split(key)
    normal = jax.random.normal(rng, flat_mean.shape)

    flat_result = flat_mean + normal * flat_variance_vectors

    result = jnp.reshape(flat_result, (*old_pre_shape, dim))

    return result


def get_latent_state_gaussian(state, net_state: NetState, train_config: TrainState):
    latent_state_gaussian = train_config.state_encoder.apply(
        net_state.state_encoder_params,
        state,
    )
    return latent_state_gaussian


def encode_state(
    key,
    state,
    net_state: NetState,
    train_config: TrainConfig,
):
    rng, key = jax.random.split(key)

    latent_state_gaussian = get_latent_state_gaussian(state, net_state, train_config)
    latent_state = sample_gaussian(rng, latent_state_gaussian)

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


def get_latent_action_gaussian(
    action,
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    latent_action_gaussian = train_config.action_encoder.apply(
        net_state.action_encoder_params,
        action,
        latent_state,
    )
    return latent_action_gaussian


def encode_action(
    key,
    action,
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    latent_action_gaussian = get_latent_action_gaussian(
        action, latent_state, net_state=net_state, train_config=train_config
    )

    rng, key = jax.random.split(key)
    latent_action = sample_gaussian(rng, latent_action_gaussian)

    return latent_action


def get_state_space_gaussian(
    latent_state, net_state: NetState, train_config: TrainConfig
):
    state_gaussian = train_config.state_decoder.apply(
        net_state.state_decoder_params,
        latent_state,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = jnp.clip(
        state_gaussian[..., train_config.latent_state_dim :], 1e-6, None
    )
    state_gaussian = jnp.concatenate(
        [
            state_gaussian[..., : train_config.latent_state_dim],
            clamped_variance,
        ],
        axis=-1,
    )

    return state_gaussian


def decode_state(key, latent_state, net_state: NetState, train_config: TrainConfig):
    state_space_gaussian = get_state_space_gaussian(
        latent_state, net_state, train_config
    )

    rng, key = jax.random.split(key)
    state = sample_gaussian(rng, state_space_gaussian)

    return state


def get_action_space_gaussian(
    latent_action,
    latent_state,
    net_state: NetState,
    train_config: TrainConfig,
):
    action_gaussian = train_config.action_decoder.apply(
        net_state.action_decoder_params,
        latent_action,
        latent_state,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = action_gaussian[..., train_config.latent_action_dim :] + 1e-6
    action_gaussian = jnp.concatenate(
        [
            action_gaussian[..., : train_config.latent_action_dim],
            clamped_variance,
        ],
        axis=-1,
    )

    return action_gaussian


def decode_action(
    key, latent_action, latent_state, net_state: NetState, train_config: TrainConfig
):
    action_space_gaussian = get_action_space_gaussian(
        latent_action,
        latent_state,
        net_state,
        train_config,
    )

    rng, key = jax.random.split(key)
    action = sample_gaussian(rng, action_space_gaussian)

    return action


def get_latent_state_prime_gaussians(
    latent_start_state,
    latent_actions,
    net_state: NetState,
    train_config: TrainConfig,
    current_action_i=0,
):
    next_state_gaussian = train_config.transition_model.apply(
        net_state.transition_model_params,
        latent_start_state,
        latent_actions,
        jnp.arange(latent_actions.shape[0]) * train_config.env_config.dt,
        current_action_i,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = next_state_gaussian[..., train_config.latent_state_dim :] + 1e-6
    latent_state_prime_gaussians = jnp.concatenate(
        [
            next_state_gaussian[..., : train_config.latent_state_dim],
            clamped_variance,
        ],
        axis=-1,
    )

    return latent_state_prime_gaussians


def infer_states(
    key,
    latent_start_state,
    latent_actions,
    net_state: NetState,
    train_config: TrainConfig,
    current_action_i=0,
):
    latent_state_prime_gaussians = get_latent_state_prime_gaussians(
        latent_start_state,
        latent_actions,
        net_state,
        train_config,
        current_action_i,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_state_prime_gaussians.shape[0])
    inferred_states = jax.vmap(sample_gaussian, (0, 0))(
        rngs, latent_state_prime_gaussians
    )
    return inferred_states


def make_mask(mask_len, first_known_i):
    inds = make_inds(mask_len, first_known_i)
    mask = inds >= 0
    return mask

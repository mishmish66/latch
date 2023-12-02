from typing import Any

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

from einops import einsum, rearrange


class FreqLayer(nn.Module):
    out_dim: jax.Array

    def setup(self):
        pass

    def __call__(self, x) -> Any:
        d = x.shape[-1]
        per_dim = (((self.out_dim // d) - 1) // 2) + 1
        indices = jnp.arange(per_dim)
        freq_factor = 5 / jnp.power(1e4, 2 * indices / d)
        operands = einsum(x, freq_factor, "d, w -> w d")
        sins = jnp.sin(operands)
        cosines = jnp.cos(operands)

        freq_result = rearrange([sins, cosines], "f w d -> (d f w)")
        sliced_freq_result = freq_result[: self.out_dim - d]

        cat_result = jnp.concatenate([x, sliced_freq_result], axis=-1)

        return cat_result


class StateEncoder(nn.Module):
    latent_state_dim: any

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=1024)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    1024,
                    1024,
                    512,
                    512,
                    256,
                    256,
                    self.latent_state_dim * 2,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        x = self.freq_layer(x)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        x = self.dense_layers[-1](x)
        x_mean = x[..., : self.latent_state_dim]
        x_std = x[..., self.latent_state_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class StateDecoder(nn.Module):
    state_dim: any

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=1024)

        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    1024,
                    1024,
                    512,
                    512,
                    256,
                    256,
                    self.state_dim * 2,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        x = self.freq_layer(x)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.dense_layers[-1](x)
        x_mean = x[..., : self.state_dim]
        x_std = x[..., self.state_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionEncoder(nn.Module):
    latent_action_dim: any

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=1024)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    1024,
                    1024,
                    512,
                    512,
                    256,
                    256,
                    self.latent_action_dim * 2,
                ]
            )
        ]

    def __call__(self, action, latent_state) -> Any:
        freq_action = self.freq_layer(action)
        x = jnp.concatenate([freq_action, latent_state], axis=-1)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        x = self.dense_layers[-1](x)

        x_mean = x[..., : self.latent_action_dim]
        x_std = x[..., self.latent_action_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionDecoder(nn.Module):
    act_dim: any

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=1024)

        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    1024,
                    1024,
                    512,
                    512,
                    256,
                    256,
                    self.act_dim * 2,
                ]
            )
        ]

    def __call__(self, latent_action, latent_state) -> Any:
        x = jnp.concatenate([latent_action, latent_state], axis=-1)
        x = self.freq_layer(x)
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.dense_layers[-1](x)
        x_mean = x[..., : self.act_dim]
        x_std = x[..., self.act_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class TemporalEncoder(nn.Module):
    min_freq: float = 0.2  # 5.0 second period
    max_freq: float = 20  # 0.05 second period

    def setup(self):
        pass

    def __call__(self, x, time) -> Any:
        d = x.shape[-1]
        indices = jnp.arange(d)

        # Compute frequencies
        freqs = jnp.logspace(
            jnp.log10(self.min_freq), jnp.log10(self.max_freq), num=d // 2
        )

        # Get phases
        phases = time * freqs

        # Compute sines and cosines of phases
        sines = jnp.sin(phases)
        cosines = jnp.cos(phases)

        # Give it the dims it needs
        freq_result = jnp.concatenate([sines, cosines])
        freq_result = jnp.zeros_like(x).at[0 : freq_result.shape[0]].set(freq_result)

        # Combine with input and return
        return x + freq_result


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    dropout: float

    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.heads,
            out_features=self.dim,
            name="ATTN",
            dropout_rate=self.dropout,
        )

        self.mlp_up = nn.Dense(self.dim * 4, name="MLPU")
        self.mlp_down = nn.Dense(self.dim, name="MLPD")

    def __call__(self, queries, keys_values, mask=None):
        x = queries
        x = x + self.attention(
            queries,
            keys_values,
            mask=mask,
        )
        u = self.mlp_up(x)
        z = nn.relu(u)
        r = self.mlp_down(z)
        x = x + nn.relu(r)

        return x


def make_inds(mask_len, first_known_i):
    inds = jnp.arange(mask_len) - first_known_i
    return inds


class TransitionModel(nn.Module):
    latent_state_dim: int
    n_layers: int
    latent_dim: int
    heads: int

    def setup(self):
        self.temporal_encoder = TemporalEncoder()

        self.state_action_expander = nn.Dense(self.latent_dim, name="ACTION_EXPANDER")

        self.t_layers = [
            TransformerLayer(
                dim=self.latent_dim,
                heads=self.heads,
                dropout=0.0,
                name=f"ATTN_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.state_condenser = nn.Dense(
            self.latent_state_dim * 2, name="STATE_CONDENSER"
        )

    def __call__(
        self,
        initial_latent_state,
        latent_actions,
        times,
        current_action_i,
    ) -> Any:
        inds = make_inds(latent_actions.shape[0], current_action_i)
        mask = inds >= 0
        masked_action_times = mask * times[inds]

        state_actions = jax.vmap(
            lambda s, a: jnp.concatenate([s, a]),
            (None, 0),
        )(initial_latent_state, latent_actions)

        # Upscale actions and state to latent dim
        x = jax.vmap(self.state_action_expander.__call__)(state_actions)

        # Apply temporal encodings
        x = jax.vmap(self.temporal_encoder)(x, masked_action_times)

        # Apply transformer layers
        for t_layer in self.t_layers:
            x = t_layer(x, x, mask)

        # Rescale states to original dim
        x = self.state_condenser(x)
        latent_state_prime_mean = x[..., : self.latent_state_dim]
        latent_state_prime_std = nn.softplus(x[..., self.latent_state_dim :])

        latent_state_prime_gauss_params = jnp.concatenate(
            [
                latent_state_prime_mean,
                latent_state_prime_std,
            ],
            axis=-1,
        )

        return latent_state_prime_gauss_params

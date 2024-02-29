from typing import Any, List

import flax
import jax
import jax_dataclasses as jdc
from einops import einsum, rearrange
from flax import linen as nn
from flax import struct
from jax import numpy as jnp
from jax.tree_util import Partial


class FreqLayer(nn.Module):
    out_dim: int = struct.field(pytree_node=False)
    min_freq: float = 0.0625
    max_freq: float = 1024.0

    def setup(self):
        pass

    def __call__(self, x) -> Any:
        d = x.shape[-1]

        # Compute frequencies
        freqs = jnp.logspace(
            jnp.log10(self.min_freq),
            jnp.log10(self.max_freq),
            num=self.out_dim // 2 // d,
        )

        # Get phases
        phases = einsum(x, freqs, "e, w -> e w")

        # Compute sines and cosines of phases
        sines = jnp.sin(phases)
        cosines = jnp.cos(phases)

        # Give it the dims it needs
        freq_result = rearrange([sines, cosines], "f e w -> (f w e)")
        freq_result = jnp.zeros(self.out_dim).at[: len(freq_result)].set(freq_result)

        return freq_result


class StateEncoder(nn.Module):
    latent_state_dim: int
    layer_sizes: List[int] = struct.field(pytree_node=False)

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=self.layer_sizes[0])

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    *self.layer_sizes[1:],
                    self.latent_state_dim,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        x = self.freq_layer(x)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        x = self.dense_layers[-1](x)
        return x


class StateDecoder(nn.Module):
    state_dim: int = struct.field(pytree_node=False)
    layer_sizes: List[int] = struct.field(pytree_node=False)

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=self.layer_sizes[0])

        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    *self.layer_sizes[1:],
                    self.state_dim,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        x = self.freq_layer(x)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.dense_layers[-1](x)
        return x


class ActionEncoder(nn.Module):
    latent_action_dim: int = struct.field(pytree_node=False)
    layer_sizes: List[int] = struct.field(pytree_node=False)

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=self.layer_sizes[0])

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    *self.layer_sizes[1:],
                    self.latent_action_dim,
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
        return x


class ActionDecoder(nn.Module):
    act_dim: int = struct.field(pytree_node=False)
    layer_sizes: List[int] = struct.field(pytree_node=False)

    def setup(self):
        self.freq_layer = FreqLayer(out_dim=self.layer_sizes[0])

        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    *self.layer_sizes[1:],
                    self.act_dim,
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
        return x


class TemporalEncoder(nn.Module):
    min_freq: float = 0.015625  # 64 element period
    max_freq: float = 0.5  # 2 element period

    def setup(self):
        pass

    def __call__(self, x) -> Any:
        latent_dim = x.shape[-1]
        sequence_length = x.shape[-2]
        indices = jnp.arange(sequence_length)

        # Compute frequencies
        freqs = jnp.logspace(
            jnp.log10(self.min_freq),
            jnp.log10(self.max_freq),
            num=latent_dim // 2,
        )

        # Get phases
        phases = einsum(indices, freqs, "i, d -> i d")

        # Compute sines and cosines of phases
        sines = jnp.sin(phases)
        cosines = jnp.cos(phases)

        # Give it the dims it needs
        freq_result = jnp.concatenate([sines, cosines], axis=-1)
        freq_result = (
            jnp.zeros_like(x).at[..., 0 : freq_result.shape[-1]].set(freq_result)
        )

        # Combine with input and return
        return x + freq_result


class TransformerLayer(nn.Module):
    dim: int = struct.field(pytree_node=False)
    heads: int = struct.field(pytree_node=False)
    dropout: float = 0.0

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


@Partial(jax.jit, static_argnums=(0,))
def make_indices(mask_len: int, zero_index: int):
    """Generates increasing indices.

    Args:
        mask_len (int): Length of the mask.
        zero_index (int): Index at which to place 0, before which all values are negative, and after which all values are positive.

    Returns:
        jax.Array: The indices.
    """

    indices = jnp.arange(mask_len) - zero_index
    return indices


@Partial(jax.jit, static_argnums=(0,))
def make_mask(mask_len: int, first_1_index: int):
    """Generates a causal mask.

    Args:
        mask_len (int): Length of the mask.
        first_1_index (int): Index of the first 1, before which all values are 0, and after which all values are 1.

    Returns:
        jax.Array: The causal mask.
    """

    indices = make_indices(mask_len, first_1_index)
    mask = indices >= 0
    return mask


class TransitionModel(nn.Module):
    latent_state_dim: int = struct.field(pytree_node=False)
    n_layers: int = struct.field(pytree_node=False)
    latent_dim: int = struct.field(pytree_node=False)
    heads: int = struct.field(pytree_node=False)

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

        self.state_condenser = nn.Dense(self.latent_state_dim, name="STATE_CONDENSER")

    def __call__(
        self,
        initial_latent_state,
        latent_actions,
        current_action_i,
    ) -> Any:
        state_actions = jax.vmap(
            lambda s, a: jnp.concatenate([s, a]),
            (None, 0),
        )(initial_latent_state, latent_actions)

        # Upscale actions and state to latent dim
        x = jax.vmap(self.state_action_expander.__call__)(state_actions)

        # Apply temporal encodings
        x = self.temporal_encoder(x)

        # Make mask
        mask = make_mask(len(x), current_action_i)

        # Apply transformer layers
        for t_layer in self.t_layers:
            x = t_layer(x, x, mask)

        # Rescale states to original dim
        x = self.state_condenser(x)

        return x


@jdc.pytree_dataclass
class Nets:
    state_encoder: jdc.Static[StateEncoder]
    action_encoder: jdc.Static[ActionEncoder]
    transition_model: jdc.Static[TransitionModel]
    state_decoder: jdc.Static[StateDecoder]
    action_decoder: jdc.Static[ActionDecoder]

    latent_state_radius: float
    latent_action_radius: float

    @property
    def latent_state_dim(self):
        return self.state_encoder.latent_state_dim

    @property
    def latent_action_dim(self):
        return self.action_encoder.latent_action_dim

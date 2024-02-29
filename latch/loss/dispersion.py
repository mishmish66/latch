import dataclasses
from dataclasses import dataclass
from typing import Tuple

import jax
import jax_dataclasses as jdc
from einops import rearrange
from hydra.core.config_store import ConfigStore
from jax import numpy as jnp
from overrides import override

from latch import Infos
from latch.models import ModelState

from .loss_func import WeightedLossFunc
from .loss_registry import register_loss

cs = ConfigStore.instance()


@register_loss("state_dispersion")
@jdc.pytree_dataclass(kw_only=True)
class StateDispersionLoss(WeightedLossFunc):
    # class StateDispersionLoss(SpikeGatedLoss):
    num_samples: jdc.Static[int]

    @override
    def compute_raw(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the dispersion loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): A (b x t-1 x a) array of n actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = rearrange(states, "b t s -> (b t) s")

        latent_states = jax.vmap(models.encode_state)(state=states)

        rng, key = jax.random.split(key)
        sampled_latent_states = jax.random.choice(
            rng, latent_states, shape=(self.num_samples,)
        )

        pairwise_latent_state_diffs = (
            sampled_latent_states[..., None, :] - sampled_latent_states[..., None, :, :]
        )

        pairwise_latent_state_diffs_norm = jnp.linalg.norm(
            pairwise_latent_state_diffs, ord=1, axis=-1
        )

        loss = -jnp.mean(jnp.log(pairwise_latent_state_diffs_norm + 1.0))

        infos = Infos()
        infos = infos.add_info("raw", loss)

        return loss, infos

    @dataclass
    class Config(WeightedLossFunc.Config):
        loss_type: str = "state_dispersion"
        num_samples: int = 16


cs.store(group="loss", name="state_dispersion", node=StateDispersionLoss.Config)


@register_loss("action_dispersion")
@jdc.pytree_dataclass(kw_only=True)
class ActionDispersionLoss(WeightedLossFunc):
    # class ActionDispersionLoss(SigmoidGatedLoss):
    num_samples: jdc.Static[int]

    @override
    def compute_raw(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the dispersion loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): A (b x t x s) array of n states with dim s
            actions (array): A (b x t-1 x a) array of n actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        states = states[..., :-1, :]
        states = rearrange(states, "b t s -> (b t) s")
        actions = rearrange(actions, "b t a -> (b t) a")

        latent_states = jax.vmap(
            models.encode_state,
        )(state=states)
        latent_actions = jax.vmap(
            models.encode_action,
        )(action=actions, latent_state=latent_states)

        rng, key = jax.random.split(key)
        sampled_latent_actions = jax.random.choice(
            rng, latent_actions, shape=(self.num_samples,)
        )

        pairwise_latent_action_diffs = (
            sampled_latent_actions[..., None, :]
            - sampled_latent_actions[..., None, :, :]
        )

        pairwise_latent_action_diffs_norm = jnp.linalg.norm(
            pairwise_latent_action_diffs, ord=1, axis=-1
        )

        loss = -jnp.mean(jnp.log(pairwise_latent_action_diffs_norm + 1.0))

        return loss, Infos()

    @dataclass
    class Config(WeightedLossFunc.Config):
        loss_type: str = "action_dispersion"
        num_samples: int = 16


cs.store(group="loss", name="state_dispersion", node=StateDispersionLoss.Config)

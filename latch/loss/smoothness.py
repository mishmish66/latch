from dataclasses import dataclass
from typing import Tuple

import jax
import jax_dataclasses as jdc
from hydra.core.config_store import ConfigStore
from jax import numpy as jnp
from jax.tree_util import Partial
from overrides import override

from latch import Infos
from latch.models import ModelState

from .loss_func import WeightedLossFunc
from .loss_registry import register_loss

cs = ConfigStore.instance()


@register_loss("smoothness")
@jdc.pytree_dataclass(kw_only=True)
class SmoothnessLoss(WeightedLossFunc):
    # class SmoothnessLoss(SpikeGatedLoss):
    neighborhood_sample_count: jdc.Static[int] = 8

    @override
    def compute_raw(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the smoothness loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x l x s) array of b trajectories of l states with dim s
            actions (array): An (b x l x a) array of b trajectories of l actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        def single_traj_loss_smoothness(
            key,
            states,
            actions,
            start_state_idx,
        ):
            latent_states = jax.vmap(
                models.encode_state,
            )(
                state=states,
            )

            latent_prev_states = latent_states[:-1]

            latent_actions = jax.vmap(
                models.encode_action,
            )(
                action=actions,
                latent_state=latent_prev_states,
            )

            latent_start_state = latent_states[start_state_idx]

            rng, key = jax.random.split(key)
            neighborhood_latent_start_states = models.get_neighborhood_states(
                key=rng,
                latent_state=latent_start_state,
                count=self.neighborhood_sample_count,
            )
            neighborhood_start_states = jax.lax.stop_gradient(
                jax.vmap(models.decode_state)(
                    latent_state=neighborhood_latent_start_states
                )
            )
            neighborhood_latent_start_states = jax.vmap(models.encode_state)(
                state=neighborhood_start_states
            )

            start_action = actions[start_state_idx]

            latent_start_action = models.encode_action(
                action=start_action,
                latent_state=latent_start_state,
            )

            rng, key = jax.random.split(key)
            neighborhood_latent_start_actions = models.get_neighborhood_actions(
                key=rng,
                count=self.neighborhood_sample_count,
                latent_action=latent_start_action,
            )
            neighborhood_start_actions = jax.lax.stop_gradient(
                jax.vmap(models.decode_action)(
                    latent_action=neighborhood_latent_start_actions,
                    latent_state=neighborhood_latent_start_states,
                )
            )
            neighborhood_latent_start_actions = jax.vmap(models.encode_action)(
                action=neighborhood_start_actions,
                latent_state=neighborhood_start_states,
            )

            latent_actions = jnp.repeat(
                latent_actions[None, ...],
                neighborhood_latent_start_actions.shape[0],
                axis=0,
            )
            neighborhood_latent_actions = latent_actions.at[
                ..., start_state_idx, :
            ].set(neighborhood_latent_start_actions)

            neighborhood_next_latent_states_prime = jax.vmap(
                jax.tree_util.Partial(
                    models.infer_states,
                    current_action_i=start_state_idx,
                )
            )(neighborhood_latent_start_states, neighborhood_latent_actions)

            pairwise_neighborhood_state_diffs = (
                neighborhood_next_latent_states_prime[..., None, :]
                - neighborhood_latent_start_states[..., None, :, :]
            )

            pairwise_neighborhood_state_dists = jnp.linalg.norm(
                pairwise_neighborhood_state_diffs, ord=1, axis=-1
            )

            neighborhood_violations = jnp.maximum(
                0.0, pairwise_neighborhood_state_dists - 1.0
            )

            pairwise_sample_diffs = (
                neighborhood_latent_start_states[..., None, :]
                - neighborhood_latent_start_states[..., None, :, :]
            )
            pairwise_sample_dists = jnp.linalg.norm(
                pairwise_sample_diffs, ord=1, axis=-1
            )
            pairwise_sample_dist_geoms = jnp.power(1 / 5, pairwise_sample_dists)

            neighborhood_violation_logs = jnp.log(neighborhood_violations + 1)
            # neighborhood_violation_square = jnp.square(neighborhood_violations)
            closeness_loss = (
                pairwise_sample_dist_geoms[..., None, :] * neighborhood_violation_logs
            )

            total_loss = jnp.mean(closeness_loss)

            return total_loss

        rng, key = jax.random.split(key)
        start_state_idxs = jax.random.randint(
            rng, (len(states),), minval=0, maxval=len(states) - len(states) // 8
        )

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, len(states))
        losses = jax.vmap(
            single_traj_loss_smoothness,
        )(
            key=rngs,
            states=states,
            actions=actions,
            start_state_idx=start_state_idxs,
        )

        loss = jnp.mean(losses)

        return loss, Infos()

    @dataclass
    class Config(WeightedLossFunc.Config):
        loss_type: str = "smoothness"
        neighborhood_sample_count: int = 8


cs.store(group="loss", name="smoothness", node=SmoothnessLoss.Config)

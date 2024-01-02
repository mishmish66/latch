from .loss import SigmoidGatedLoss

from latch.models import ModelState

from latch import Infos

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from overrides import override

from typing import Tuple


@jdc.pytree_dataclass(kw_only=True)
class SmoothnessLoss(SigmoidGatedLoss):
    neighborhood_sample_count: jdc.Static[int] = 8

    @override
    def compute(
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
            latent_start_state = latent_states[start_state_idx]

            rng, key = jax.random.split(key)
            neighborhood_latent_start_states = models.get_neighborhood_states(
                key=rng,
                latent_state=latent_start_state,
                count=self.neighborhood_sample_count,
            )

            latent_actions = jax.vmap(
                models.encode_action,
            )(
                action=actions,
                latent_state=latent_prev_states,
            )

            rng, key = jax.random.split(key)
            rngs = jax.random.split(rng, len(latent_actions))
            neighborhood_latent_actions = jax.vmap(
                Partial(
                    models.get_neighborhood_actions,
                    count=self.neighborhood_sample_count,
                ),
                out_axes=1,
            )(key=rngs, latent_action=latent_actions)

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

            neighborhood_violation_logs = jnp.log(neighborhood_violations + 1e-6)

            total_loss = jnp.mean(neighborhood_violation_logs)

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
        infos = Infos()
        infos = infos.add_info("loss", loss)

        return loss, infos

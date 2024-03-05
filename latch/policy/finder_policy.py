from .optimizer_policy import OptimizerPolicy

from latch import LatchState
from latch.models import make_mask

import jax_dataclasses as jdc

from jax.tree_util import Partial

import jax
from jax import numpy as jnp

from overrides import override


@jdc.pytree_dataclass(kw_only=True)
class FinderPolicy(OptimizerPolicy):
    latent_target: jax.Array

    @override
    def cost_func(
        self,
        key,
        latent_actions,
        latent_start_state,
        train_state: LatchState,
        current_action_i=0,
    ) -> jax.Array:
        latent_states_prime = train_state.target_models.infer_states(
            latent_start_state=latent_start_state,
            latent_actions=latent_actions,
            current_action_i=current_action_i,
        )
        latent_states_prime_err = latent_states_prime - self.latent_target
        latent_states_prime_err_norm = jnp.linalg.norm(
            latent_states_prime_err, ord=1, axis=-1
        )
        causal_mask = make_mask(len(latent_actions), current_action_i)
        future_err_norms = jnp.where(causal_mask, latent_states_prime_err_norm, 0.0)
        future_state_count = jnp.sum(causal_mask)

        return jnp.sum(future_err_norms) / future_state_count  # type: ignore

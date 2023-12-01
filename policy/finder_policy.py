from policy.optimizer_policy import OptimizerPolicy

from learning.train_state import TrainState

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    get_latent_state_prime_gaussians,
    eval_log_gaussian,
    make_mask,
)

from jax.tree_util import register_pytree_node_class, Partial

import jax
from jax import numpy as jnp

from dataclasses import dataclass, replace


# @register_pytree_node_class
@dataclass
class FinderPolicy(OptimizerPolicy):
    @staticmethod
    def cost_func(
        key,
        latent_actions,
        latent_start_state,
        aux,
        train_state: TrainState,
        current_action_i=0,
    ):
        target_state = aux
        rng, key = jax.random.split(key)
        latent_states_prime = infer_states(
            rng, latent_start_state, latent_actions, train_state
        )
        latent_states_prime_err = latent_states_prime - target_state
        latent_states_prime_err_norm = jnp.linalg.norm(
            latent_states_prime_err, ord=1, axis=-1
        )
        causal_mask = make_mask(len(latent_actions), current_action_i)
        future_err_norms = jnp.where(causal_mask, latent_states_prime_err_norm, 0.0)

        return jnp.mean(future_err_norms)

    @staticmethod
    def make_aux(target_state):
        return target_state

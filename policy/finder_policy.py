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
    ):
        target_state = aux
        rng, key = jax.random.split(key)
        latent_state_prime_gaussians = get_latent_state_prime_gaussians(
            latent_start_state, latent_actions, train_state
        )

        log_gauss_vals = jax.vmap(
            jax.tree_util.Partial(eval_log_gaussian, point=target_state)
        )(latent_state_prime_gaussians)

        return jnp.mean(-log_gauss_vals)

    @staticmethod
    def make_aux(target_state):
        return target_state

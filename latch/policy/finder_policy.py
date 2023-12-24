from policy.optimizer_policy import OptimizerPolicy

from learning.train_state import TrainState, NetState, TrainConfig

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
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
        net_state: NetState,
        train_config: TrainConfig,
        current_action_i=0,
    ):
        target_state = aux
        latent_states_prime = infer_states(
            latent_start_state,
            latent_actions,
            net_state=net_state,
            train_config=train_config,
            current_action_i=current_action_i,
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

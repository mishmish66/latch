from policy.optimizer_policy import OptimizerPolicy

from learning.train_state import TrainState, NetState, TrainConfig

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


class ActorPolicy(OptimizerPolicy):
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
        target_q = aux

        def cost_func(state):
            state_cost = jnp.abs(state[0] - target_q)

            return state_cost

        def traj_cost_func(states, actions, current_action_i):
            costs = jax.vmap(cost_func)(states)
            causal_mask = make_mask(len(costs), current_action_i)
            future_costs = jnp.where(causal_mask, costs, 0.0)

            return jnp.mean(future_costs)

        def latent_traj_cost_func(key, latent_states, latent_actions, current_action_i):
            rng, key = jax.random.split(key)
            rngs = jax.random.split(rng, latent_states.shape[0])
            states = jax.vmap(
                jax.tree_util.Partial(
                    decode_state, net_state=net_state, train_config=train_config
                )
            )(
                key=rngs,
                latent_state=latent_states,
            )
            rng, key = jax.random.split(key)
            rngs = jax.random.split(rng, latent_actions.shape[0])
            actions = jax.vmap(
                jax.tree_util.Partial(
                    decode_action, net_state=net_state, train_config=train_config
                )
            )(
                key=rngs,
                latent_action=latent_actions,
                latent_state=latent_states,
            )

            return jnp.sum(traj_cost_func(states, actions, current_action_i))

        rng, key = jax.random.split(key)
        latent_states_prime = infer_states(
            key=rng,
            latent_start_state=latent_start_state,
            latent_actions=latent_actions,
            net_state=net_state,
            train_config=train_config,
            current_action_i=current_action_i,
        )
        latent_states = jnp.concatenate(
            [latent_start_state[None], latent_states_prime], axis=0
        )[:-1]

        return latent_traj_cost_func(
            key, latent_states, latent_actions, current_action_i
        )

    @staticmethod
    def make_aux(target_q):
        return target_q

    @classmethod
    def init(
        cls,
        big_step_size=0.5,
        big_steps=2048,
        small_step_size=0.001,
        small_steps=2048,
        big_post_steps=32,
        small_post_steps=32,
    ):
        policy = super().init(
            big_step_size=big_step_size,
            big_steps=big_steps,
            small_step_size=small_step_size,
            small_steps=small_steps,
            big_post_steps=big_post_steps,
            small_post_steps=small_post_steps,
        )
        return policy

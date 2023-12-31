from policy.optimizer_policy import OptimizerPolicy

from latch.nets import make_mask

from latch import LatchState

from jax.tree_util import Partial

import jax
from jax import numpy as jnp

import jax_dataclasses as jdc

from typing import override


@jdc.pytree_dataclass(kw_only=True)
class ActorPolicy(OptimizerPolicy):
    state_target: float
    state_weights: jax.Array

    def true_space_cost_func(self, state, action):
        return jnp.abs(state - self.state_target) * self.state_weights

    def true_traj_cost_func(self, states, actions, current_action_i):
        costs = jax.vmap(self.true_space_cost_func)(states, actions)
        causal_mask = make_mask(len(costs), current_action_i + 1)
        future_costs = jnp.where(causal_mask, costs, 0.0)
        return jnp.mean(future_costs)

    @override
    def cost_func(
        self,
        key: jax.Array,
        latent_actions: jax.Array,
        latent_start_state: jax.Array,
        train_state: LatchState,
        current_action_i=0,
    ) -> float:
        def latent_traj_cost_func(latent_states, latent_actions, current_action_i):
            states = jax.vmap(train_state.target_net_state.decode_state)(
                latent_state=latent_states
            )
            actions = jax.vmap(train_state.target_net_state.decode_action)(
                latent_action=latent_actions,
                latent_state=latent_states,
            )

            return jnp.sum(
                self.true_traj_cost_func(
                    states,
                    actions,
                    current_action_i,
                )
            ).item()

        latent_states_prime = train_state.target_net_state.infer_states(
            latent_start_state=latent_start_state,
            latent_actions=latent_actions,
            current_action_i=current_action_i,
        )
        latent_states = jnp.concatenate(
            [latent_start_state[None], latent_states_prime], axis=0
        )[:-1]

        return latent_traj_cost_func(latent_states, latent_actions, current_action_i)

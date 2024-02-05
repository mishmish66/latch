import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from jax.tree_util import Partial
from overrides import override

from latch import LatchState
from latch.models import make_mask

from .optimizer_policy import OptimizerPolicy


@jdc.pytree_dataclass(kw_only=True)
class ActorPolicy(OptimizerPolicy):
    state_target: jax.Array
    state_weights: jax.Array

    def true_space_cost_func(self, state, action):
        return jnp.abs(state - self.state_target) * self.state_weights

    def true_traj_cost_func(
        self,
        states: jax.Array,
        actions: jax.Array,
        current_action_i: int,
    ):
        costs = jax.vmap(self.true_space_cost_func)(states, actions)
        causal_mask = make_mask(len(costs), current_action_i + 1)
        future_costs = jnp.where(causal_mask[..., None], costs, 0.0)
        return jnp.mean(future_costs)

    @override
    def cost_func(
        self,
        key: jax.Array,
        latent_actions: jax.Array,
        latent_start_state: jax.Array,
        train_state: LatchState,
        current_action_i=0,
    ) -> jax.Array:
        def latent_traj_cost_func(latent_states, latent_actions, current_action_i):
            states = jax.vmap(train_state.target_models.decode_state)(
                latent_state=latent_states
            )
            actions = jax.vmap(train_state.target_models.decode_action)(
                latent_action=latent_actions,
                latent_state=latent_states,
            )

            return jnp.sum(
                self.true_traj_cost_func(
                    states,
                    actions,
                    current_action_i,
                )
            )

        latent_states_prime = train_state.target_models.infer_states(
            latent_start_state=latent_start_state,
            latent_actions=latent_actions,
            current_action_i=current_action_i,
        )
        latent_states = jnp.concatenate(
            [latent_start_state[None], latent_states_prime], axis=0
        )[:-1]

        return latent_traj_cost_func(latent_states, latent_actions, current_action_i)

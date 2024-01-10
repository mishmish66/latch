from .policy import Policy

from latch import LatchState, LatchConfig, Infos
from latch.models import ModelState, make_mask

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp
from jax.tree_util import Partial

from einops import einsum

from typing import Tuple, TypeVar, Generic, Callable
from abc import ABC, abstractmethod


@Partial(jax.jit, static_argnames=["cost_func", "big_steps", "small_steps"])
def optimize_actions(
    key,
    start_state,
    initial_guess,
    cost_func: Callable[[jax.Array, jax.Array, jax.Array, int], float],
    models: ModelState,
    train_config: LatchConfig,
    start_state_idx=0,
    big_step_size=0.5,
    big_steps=512,
    small_step_size=0.005,
    small_steps=512,
) -> Tuple[jax.Array, Tuple[jax.Array, jax.Array]]:
    """Optimizes a sequence of actions for a given start state and cost function.

    Args:
        key (PRNGKey): A random seed to use for the optimization.
        start_state (array): An (s,) array containing the starting state.
        initial_guess (array): An (l x a) array containing the initial guess for the latent actions over the trajectory.
        cost_func (Callable[[PRNGKey, array, array, int], float]): The cost function to optimize. Has signature (key, latent_actions, latent_start_state) -> scalar.
        train_state (TrainState): The current training state.
        start_state_idx (int, optional): The index of the action corresponding to the start state, the ones before this one might be past actions or otherwise irrelevant data. Defaults to 0.
        big_step_size (float, optional): The size of the big steps. Defaults to 0.5.
        big_steps (int, optional): The number of big steps to take. Defaults to 512.
        small_step_size (float, optional): The size of the small steps. Defaults to 0.005.
        small_steps (int, optional): The number of small steps to take. Defaults to 512.

    Returns:
        (array, (array, array)): A tuple containing the optimized latent actions and a tuple of (costs, big_active_inds) where costs is the cost of each plan during the optimization and big_active_inds is the indices that the algorithm changed during the big steps.
    """
    horizon = train_config.rollout_length

    causal_mask = make_mask(horizon, start_state_idx)

    rng, key = jax.random.split(key)
    latent_start_state = models.encode_state(start_state)

    def big_scanf(current_plan, key):
        rng, key = jax.random.split(key)

        def cost_for_grad(current_plan) -> float:
            return Partial(
                cost_func,
                key=rng,
                latent_start_state=latent_start_state,
                current_action_i=start_state_idx,
            )(latent_actions=current_plan)

        cost, act_grad = jax.value_and_grad(cost_for_grad)(current_plan)

        column_grads = einsum(act_grad, causal_mask, "i ..., i -> i ...")
        column_norms = jnp.linalg.norm(column_grads, ord=1, axis=-1)
        normalized_grad_columns = einsum(
            column_grads, 1 / column_norms, "i ..., i -> i ..."
        )

        new_columns = current_plan - big_step_size * normalized_grad_columns
        new_column_is_in_space = (
            jnp.linalg.norm(new_columns, ord=1, axis=-1)
            < train_config.latent_action_radius
        )
        safe_column_norms = column_norms * new_column_is_in_space

        max_norm = jnp.max(safe_column_norms)
        max_column_idx = jnp.argmax(safe_column_norms)
        new_column, changed_idx = jax.lax.cond(
            max_norm > 0,
            lambda: (new_columns[max_column_idx], max_column_idx),
            lambda: (current_plan[max_column_idx], -1),
        )
        new_column = new_columns[max_column_idx]

        next_plan = current_plan.at[max_column_idx].set(new_column)
        return next_plan, (cost, changed_idx)

    def small_scanf(
        current_plan: jax.Array,
        key: jax.Array,
    ) -> Tuple[jax.Array, float]:
        rng, key = jax.random.split(key)

        def cost_for_grad(current_plan):
            return Partial(
                cost_func,
                key=rng,
                latent_start_state=latent_start_state,
                current_action_i=start_state_idx,
            )(latent_actions=current_plan)

        result: Tuple[float, jax.Array] = jax.value_and_grad(cost_for_grad)(
            current_plan
        )
        (cost, act_grad) = result

        act_grad_future: jax.Array = einsum(act_grad, causal_mask, "i ..., i -> i ...")

        next_plan = current_plan - small_step_size * act_grad_future

        next_plan_norms: jax.Array = jnp.linalg.norm(next_plan, ord=1, axis=-1)
        next_plan_is_in_space = next_plan_norms < train_config.latent_action_radius
        clip_scale = jnp.where(
            next_plan_is_in_space,
            jnp.ones_like(next_plan_norms),
            next_plan_norms / train_config.latent_action_radius,
        )
        next_plan_clipped = next_plan / clip_scale[..., None]  # type: ignore

        next_plan_if_good: jax.Array = jax.lax.cond(
            jnp.any(jnp.isnan(next_plan_clipped)),
            lambda: current_plan,
            lambda: next_plan_clipped,
        )
        return next_plan_if_good, cost

    rng, key = jax.random.split(key)
    scan_rng = jax.random.split(rng, big_steps)
    coarse_latent_action_sequence, (big_costs, big_active_inds) = jax.lax.scan(
        big_scanf, initial_guess, scan_rng
    )

    rng, key = jax.random.split(key)
    scan_rng = jax.random.split(rng, small_steps)
    fine_latent_action_sequence, small_costs = jax.lax.scan(
        small_scanf, coarse_latent_action_sequence, scan_rng
    )

    costs = jnp.concatenate([big_costs, small_costs], axis=0)

    return fine_latent_action_sequence, (costs, big_active_inds)


@jdc.pytree_dataclass(kw_only=True)
class OptimizerPolicy(Policy[jax.Array], ABC):
    big_step_size: float = 0.5
    small_step_size: float = 0.005

    big_steps: jdc.Static[int] = 2048
    small_steps: jdc.Static[int] = 2048

    big_post_steps: jdc.Static[int] = 32
    small_post_steps: jdc.Static[int] = 32

    @abstractmethod
    def cost_func(
        self,
        key: jax.Array,
        latent_actions: jax.Array,
        latent_start_state: jax.Array,
        train_state: LatchState,
        current_action_i=0,
    ) -> jax.Array:
        raise NotImplementedError("Must be implemented by subclass.")

    def make_init_carry(
        self,
        key: jax.Array,
        start_state: jax.Array,
        train_state: LatchState,
    ) -> Tuple[jax.Array, Infos]:
        rng, key = jax.random.split(key)
        random_latent_actions = (
            jax.random.ball(
                rng,
                d=train_state.config.latent_action_dim,
                p=1,
                shape=[train_state.config.rollout_length],
            )
            * train_state.config.latent_action_radius
        )

        cost_func = Partial(self.cost_func, train_state=train_state)

        rng, key = jax.random.split(key)
        optimized_actions, (costs, big_active_inds) = optimize_actions(
            key=rng,
            start_state=start_state,
            initial_guess=random_latent_actions,
            cost_func=cost_func,
            models=train_state.target_models,
            train_config=train_state.config,
            start_state_idx=0,
            big_step_size=self.big_step_size,
            big_steps=self.big_steps,
            small_step_size=self.small_step_size,
            small_steps=self.small_steps,
        )

        infos = Infos()

        infos = infos.add_info("starting expected cost", costs[0])
        infos = infos.add_info("mid expected cost", costs[costs.shape[0] // 2])
        infos = infos.add_info("min expected cost", jnp.min(costs))
        infos = infos.add_info("max expected cost", jnp.max(costs))
        infos = infos.add_info("final expected cost", costs[-1])
        infos = infos.add_info("big active inds", big_active_inds)
        infos = infos.add_info("max cost idx", jnp.argmax(costs))
        infos = infos.add_info("min cost idx", jnp.argmin(costs))

        return optimized_actions, infos

    def __call__(
        self,
        key: jax.Array,
        state: jax.Array,
        i: int,
        carry: jax.Array,
        train_state: LatchState,
    ):
        last_guess = carry

        cost_func = Partial(self.cost_func, train_state=train_state)

        rng, key = jax.random.split(key)
        next_guess, _ = optimize_actions(
            key=rng,
            start_state=state,
            initial_guess=last_guess,
            cost_func=cost_func,
            models=train_state.target_models,
            train_config=train_state.config,
            start_state_idx=i,
            big_steps=self.big_post_steps,
            small_steps=self.small_post_steps,
        )

        latent_action = next_guess[i]
        latent_state = train_state.target_models.encode_state(state)
        action = train_state.target_models.decode_action(latent_action, latent_state)

        return action, next_guess, Infos()

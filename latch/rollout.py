from latch.policy import Policy, ActorPolicy
from latch.models import ModelState
from latch import LatchState, Infos

from einops import rearrange

import jax
from jax import numpy as jnp
from jax.tree_util import Partial


@Partial(jax.jit, static_argnames=("return_dense_states",))
def collect_rollout(
    key: jax.Array,
    start_state: jax.Array,
    policy: Policy,
    train_state: LatchState,
    return_dense_states: bool = False,
):
    """Collects a single rollout of physics data.

    Args:
        key (PRNGKey): Random seed to use to for the rollout.
        start_state (array): a (s,) array of the starting state.
        policy (Policy): The policy to use to collect the rollout.
        train_config (TrainConfig): The training configuration.
        return_dense_states (bool): Whether to return the dense states between substeps for rendering.

    Returns:
        ((array, array), Infos, array): A tuple of a (t x s) array of states and a (t-1 x a) array of actions and the Infos from collecting them and then an (n x d) array of the dense states between substeps for rendering.
    """

    rng, key = jax.random.split(key)
    init_policy_carry, init_policy_info = policy.make_init_carry(
        key=rng,
        start_state=start_state,
        train_state=train_state,
    )

    # Collect a rollout of physics data
    def scanf(carry, key):
        """Scans to collect a single rollout of physics data."""
        state, i, policy_carry = carry

        rng, key = jax.random.split(key)
        action, policy_carry, policy_info = policy(
            key=rng,
            state=state,
            i=i,
            carry=policy_carry,
            train_state=train_state,
        )
        action = jnp.clip(
            action,
            a_min=train_state.config.env.action_bounds[..., 0],
            a_max=train_state.config.env.action_bounds[..., -1],
        )
        dense_states = train_state.config.env.dense_step(state, action)
        next_state = dense_states[-1]

        return (next_state, i + 1, policy_carry), (
            (state, action),
            dense_states,
            policy_info,
        )

    rng, key = jax.random.split(key)
    scan_rngs = jax.random.split(rng, train_state.config.rollout_length - 1)
    _, ((states, actions), dense_states, policy_info) = jax.lax.scan(
        scanf,
        (start_state, 0, init_policy_carry),
        scan_rngs,
    )

    dense_states = rearrange(dense_states, "t u s -> (t u) s")

    dense_states = jnp.concatenate([start_state[None], dense_states])
    states = jnp.concatenate([states, start_state[None]])

    if return_dense_states:
        return (
            (states, actions),
            Infos.merge(policy_info, init_policy_info),
            dense_states,
        )
    else:
        return (states, actions), Infos.merge(policy_info, init_policy_info)


def eval_actor(
    key,
    start_state,
    train_state: LatchState,
    policy: ActorPolicy,
):
    """Evaluates a single rollout of the actor.

    Args:
        key (PRNGKey): The random seed to use for the rollout.
        start_state (array): The starting state.
        train_state (TrainState): The training state.
        policy (OptimizerPolicy): The policy to use to collect the rollout.

    Returns:
        (array, Infos, array): A tuple of a (t x s) array of states, the infos from the eval, and a (t * substep x s) array of dense states.
    """

    rng, key = jax.random.split(key)
    (result_states, result_actions), rollout_infos, dense_states = collect_rollout(  # type: ignore
        key=rng,
        start_state=start_state,
        policy=policy,
        train_state=train_state,
        return_dense_states=True,
    )

    # Repeat final action to match length of states
    # TODO: This is a workaround to make the cost function work. Figure out a more elegant solution.
    result_actions = jnp.concatenate(
        [result_actions, result_actions[-1][None, ...]], axis=0
    )

    rng, key = jax.random.split(key)
    achieved_cost = policy.true_traj_cost_func(result_states, result_actions, 0)
    infos = rollout_infos.add_info("achieved_cost", achieved_cost)

    return result_states, infos, dense_states

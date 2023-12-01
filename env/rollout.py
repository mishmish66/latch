from learning.train_state import NetState, TrainConfig

from policy.policy import Policy


from einops import rearrange

from infos import Infos

import jax
from jax import numpy as jnp


def collect_single_rollout(
    key,
    start_state,
    policy: Policy,
    policy_aux,
    net_state: NetState,
    train_config: TrainConfig,
):
    """Collects a single rollout of physics data.

    Args:
        key (PRNGKey): Random seed to use to for the rollout.
        start_state (array): a (s,) array of the starting state.
        policy (Policy): The policy to use to collect the rollout.
        net_state (NetState): The current network state.
        train_config (TrainConfig): The training configuration.

    Returns:
        ((array, array), Infos, array): A tuple of a (t x s) array of states and a (t-1 x a) array of actions and the Infos from collecting them and then an (n x d) array of the dense states between substeps for rendering.
    """

    rng, key = jax.random.split(key)
    init_policy_carry, init_policy_info = policy.make_init_carry(
        key=rng,
        start_state=start_state,
        aux=policy_aux,
        net_state=net_state,
        train_config=train_config,
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
            net_state=net_state,
            train_config=train_config,
        )
        action = jnp.clip(
            action,
            a_min=train_config.env_config.action_bounds[..., 0],
            a_max=train_config.env_config.action_bounds[..., -1],
        )
        next_state, dense_states = train_config.env_cls.step(
            state, action, train_config.env_config
        )

        return (next_state, i + 1, policy_carry), (
            (state, action),
            dense_states,
            policy_info,
        )

    rng, key = jax.random.split(key)
    scan_rngs = jax.random.split(rng, train_config.rollout_length - 1)
    _, ((states, actions), dense_states, policy_info) = jax.lax.scan(
        scanf,
        (start_state, 0, init_policy_carry),
        scan_rngs,
    )

    dense_states = rearrange(dense_states, "t u s -> (t u) s")

    dense_states = jnp.concatenate([start_state[None], dense_states])
    states = jnp.concatenate([states, start_state[None]])

    return (states, actions), Infos.merge(policy_info, init_policy_info), dense_states


def collect_rollout_batch(
    key,
    start_state,
    policy: Policy,
    policy_auxs,
    net_state: NetState,
    train_config: TrainConfig,
    batch_size,
):
    """Collects a batch of rollouts of physics data.

    Args:
        key (PRNGKey): Random seed to use to for the rollout.
        start_state (array): a (s,) array containing the starting state.
        policies (Policy): A batch of policies to use to collect the rollout.
        train_state (TrainState): The current training state.

    Returns:
        ((array, array), Infos, array): A tuple of a (n x t x s) array of states and a (n x t-1 x a) array of actions and the infos from collecting them and an array of (n x t * substep x s) dense unstrided states.
    """

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, batch_size)
    (states, actions), infos, dense_states = jax.vmap(
        jax.tree_util.Partial(
            collect_single_rollout,
            start_state=start_state,
            net_state=net_state,
            train_config=train_config,
            policy=policy,
        )
    )(
        key=rngs,
        policy_aux=policy_auxs,
    )

    # states = rearrange(states, "t s b -> b t s")
    # actions = rearrange(actions, "t a b -> b t a")
    # dense_states = rearrange(dense_states, "t s b -> b t s")

    return (states, actions), infos, dense_states

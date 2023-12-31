import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

from einops import rearrange

from latch.latch_state import NetState, TrainConfig

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
)

from dataclasses import dataclass

from infos import Infos

from policy.actor_policy import ActorPolicy

from latch.rollout import collect_rollout

from jax.tree_util import Partial


def eval_batch_actor(
    key,
    start_state,
    net_state: NetState,
    train_config: TrainConfig,
    eval_count=64,
    target_q=1.0,
    big_step_size=0.5,
    big_steps=2048,
    small_step_size=0.005,
    small_steps=2048,
    big_post_steps=32,
    small_post_steps=32,
):
    """Evaluates a batch of actors.

    Args:
        key (PRNGKey): The random seed to use for the rollout.
        start_state (array): The starting state.
        net_state (NetState): The network weights.
        train_config (TrainConfig): The training configuration.
        eval_count (int, optional): The number of eval trajectories to collect. Defaults to 64.
        target_q (float, optional): The target angle of the knob to go for. Defaults to 1.0.
        big_step_size (float, optional): The big step size. Defaults to 0.5.
        big_steps (int, optional): The number of initial big steps. Defaults to 2048.
        small_step_size (float, optional): The small step size. Defaults to 0.005.
        small_steps (int, optional): The number of initial small steps. Defaults to 2048.
        big_post_steps (int, optional): The number of big steps to take each env step. Defaults to 32.
        small_post_steps (int, optional): The number of small steps to take each env step. Defaults to 32.

    Returns:
        (array, infos, array): A tuple of a (n x t x s) array of states, the infos from the eval, and a (n x t * substep x s) array of dense states.
    """
    partial_eval_single_actor = Partial(
        eval_single_actor,
        start_state=start_state,
        net_state=net_state,
        train_config=train_config,
        target_q=target_q,
        big_step_size=big_step_size,
        big_steps=big_steps,
        small_step_size=small_step_size,
        small_steps=small_steps,
        big_post_steps=big_post_steps,
        small_post_steps=small_post_steps,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, eval_count)
    states, infos, dense_states = jax.vmap(partial_eval_single_actor, out_axes=0)(
        key=rngs
    )

    return states, infos.condense(method="unstack"), dense_states

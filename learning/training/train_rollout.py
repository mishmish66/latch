from env.env import Env
from env.rollout import collect_rollout_batch

from policy.finder_policy import FinderPolicy
from policy.random_policy import RandomPolicy

from learning.training.train_epoch import train_epoch
from learning.train_state import TrainState

import jax
import jax.numpy as jnp
from jax.tree_util import Partial


def train_rollout(key, train_state: TrainState):
    """Trains the model for a single rollout.

    Args:
        key (PRNGKey): Random seed for the rollout.
        train_state (TrainState): The current training state.
    """

    # Collect rollout data
    rng, key = jax.random.split(key)
    target_states = (
        jax.random.ball(
            key=rng,
            d=train_state.train_config.latent_state_dim,
            p=1,
            shape=[train_state.train_config.traj_per_rollout],
        )
        * train_state.train_config.state_radius
        * 1.5
    )
    policy = FinderPolicy.init()
    policy_auxes = jax.vmap(policy.make_aux)(target_state=target_states)

    rng, key = jax.random.split(key)
    start_state = train_state.train_config.env_cls.init()
    (states, actions), rollout_infos, dense_states = collect_rollout_batch(
        key=rng,
        start_state=start_state,
        policy=policy,
        policy_auxs=policy_auxes,
        train_state=train_state,
        batch_size=train_state.train_config.traj_per_rollout,
    )

    traj_states_for_render = dense_states[0]
    train_state.train_config.env_cls.send_wandb_video(
        name="Rollout Video",
        states=traj_states_for_render,
        env_config=train_state.train_config.env_config,
        step=train_state.step,
    )

    # Comment this out to not clog up the actor info
    # rollout_infos.dump_to_wandb(train_state)

    # Train the model for a bunch of epochs over the rollout data
    def train_epoch_for_scan(train_state, key):
        new_train_state = train_epoch(key, states, actions, train_state)
        return new_train_state, new_train_state

    rng, key = jax.random.split(rng)
    rngs = jax.random.split(key, train_state.train_config.epochs)
    train_state, _ = jax.lax.scan(train_epoch_for_scan, train_state, rngs)

    return train_state

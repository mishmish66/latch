from ..train_state import TrainState, NetState, TrainConfig

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    get_neighborhood_states,
    get_neighborhood_actions,
)

import jax
from jax import numpy as jnp


def loss_smoothness(
    key,
    states,
    actions,
    net_state: NetState,
    train_config: TrainConfig,
    neighborhood_sample_count=8,
):
    """Comutes the smoothness loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (b x l x s) array of b trajectories of l states with dim s
        actions (array): An (b x l x a) array of b trajectories of l actions with dim a
        net_state (NetState): The network weights to use.
        train_config (TrainConfig): The training config.
        neighborhood_sample_count (int, optional): The number of samples to take in the neighborhood. Defaults to 8.

    Returns:
        (scalar, Info): A tuple containing the loss value and associated info object.
    """

    def single_traj_loss_smoothness(
        key,
        states,
        actions,
        start_state_idx,
        neighborhood_sample_count=8,
    ):
        """Computes the smoothness loss for a single trajectory.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (l x s) array of l states with dim s
            actions (array): An (l x a) array of l actions with dim a
            start_state_idx (int): The index of the start state in the trajectory.
            neighborhood_sample_count (int, optional): The number of samples to take in the neighborhood. Defaults to 8.
        """

        latent_states = jax.vmap(
            jax.tree_util.Partial(
                encode_state,
                net_state=net_state,
                train_config=train_config,
            )
        )(
            state=states,
        )

        latent_prev_states = latent_states[:-1]
        latent_next_states = latent_states[1:]
        latent_start_state = latent_states[start_state_idx]

        rng, key = jax.random.split(key)
        neighborhood_latent_start_states = get_neighborhood_states(
            key=rng,
            latent_state=latent_start_state,
            train_config=train_config,
            count=neighborhood_sample_count,
        )

        latent_actions = jax.vmap(
            jax.tree_util.Partial(
                encode_action,
                net_state=net_state,
                train_config=train_config,
            )
        )(
            action=actions,
            latent_state=latent_prev_states,
        )

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, len(latent_actions))
        neighborhood_latent_actions = jax.vmap(
            jax.tree_util.Partial(
                get_neighborhood_actions,
                train_config=train_config,
                count=neighborhood_sample_count,
            ),
            out_axes=1,
        )(key=rngs, latent_action=latent_actions)

        neighborhood_next_latent_states_prime = jax.vmap(
            jax.tree_util.Partial(
                infer_states,
                net_state=net_state,
                train_config=train_config,
                current_action_i=start_state_idx,
            )
        )(neighborhood_latent_start_states, neighborhood_latent_actions)

        pairwise_neighborhood_state_diffs = (
            neighborhood_next_latent_states_prime[..., None, :]
            - neighborhood_latent_start_states[..., None, :, :]
        )

        pairwise_neighborhood_state_dists = jnp.linalg.norm(
            pairwise_neighborhood_state_diffs, ord=1, axis=-1
        )

        neighborhood_violations = jnp.maximum(
            0.0, pairwise_neighborhood_state_dists - 1.0
        )

        neighborhood_violation_logs = jnp.log(neighborhood_violations + 1e-6)

        total_loss = jnp.mean(neighborhood_violation_logs)

        infos = Infos.init()

        return total_loss, infos

    rng, key = jax.random.split(key)
    start_state_idxs = jax.random.randint(
        rng, (len(states),), minval=0, maxval=len(states) - len(states) // 8
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(states))
    losses, infos = jax.vmap(
        jax.tree_util.Partial(
            single_traj_loss_smoothness,
            neighborhood_sample_count=neighborhood_sample_count,
        )
    )(
        key=rngs,
        states=states,
        actions=actions,
        start_state_idx=start_state_idxs,
    )

    condensed_infos = infos.condense()
    condensed_loss = jnp.mean(losses)

    return jnp.zeros_like(condensed_loss), condensed_infos

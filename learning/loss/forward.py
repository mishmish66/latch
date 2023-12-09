from ..train_state import TrainState, NetState, TrainConfig

from infos import Infos

from nets.inference import (
    encode_state,
    encode_action,
    make_mask,
    infer_states,
)

import jax
from jax import numpy as jnp

from einops import einsum


def loss_forward(key, states, actions, net_state: NetState, train_config: TrainConfig):
    """Comutes the forward loss for a set of states and actions.

    Args:
        key (PRNGKey): Random seed to calculate the loss.
        states (array): An (b x l x s) array of b trajectories of l states with dim s
        actions (array): An (b x l x a) array of b trajectories of l actions with dim a
        net_state (NetState): The network weights to use.
        train_config (TrainConfig): The training config.

    Returns:
        (scalar, Info): A tuple containing the loss value and associated info object.
    """

    def single_traj_loss_forward(
        key,
        states,
        actions,
        start_state_idx,
    ):
        """Computes the forward loss for a single trajectory.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (l x s) array of l states with dim s
            actions (array): An (l x a) array of l actions with dim a
            start_state_idx (int): The index of the start state in the trajectory.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
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

        latent_next_state_prime = infer_states(
            latent_start_state=latent_start_state,
            latent_actions=latent_actions,
            net_state=net_state,
            train_config=train_config,
            current_action_i=start_state_idx,
        )

        future_mask = make_mask(len(latent_next_states), start_state_idx)

        errs = jnp.abs(latent_next_states - latent_next_state_prime)
        ln_errs = jnp.log(errs + 1e-8)
        squared_errs = jnp.square(errs)

        losses = ln_errs + squared_errs

        future_losses = einsum(losses, future_mask, "t ..., t -> t ...")
        mean_future_loss = jnp.mean(future_losses)

        infos = Infos.init()

        return mean_future_loss, infos

    rng, key = jax.random.split(key)
    start_state_idxs = jax.random.randint(
        rng, (len(states),), minval=0, maxval=len(states) - len(states) // 8
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, len(states))
    losses, infos = jax.vmap(
        jax.tree_util.Partial(
            single_traj_loss_forward,
        )
    )(
        key=rngs,
        states=states,
        actions=actions,
        start_state_idx=start_state_idxs,
    )

    condensed_infos = infos.condense()
    condensed_loss = jnp.mean(losses)

    return condensed_loss, condensed_infos

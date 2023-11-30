from learning.training.train_step import train_step

from learning.train_state import TrainState

from nets.inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
    sample_gaussian,
    get_neighborhood_states,
    get_neighborhood_actions,
)

from einops import rearrange

import jax
import jax.numpy as jnp


def train_epoch(key, states, actions, train_state: TrainState):
    """Trains the model for a single epoch.

    Args:
        key (PRNGKey): Random seed for the epoch.
        states (array): An (n x t x s) array of n trajectories containing t states with dim s.
        actions (array): An (n x t-1 x s) array of n trajectories containing t-1 actions with dim a.
        train_state (TrainState): The current training state.

    Returns:
        TrainState: The updated training state.
    """

    # # Drop the last state (there is no associated action)
    # states = states[..., :-1, :]

    # Shuffle the data
    rng, key = jax.random.split(key)
    shuffled_indices = jax.random.permutation(rng, jnp.arange(len(states)))
    states = states[shuffled_indices]
    actions = actions[shuffled_indices]

    # Batch the data
    num_batches = len(states) // train_state.train_config.batch_size
    batched_states = rearrange(states, "(b n) t s -> b n t s", b=num_batches)
    batched_actions = rearrange(actions, "(b n) t a -> b n t a", b=num_batches)

    # Define a function to scan over the batches
    def scan_func(carry, batch):
        key, train_state = carry
        batch_states, batch_actions = batch

        # Do the train step
        train_state = train_step(rng, batch_states, batch_actions, train_state)

        return (key, train_state), None

    # Run the scan
    rng, key = jax.random.split(key)
    init_carry = (rng, train_state)
    (_, train_state), _ = jax.lax.scan(
        scan_func, init_carry, (batched_states, batched_actions)
    )

    return train_state

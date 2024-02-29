from latch import LatchState

from einops import rearrange

import jax
import jax.numpy as jnp


def train_epoch(states, actions, train_state: LatchState):
    """Trains the model for a single epoch.

    Args:
        states (array): An (n x t x s) array of n trajectories containing t states with dim s.
        actions (array): An (n x t-1 x s) array of n trajectories containing t-1 actions with dim a.
        train_state (TrainState): The current training state.

    Returns:
        TrainState: The updated training state.
    """

    # Fork out a key from the train_state
    key, train_state = train_state.split_key()

    # Shuffle the data
    rng, key = jax.random.split(key)
    shuffled_indices = jax.random.permutation(rng, jnp.arange(len(states)))
    states = states[shuffled_indices]
    actions = actions[shuffled_indices]

    # Batch the data
    num_batches = len(states) // train_state.config.batch_size
    batched_states = rearrange(states, "(b n) t s -> b n t s", b=num_batches)
    batched_actions = rearrange(actions, "(b n) t a -> b n t a", b=num_batches)

    # Define a function to scan over the batches
    def scan_func(carry: LatchState, batch):
        train_state = carry
        batch_states, batch_actions = batch

        # Fork out a key from the train_state
        rng, train_state = train_state.split_key()

        # Do the train step
        next_train_state, infos = train_state.train_step(
            rng,
            batch_states,
            batch_actions,
        )

        return next_train_state, None

    # Run the scan
    init_carry = train_state
    train_state, _ = jax.lax.scan(  # type: ignore
        scan_func, init_carry, (batched_states, batched_actions)
    )

    return train_state
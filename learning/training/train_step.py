from learning.train_state import TrainState, NetState
from learning.loss.loss import Losses

from nets.inference import (
    encode_state,
    decode_state,
    encode_action,
    decode_action,
)

from infos import Infos

import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten, Partial
from jax.experimental.host_callback import id_tap


from einops import rearrange

import wandb


def train_step(
    states,
    actions,
    train_state: TrainState,
):
    """Train for a single step.

    Args:
        states (array): An (n x t x s) array of n rollouts containing t states with dim s
        actions (array): An (n x t-1 x a) array of n rollouts containing t-1 actions with dim a
        train_state (TrainState): The current training state.

    Returns:
        TrainState: The updated training state.
    """

    # Fork out a key from the train_state
    key, train_state = train_state.split_key()

    # Isolate just the net state from the train state
    def loss_for_grad(primary_net_state: NetState) -> Infos:
        def encode_and_decode(state, action):
            latent_state = train_state.train_config.state_encoder.apply(
                primary_net_state.state_encoder_params, state
            )
            latent_action = train_state.train_config.action_encoder.apply(
                primary_net_state.action_encoder_params, action, latent_state
            )
            reconstructed_state = train_state.train_config.state_decoder.apply(
                primary_net_state.state_decoder_params, latent_state
            )
            reconstructed_action = train_state.train_config.action_decoder.apply(
                primary_net_state.action_decoder_params, latent_action, latent_state
            )
            return reconstructed_state, reconstructed_action

        flat_states = rearrange(states[..., :-1, :], "n t s -> (n t) s")
        flat_actions = rearrange(actions, "n t a -> (n t) a")

        reconstructed_states, reconstructed_actions = jax.vmap(encode_and_decode)(
            flat_states, flat_actions
        )

        state_errs = jnp.abs(flat_states - reconstructed_states)
        action_errs = jnp.abs(flat_actions - reconstructed_actions)

        state_err_sq = jnp.square(state_errs)
        action_err_sq = jnp.square(action_errs)

        state_err_ln = jnp.log(state_errs + 1e-8)
        action_err_ln = jnp.log(action_errs + 1e-8)

        state_loss = jnp.mean(state_err_sq + state_err_ln)
        action_loss = jnp.mean(action_err_sq + action_err_ln)

        infos = Infos.init()
        infos = infos.add_loss_info("state_reconstruction_loss", state_loss)
        infos = infos.add_loss_info("action_reconstruction_loss", action_loss)

        return state_loss + action_loss, infos  # ,  scaled_losses, (loss_gates, infos)

        # return loss, (loss_gates, infos)  # scaled_losses, (loss_gates, infos)

    # Get the gradients
    cumulative_grad, loss_infos = jax.grad(
        loss_for_grad,
        has_aux=True,
    )(train_state.primary_net_state)

    def scale_grad(grad, scale):
        return jax.tree_map(lambda x: x * scale, grad)

    loss_infos.dump_to_console(train_state=train_state)

    train_state = train_state.apply_gradients(cumulative_grad)

    return train_state

from learning.train_state import TrainState, NetState
from learning.loss.loss import Losses

from infos import Infos

import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from jax.experimental.host_callback import id_tap


from einops import rearrange

import wandb


def train_step(
    key,
    states,
    actions,
    train_state: TrainState,
):
    """Train for a single step.

    Args:
        key (PRNGKey): Random seed for the train step.
        states (array): An (n x t x s) array of n rollouts containing t states with dim s
        actions (array): An (n x t-1 x a) array of n rollouts containing t-1 actions with dim a
        train_state (TrainState): The current training state.

    Returns:
        TrainState: The updated training state.
    """

    # Isolate just the net state from the train state
    def loss_for_grad(key, primary_net_state: NetState) -> Infos:
        rng, key = jax.random.split(key)
        losses, infos = Losses.compute(
            rng,
            states=states,
            actions=actions,
            net_state=primary_net_state,
            train_config=train_state.train_config,
        )

        scaled_gated_losses, loss_infos = losses.scale_gate_info(
            train_state.train_config
        )

        infos = Infos.merge(infos, loss_infos)

        return scaled_gated_losses, infos

    # Get the gradients
    rng, key = jax.random.split(key)
    grads_loss_obj, (loss_infos) = jax.jacrev(
        loss_for_grad,
        argnums=1,
        has_aux=True,
    )(rng, train_state.primary_net_state)

    # Find the magnitude of each gradient for logging
    def compute_net_grad_norm(grads_net_obj: NetState):
        flat_grads, _ = tree_flatten(grads_net_obj)
        flat = jnp.concatenate([jnp.ravel(x) for x in flat_grads])
        norm = jnp.linalg.norm(flat)
        return norm

    # Find the proportion of nans for logging
    def compute_nan_proportion(grads_net_obj: NetState):
        flat_grads, _ = tree_flatten(grads_net_obj)
        flat = jnp.concatenate([jnp.ravel(x) for x in flat_grads])
        nan_proportion = jnp.mean(jnp.isnan(flat))
        return nan_proportion

    loss_infos = loss_infos.add_plain_info(
        "reconstruction_grad_norm",
        compute_net_grad_norm(grads_loss_obj.reconstruction_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "reconstruction_nan_portion",
        compute_nan_proportion(grads_loss_obj.reconstruction_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "forward_grad_norm",
        compute_net_grad_norm(grads_loss_obj.forward_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "forward_nan_portion",
        compute_nan_proportion(grads_loss_obj.forward_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "smoothness_grad_norm",
        compute_net_grad_norm(grads_loss_obj.smoothness_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "smoothness_nan_portion",
        compute_nan_proportion(grads_loss_obj.smoothness_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "dispersion_grad_norm",
        compute_net_grad_norm(grads_loss_obj.dispersion_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "dispersion_nan_portion",
        compute_nan_proportion(grads_loss_obj.dispersion_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "condensation_grad_norm",
        compute_net_grad_norm(grads_loss_obj.condensation_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "condensation_nan_portion",
        compute_nan_proportion(grads_loss_obj.condensation_loss),
    )

    cumulative_grad = jax.tree_map(
        lambda *x: jnp.sum(jnp.stack(x), axis=0),
        *grads_loss_obj.to_list(),
    )

    # Find the magnitude of the whole gradient together
    total_grad_norm = compute_net_grad_norm(cumulative_grad)
    total_nan_proportion = compute_nan_proportion(cumulative_grad)

    loss_infos = loss_infos.add_plain_info(
        "state_encoder_grad_nan_proportion",
        compute_nan_proportion(cumulative_grad.state_encoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_encoder_grad_nan_proportion",
        compute_nan_proportion(cumulative_grad.action_encoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "transition_model_grad_nan_proportion",
        compute_nan_proportion(cumulative_grad.transition_model_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "state_decoder_grad_nan_proportion",
        compute_nan_proportion(cumulative_grad.state_decoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_decoder_grad_nan_proportion",
        compute_nan_proportion(cumulative_grad.action_decoder_params),
    )

    loss_infos = loss_infos.add_plain_info("total_grad_norm", total_grad_norm)
    loss_infos = loss_infos.add_plain_info("total_nan_proportion", total_nan_proportion)

    loss_infos.dump_to_wandb(train_state=train_state)

    # This was clogging up the console and slowing down training
    # loss_infos.dump_to_console(train_state=train_state)

    train_state = train_state.apply_gradients(cumulative_grad)

    return train_state

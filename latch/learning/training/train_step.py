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
    def loss_for_grad(key, primary_net_state: NetState) -> Infos:
        rng, key = jax.random.split(key)
        losses, infos = Losses.compute(
            rng,
            states=states,
            actions=actions,
            net_state=primary_net_state,
            train_config=train_state.train_config,
        )

        scaled_losses, loss_gates, loss_infos = losses.scale_gate_info(
            train_state.train_config
        )

        infos = Infos.merge(infos, loss_infos)

        return scaled_losses, (loss_gates, infos)

    # Get the gradients
    rng, key = jax.random.split(key)
    grads_loss_obj, (loss_gates, loss_infos) = jax.jacrev(
        loss_for_grad,
        argnums=1,
        has_aux=True,
    )(rng, train_state.primary_net_state)

    def scale_grad(grad, scale):
        return jax.tree_map(lambda x: x * scale, grad)

    ### Process the gradients ###
    # Extract the transition model grads for the forward loss
    forward_transition_model_grad = grads_loss_obj.forward_loss.transition_model_params
    # Gate the grads
    grads_loss_obj = Losses.from_list(
        [
            scale_grad(loss_grads, loss_gate)
            for loss_grads, loss_gate in zip(
                grads_loss_obj.to_list(), loss_gates.to_list()
            )
        ]
    )
    # Swap back in the ungated transition model grads for the forward loss
    # doing this prevents the transition model from going crazy before the gate kicks in
    swapped_forward_loss_grad = grads_loss_obj.forward_loss.replace(
        transition_model_params=forward_transition_model_grad
    )
    grads_loss_obj = grads_loss_obj.replace(forward_loss=swapped_forward_loss_grad)

    cumulative_grad = jax.tree_map(
        lambda *x: jnp.sum(jnp.stack(x), axis=0),
        *grads_loss_obj.to_list(),
    )

    # Multiply the transition model grads by the transition factor
    scaled_transition_model_grad = scale_grad(
        cumulative_grad.transition_model_params,
        train_state.train_config.transition_factor,
    )
    cumulative_grad = cumulative_grad.replace(
        transition_model_params=scaled_transition_model_grad
    )

    def compute_net_grad_norm(grads_net_obj):
        flat_grads, _ = tree_flatten(grads_net_obj)
        flat = jnp.concatenate([jnp.ravel(x) for x in flat_grads])
        nan_filtered_flat = jnp.nan_to_num(flat)
        norm = jnp.linalg.norm(nan_filtered_flat)
        return norm

    # TODO: Make this a hyperparam or something
    max_grad_norm = 1e4
    max_transition_model_grad_norm = 1e6
    # Here let's clip the gradients for each network
    grad_norms = jnp.array(
        [compute_net_grad_norm(net_grad) for net_grad in cumulative_grad.to_list()]
    )
    grad_factors = jnp.minimum(1, max_grad_norm / grad_norms)

    clipped_grads = NetState.from_list(
        [
            scale_grad(net_grad, grad_factor)
            for grad_factor, net_grad in zip(grad_factors, cumulative_grad.to_list())
        ]
    )

    # For the transition model though, we want to clip it to a different value
    transition_grad_norm = compute_net_grad_norm(
        cumulative_grad.transition_model_params
    )
    transition_grad_factor = jnp.minimum(
        1, max_transition_model_grad_norm / transition_grad_norm
    )
    clipped_grads = clipped_grads.replace(
        transition_model_params=scale_grad(
            cumulative_grad.transition_model_params,
            transition_grad_factor,
        )
    )

    # Find the proportion of nans for logging
    def compute_nan_proportion(grads_net_obj: NetState):
        flat_grads, _ = tree_flatten(grads_net_obj)
        flat = jnp.concatenate([jnp.ravel(x) for x in flat_grads])
        nan_proportion = jnp.mean(jnp.isnan(flat))
        return nan_proportion

    loss_infos = loss_infos.add_plain_info(
        "state_reconstruction_grad_norm",
        compute_net_grad_norm(grads_loss_obj.state_reconstruction_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_reconstruction_grad_norm",
        compute_net_grad_norm(grads_loss_obj.action_reconstruction_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "state_reconstruction_nan_portion",
        compute_nan_proportion(grads_loss_obj.state_reconstruction_loss),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_reconstruction_nan_portion",
        compute_nan_proportion(grads_loss_obj.action_reconstruction_loss),
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

    # Find the magnitude of the whole gradient together
    total_grad_norm = compute_net_grad_norm(clipped_grads)
    total_nan_proportion = compute_nan_proportion(clipped_grads)

    loss_infos = loss_infos.add_plain_info(
        "state_encoder_grad_nan_proportion",
        compute_nan_proportion(clipped_grads.state_encoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_encoder_grad_nan_proportion",
        compute_nan_proportion(clipped_grads.action_encoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "transition_model_grad_nan_proportion",
        compute_nan_proportion(clipped_grads.transition_model_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "state_decoder_grad_nan_proportion",
        compute_nan_proportion(clipped_grads.state_decoder_params),
    )
    loss_infos = loss_infos.add_plain_info(
        "action_decoder_grad_nan_proportion",
        compute_nan_proportion(clipped_grads.action_decoder_params),
    )

    loss_infos = loss_infos.add_plain_info("total_grad_norm", total_grad_norm)
    loss_infos = loss_infos.add_plain_info("total_nan_proportion", total_nan_proportion)

    loss_infos.dump_to_wandb(train_state=train_state)

    # train_state = train_state.apply_gradients(normalized_grads)
    train_state = train_state.apply_gradients(clipped_grads)

    return train_state
from typing import Tuple

import jax
import jax_dataclasses as jdc
from einops import einsum
from jax import numpy as jnp
from overrides import override

from latch import Infos
from latch.models import ModelState, make_mask

from .loss_func import WeightedLossFunc


@jdc.pytree_dataclass(kw_only=True)
class ForwardLoss(WeightedLossFunc):
    # class ForwardLoss(SpikeGatedLoss):
    """Computes the forward loss for a batch of trajectories."""

    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Computes the forward loss for a set of states and actions.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x l x s) array of b trajectories of l states with dim s
            actions (array): An (b x l x a) array of b trajectories of l actions with dim a
            models (ModelState): The models to use.

        Returns:
            (scalar, Info): A tuple containing the loss value and associated info object.
        """

        def single_traj_loss_forward(
            states,
            actions,
            start_state_idx,
        ):
            """Computes the forward loss for a single trajectory.

            Args:
                key (PRNGKey): Random seed to calculate the loss.
                states (array): An (l x s) array of l states with dim s
                actions (array): An (l-1 x a) array of l actions with dim a
                start_state_idx (int): The index of the start state in the trajectory.

            Returns:
                (scalar): The loss value.
            """

            latent_states = jax.vmap(models.encode_state)(state=states)

            latent_prev_states = latent_states[:-1]
            latent_next_states = latent_states[1:]
            latent_start_state = latent_states[start_state_idx]

            latent_actions = jax.vmap(models.encode_action)(
                action=actions,
                latent_state=latent_prev_states,
            )

            latent_next_state_prime = models.infer_states(
                latent_start_state=latent_start_state,
                latent_actions=latent_actions,
                current_action_i=start_state_idx,
            )

            future_mask = make_mask(len(latent_next_states), start_state_idx)

            errs = jnp.abs(latent_next_states - latent_next_state_prime)
            ln_errs = jnp.log(errs + 1e-8)
            squared_errs = jnp.square(errs)

            losses = squared_errs  # + ln_errs

            future_losses = einsum(losses, future_mask, "t ..., t -> t ...")
            mean_future_loss = jnp.mean(future_losses)

            return mean_future_loss

        rng, key = jax.random.split(key)
        start_state_idxs = jax.random.randint(
            rng, (len(states),), minval=0, maxval=len(states) - len(states) // 8
        )

        losses = jax.vmap(
            single_traj_loss_forward,
        )(
            states=states,
            actions=actions,
            start_state_idx=start_state_idxs,
        )

        loss = jnp.mean(losses)

        infos = Infos()
        infos = infos.add_info("raw", loss)

        return loss, infos

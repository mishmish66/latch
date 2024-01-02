# Import all the individual losses that make up the LATCH loss
from .reconstruction import StateReconstructionLoss, ActionReconstructionLoss
from .forward import ForwardLoss
from .condensation import StateCondensationLoss, ActionCondensationLoss
from .smoothness import SmoothnessLoss
from .dispersion import StateDispersionLoss, ActionDispersionLoss

# Import the base loss class
from .loss import Loss

from latch.infos import Infos
from latch.models import ModelState

import jax_dataclasses as jdc

import jax

from overrides import override

from typing import Tuple


@jdc.pytree_dataclass
class LatchLoss(Loss):
    state_reconstruction_loss: StateReconstructionLoss = StateReconstructionLoss()
    action_reconstruction_loss: ActionReconstructionLoss = ActionReconstructionLoss()

    forward_loss: ForwardLoss = ForwardLoss(weight=1.0, sharpness=1.0, center=-3.0)

    state_condensation_loss: StateCondensationLoss = StateCondensationLoss(
        weight=1.0, sharpness=1.0, center=-5.0
    )
    action_condensation_loss: ActionCondensationLoss = ActionCondensationLoss(
        weight=1.0, sharpness=1.0, center=-5.0
    )

    smoothness_loss: SmoothnessLoss = SmoothnessLoss(
        weight=1.0, sharpness=1.0, center=-5.0
    )

    state_dispersion_loss: StateDispersionLoss = StateDispersionLoss(
        weight=1.0, sharpness=1.0, center=-5.0, num_samples=16
    )
    action_dispersion_loss: ActionDispersionLoss = ActionDispersionLoss(
        weight=1.0, sharpness=1.0, center=-5.0, num_samples=16
    )

    @override
    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[jax.Array, Infos]:
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (jax.Array): An (b x t x s) array of b trajectories of l states with dim s
            actions (jax.Array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            models (ModelState): The models to use.

        Returns:
            (Losses, Infos): A tuple containing the loss object and associated infos object.
        """

        infos = Infos()

        ### Reconstruction Loss ###

        loss_args = {
            "states": states,
            "actions": actions,
            "models": models,
        }
        rng, key = jax.random.split(key)
        (
            state_reconstruction_loss,
            state_reconstruction_infos,
        ) = self.state_reconstruction_loss.compute(
            key=rng,
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_reconstruction_loss,
            action_reconstruction_infos,
        ) = self.action_reconstruction_loss.compute(
            key=rng,
            **loss_args,
        )

        reconstruction_loss = state_reconstruction_loss + action_reconstruction_loss

        reconstruction_infos = Infos()
        reconstruction_infos = reconstruction_infos.add_info(
            "state", state_reconstruction_infos
        )
        reconstruction_infos = reconstruction_infos.add_info(
            "action", action_reconstruction_infos
        )

        infos = infos.add_info("reconstruction", reconstruction_infos)

        ### Forward Loss ###

        forward_gate_in = reconstruction_loss
        rng, key = jax.random.split(key)
        (
            forward_loss,
            forward_infos,
        ) = self.forward_loss(
            key=rng,
            gate_in=forward_gate_in,  # type: ignore
            **loss_args,
        )

        infos = infos.add_info("forward", forward_infos)

        ### Condensation Loss ###

        condensation_gate_in = forward_loss + reconstruction_loss

        rng, key = jax.random.split(key)
        (
            state_condensation_loss,
            state_condensation_info,
        ) = self.state_condensation_loss(
            key=rng,
            gate_in=condensation_gate_in,  # type: ignore
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_condensation_loss,
            action_condensation_info,
        ) = self.action_condensation_loss(
            key=rng,
            gate_in=condensation_gate_in,  # type: ignore
            **loss_args,
        )

        condensation_loss = state_condensation_loss + action_condensation_loss

        condensation_infos = Infos()
        condensation_infos = condensation_infos.add_info(
            "state", state_condensation_info
        )
        condensation_infos = condensation_infos.add_info(
            "action", action_condensation_info
        )

        infos = infos.add_info("condensation", condensation_infos)

        ### Smoothness Loss ###

        smoothness_gate_in = condensation_loss + forward_loss + reconstruction_loss
        rng, key = jax.random.split(key)
        (
            smoothness_loss,
            smoothness_info,
        ) = self.smoothness_loss(
            key=rng,
            gate_in=smoothness_gate_in,  # type: ignore
            **loss_args,
        )

        infos = infos.add_info("smoothness", smoothness_info)

        ### Dispersion Loss ###
        dispersion_gate_in = condensation_loss + forward_loss + reconstruction_loss

        rng, key = jax.random.split(key)
        (
            state_dispersion_loss,
            state_dispersion_info,
        ) = self.state_dispersion_loss(
            key=rng,
            gate_in=dispersion_gate_in,  # type: ignore
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_dispersion_loss,
            action_dispersion_info,
        ) = self.action_dispersion_loss(
            key=rng,
            gate_in=dispersion_gate_in,  # type: ignore
            **loss_args,
        )

        dispersion_loss = state_dispersion_loss + action_dispersion_loss

        dispersion_infos = Infos()
        dispersion_infos = dispersion_infos.add_info("state", state_dispersion_info)
        dispersion_infos = dispersion_infos.add_info("action", action_dispersion_info)

        infos = infos.add_info("dispersion", dispersion_infos)

        ### Total Loss ###
        total_loss = (
            reconstruction_loss
            + forward_loss
            + condensation_loss
            + smoothness_loss
            + dispersion_loss
        )

        return total_loss, infos

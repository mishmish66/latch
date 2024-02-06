from typing import Any, Dict, Tuple

import jax
import jax_dataclasses as jdc
from loss_gate_graph import LossGateGraph

from latch.infos import Infos
from latch.models import ModelState

from .condensation import ActionCondensationLoss, StateCondensationLoss
from .dispersion import ActionDispersionLoss, StateDispersionLoss
from .forward import ForwardLoss
from .reconstruction import ActionReconstructionLoss, StateReconstructionLoss
from .smoothness import SmoothnessLoss


@jdc.pytree_dataclass
class LatchLoss:
    loss_gate_graph: jdc.Static[LossGateGraph]

    state_reconstruction_loss: StateReconstructionLoss
    action_reconstruction_loss: ActionReconstructionLoss

    forward_loss: ForwardLoss

    state_condensation_loss: StateCondensationLoss
    action_condensation_loss: ActionCondensationLoss

    smoothness_loss: SmoothnessLoss

    state_dispersion_loss: StateDispersionLoss
    action_dispersion_loss: ActionDispersionLoss

    def compute(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
        models: ModelState,
    ) -> Tuple[Dict[Any, jax.Array], Infos]:
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (jax.Array): An (b x t x s) array of b trajectories of l states with dim s
            actions (jax.Array): An (b x t-1 x a) array of b trajectories of l-1 actions with dim a
            models (ModelState): The models to use.

        Returns:
            (Losses, Infos): A tuple containing the loss object and associated infos object.
        """

        loss_vals = {}

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
        ) = self.state_reconstruction_loss(
            key=rng,
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_reconstruction_loss,
            action_reconstruction_infos,
        ) = self.action_reconstruction_loss(
            key=rng,
            **loss_args,
        )

        loss_vals[self.state_reconstruction_loss.__class__] = state_reconstruction_loss
        loss_vals[self.action_reconstruction_loss.__class__] = (
            action_reconstruction_loss
        )

        ### Forward Loss ###

        rng, key = jax.random.split(key)
        (
            forward_loss,
            forward_infos,
        ) = self.forward_loss(
            key=rng,
            **loss_args,
        )

        loss_vals[self.forward_loss.__class__] = forward_loss

        ### Condensation Loss ###

        rng, key = jax.random.split(key)
        (
            state_condensation_loss,
            state_condensation_infos,
        ) = self.state_condensation_loss(
            key=rng,
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_condensation_loss,
            action_condensation_infos,
        ) = self.action_condensation_loss(
            key=rng,
            **loss_args,
        )

        loss_vals[self.state_condensation_loss.__class__] = state_condensation_loss
        loss_vals[self.action_condensation_loss.__class__] = action_condensation_loss

        ### Smoothness Loss ###

        rng, key = jax.random.split(key)
        (
            smoothness_loss,
            smoothness_infos,
        ) = self.smoothness_loss(
            key=rng,
            **loss_args,
        )

        loss_vals[self.smoothness_loss.__class__] = smoothness_loss

        ### Dispersion Loss ###

        rng, key = jax.random.split(key)
        (
            state_dispersion_loss,
            state_dispersion_infos,
        ) = self.state_dispersion_loss(
            key=rng,
            **loss_args,
        )

        rng, key = jax.random.split(key)
        (
            action_dispersion_loss,
            action_dispersion_infos,
        ) = self.action_dispersion_loss(
            key=rng,
            **loss_args,
        )

        loss_vals[self.state_dispersion_loss.__class__] = state_dispersion_loss
        loss_vals[self.action_dispersion_loss.__class__] = action_dispersion_loss

        ### Loss Gating ###

        infos = Infos()
        gate_vals = self.loss_gate_graph.compute_gates(loss_vals)

        ### Gated Reconstruction Loss ###

        state_reconstruction_gate_val = gate_vals[
            self.state_reconstruction_loss.__class__
        ]
        gated_state_reconstruction_loss = (
            state_reconstruction_loss * state_reconstruction_gate_val
        )
        action_reconstruction_gate_val = gate_vals[
            self.action_reconstruction_loss.__class__
        ]
        gated_action_reconstruction_loss = (
            action_reconstruction_loss * action_reconstruction_gate_val
        )

        state_reconstruction_infos = state_reconstruction_infos.add_info(
            "gate", state_reconstruction_gate_val
        ).add_info("final", gated_state_reconstruction_loss)
        action_reconstruction_infos = action_reconstruction_infos.add_info(
            "gate", action_reconstruction_gate_val
        ).add_info("final", gated_action_reconstruction_loss)

        reconstruction_infos = Infos()
        reconstruction_infos = reconstruction_infos.add_info(
            "state", state_reconstruction_infos
        )
        reconstruction_infos = reconstruction_infos.add_info(
            "action", action_reconstruction_infos
        )

        infos = infos.add_info("reconstruction", reconstruction_infos)

        final_reconstruction_loss = (
            gated_action_reconstruction_loss + gated_state_reconstruction_loss
        )

        ### Gated Forward Loss ###

        forward_gate_val = gate_vals[self.forward_loss.__class__]
        gated_forward_loss = forward_loss * forward_gate_val

        forward_infos = Infos()
        forward_infos = forward_infos.add_info("gate", forward_gate_val).add_info(
            "final", gated_forward_loss
        )
        infos = infos.add_info("forward", forward_infos)

        final_forward_loss = gated_forward_loss

        ### Gated Condensation Loss ###

        state_condensation_gate_val = gate_vals[self.state_condensation_loss.__class__]
        gated_state_condensation_loss = (
            state_condensation_loss * state_condensation_gate_val
        )
        action_condensation_gate_val = gate_vals[
            self.action_condensation_loss.__class__
        ]
        gated_action_condensation_loss = (
            action_condensation_loss * action_condensation_gate_val
        )

        state_condensation_infos = state_condensation_infos.add_info(
            "gate", state_condensation_gate_val
        ).add_info("final", gated_state_condensation_loss)
        action_condensation_infos = action_condensation_infos.add_info(
            "gate", action_condensation_gate_val
        ).add_info("final", gated_action_condensation_loss)

        condensation_infos = Infos()
        condensation_infos = condensation_infos.add_info(
            "state", state_condensation_infos
        )
        condensation_infos = condensation_infos.add_info(
            "action", action_condensation_infos
        )
        infos = infos.add_info("condensation", condensation_infos)

        final_condensation_loss = (
            gated_state_condensation_loss + gated_action_condensation_loss
        )

        ### Gated Smoothness Loss ###

        smoothness_gate_val = gate_vals[self.smoothness_loss.__class__]
        gated_smoothness_loss = smoothness_loss * smoothness_gate_val

        smoothness_infos = smoothness_infos.add_info(
            "gate", smoothness_gate_val
        ).add_info("final", gated_smoothness_loss)
        infos = infos.add_info("smoothness", smoothness_infos)

        final_smoothness_loss = gated_smoothness_loss

        ### Gated Dispersion Loss ###

        state_dispersion_gate_val = gate_vals[self.state_dispersion_loss.__class__]
        gated_state_dispersion_loss = state_dispersion_loss * state_dispersion_gate_val
        action_dispersion_gate_val = gate_vals[self.action_dispersion_loss.__class__]
        gated_action_dispersion_loss = (
            action_dispersion_loss * action_dispersion_gate_val
        )

        state_dispersion_infos = state_dispersion_infos.add_info(
            "gate", state_dispersion_gate_val
        ).add_info("final", gated_state_dispersion_loss)
        action_dispersion_infos = action_dispersion_infos.add_info(
            "gate", action_dispersion_gate_val
        ).add_info("final", gated_action_dispersion_loss)

        dispersion_infos = Infos()
        dispersion_infos = dispersion_infos.add_info("state", state_dispersion_infos)
        dispersion_infos = dispersion_infos.add_info("action", action_dispersion_infos)
        infos = infos.add_info("dispersion", dispersion_infos)

        final_dispersion_loss = (
            gated_state_dispersion_loss + gated_action_dispersion_loss
        )

        ### Total Loss ###
        total_loss = (
            final_reconstruction_loss
            + final_forward_loss
            + final_condensation_loss
            + final_smoothness_loss
            + final_dispersion_loss
        )

        infos = infos.add_info("total", total_loss)

        return total_loss, infos

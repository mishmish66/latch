from typing import Any, Dict, Tuple, List

import jax
import jax_dataclasses as jdc
from jax import numpy as jnp
from latch.loss.loss_graph import LossGateGraph

from latch.infos import Infos
from latch.models import ModelState

from .loss_func import LossFunc
from .condensation import ActionCondensationLoss, StateCondensationLoss
from .dispersion import ActionDispersionLoss, StateDispersionLoss
from .forward import ForwardLoss
from .reconstruction import ActionReconstructionLoss, StateReconstructionLoss
from .smoothness import SmoothnessLoss


@jdc.pytree_dataclass
class LatchLoss:

    loss_list: jdc.Static[List[LossFunc]]
    loss_gate_graph: jdc.Static[LossGateGraph]

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
            (Losses, Infos): A tuple containing the loss and associated infos object.
        """

        # Create a dictionary of loss values
        raw_loss_info_vals: Dict[str, Tuple[jax.Array, Infos]] = {}
        for loss in self.loss_list:
            rng, key = jax.random.split(key)
            raw_loss, info = loss.compute_raw(rng, states, actions, models)
            raw_loss_info_vals[loss.name] = raw_loss, info

        # Compute the gate values for the graph
        raw_loss_vals = {
            loss_name: raw_loss
            for loss_name, (raw_loss, info) in raw_loss_info_vals.items()
        }
        gate_vals = self.loss_gate_graph.forward(raw_loss_vals)

        final_vals_infos = {
            loss_name: (
                raw_loss * gate_vals[loss_name],
                info.add_info("gate", gate_vals[loss_name]),
            )
            for loss_name, (raw_loss, infos) in raw_loss_info_vals.items()
        }

        # Merge the infos
        infos = Infos()
        for loss_name, (loss_val, loss_infos) in final_vals_infos.items():
            infos = infos.add_info(loss_name, loss_infos.add_info("final", loss_val))

        # Compute the final loss

        ### Total Loss ###
        loss_array = jnp.array(
            [loss_val for loss_val, info in final_vals_infos.values()]
        )
        total_loss = jnp.sum(loss_array)
        infos = infos.add_info("total", total_loss)

        return total_loss, infos

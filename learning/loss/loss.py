from .condensation import loss_condensation
from .dispersion import loss_dispersion
from .forward import loss_forward
from .reconstruction import loss_reconstruction
from .smoothness import loss_smoothness

from learning.train_state import NetState, TrainConfig

from infos import Infos

from einops import rearrange

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from dataclasses import dataclass


@register_pytree_node_class
@dataclass
class Losses:
    state_reconstruction_loss: any
    action_reconstruction_loss: any
    forward_loss: any
    smoothness_loss: any
    dispersion_loss: any
    condensation_loss: any

    @classmethod
    def init(
        cls,
        state_reconstruction_loss=0,
        action_reconstruction_loss=0,
        forward_loss=0,
        smoothness_loss=0,
        dispersion_loss=0,
        condensation_loss=0,
    ):
        return cls(
            state_reconstruction_loss=state_reconstruction_loss,
            action_reconstruction_loss=action_reconstruction_loss,
            forward_loss=forward_loss,
            smoothness_loss=smoothness_loss,
            dispersion_loss=dispersion_loss,
            condensation_loss=condensation_loss,
        )

    @classmethod
    def compute(
        cls, key, states, actions, net_state: NetState, train_config: TrainConfig
    ):
        """Compute the loss for a batch of trajectories.

        Args:
            key (PRNGKey): Random seed to calculate the loss.
            states (array): An (b x l x s) array of b trajectories of l states with dim s
            actions (array): An (b x l-1 x a) array of b trajectories of l-1 actions with dim a
            net_state (NetState): The network weights.
            train_config (TrainConfig): The training configuration.

        Returns:
            (Losses, Infos): A tuple containing the loss object and associated infos object.
        """

        forward_loss, forward_info = loss_forward(
            key=key,
            states=states,
            actions=actions,
            net_state=net_state,
            train_config=train_config,
        )
        smoothness_loss, smoothness_info = loss_smoothness(
            key=key,
            states=states,
            actions=actions,
            net_state=net_state,
            train_config=train_config,
        )

        # Drop the last state to make the states and actions the same length
        states_last_dropped = states[..., :-1, :]
        flat_states = rearrange(states_last_dropped, "b l s -> (b l) s")
        flat_actions = rearrange(actions, "b l a -> (b l) a")

        condensation_loss, condensation_info = loss_condensation(
            key=key,
            states=flat_states,
            actions=flat_actions,
            net_state=net_state,
            train_config=train_config,
        )
        dispersion_loss, dispersion_info = loss_dispersion(
            key=key,
            states=flat_states,
            actions=flat_actions,
            net_state=net_state,
            train_config=train_config,
        )
        (
            state_reconstruction_loss,
            action_reconstruction_loss,
        ), reconstruction_info = loss_reconstruction(
            key=key,
            states=flat_states,
            actions=flat_actions,
            net_state=net_state,
            train_config=train_config,
        )

        losses = Losses.init(
            state_reconstruction_loss=state_reconstruction_loss,
            action_reconstruction_loss=action_reconstruction_loss,
            forward_loss=forward_loss,
            smoothness_loss=smoothness_loss,
            dispersion_loss=dispersion_loss,
            condensation_loss=condensation_loss,
        )

        infos = Infos.merge(
            condensation_info,
            dispersion_info,
            forward_info,
            smoothness_info,
            reconstruction_info,
        )

        return losses, infos

    def tree_flatten(self):
        return [
            self.state_reconstruction_loss,
            self.action_reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls.init(
            state_reconstruction_loss=data[0],
            action_reconstruction_loss=data[1],
            forward_loss=data[2],
            smoothness_loss=data[3],
            dispersion_loss=data[4],
            condensation_loss=data[5],
        )

    @classmethod
    def merge(cls, *losses):
        return cls.init(
            state_reconstruction_loss=jnp.sum(
                [l.state_reconstruction_loss for l in losses]
            ),
            action_reconstruction_loss=jnp.sum(
                [l.action_reconstruction_loss for l in losses]
            ),
            forward_loss=jnp.sum([l.forward_loss for l in losses]),
            smoothness_loss=jnp.sum([l.smoothness_loss for l in losses]),
            dispersion_loss=jnp.sum([l.dispersion_loss for l in losses]),
            condensation_loss=jnp.sum([l.condensation_loss for l in losses]),
        )

    def scale_gate_info(self, train_config: TrainConfig):
        infos = Infos.init()

        forward_gate = make_gate_value(
            self.state_reconstruction_loss,
            train_config.forward_gate_sharpness,
            train_config.forward_gate_center,
        ) * make_gate_value(
            self.action_reconstruction_loss,
            train_config.forward_gate_sharpness,
            train_config.forward_gate_center,
        )
        condensation_gate = (
            make_gate_value(
                self.forward_loss,
                train_config.condensation_gate_sharpness,
                train_config.condensation_gate_center,
            )
            * forward_gate
        )
        smoothness_gate = (
            make_gate_value(
                self.condensation_loss,
                train_config.smoothness_gate_sharpness,
                train_config.smoothness_gate_center,
            )
            * forward_gate
            * condensation_gate
        )
        dispersion_gate = (
            make_gate_value(
                self.condensation_loss,
                train_config.dispersion_gate_sharpness,
                train_config.dispersion_gate_center,
            )
            * forward_gate
            * condensation_gate
        )

        scaled_state_reconstruction_loss = (
            self.state_reconstruction_loss * train_config.reconstruction_weight
        )
        scaled_action_reconstruction_loss = (
            self.action_reconstruction_loss * train_config.reconstruction_weight
        )
        scaled_forward_loss = self.forward_loss * train_config.forward_weight
        scaled_smoothness_loss = self.smoothness_loss * train_config.smoothness_weight
        scaled_dispersion_loss = self.dispersion_loss * train_config.dispersion_weight
        scaled_condensation_loss = (
            self.condensation_loss * train_config.condensation_weight
        )

        total_loss = (
            scaled_state_reconstruction_loss
            + scaled_action_reconstruction_loss
            + scaled_forward_loss
            + scaled_smoothness_loss
            + scaled_dispersion_loss
            + scaled_condensation_loss
        )

        infos = infos.add_loss_info("total_loss", total_loss)

        infos = infos.add_loss_info(
            "state_reconstruction_loss", self.state_reconstruction_loss
        )
        infos = infos.add_loss_info(
            "action_reconstruction_loss", self.action_reconstruction_loss
        )
        infos = infos.add_loss_info("forward_loss", self.forward_loss)
        infos = infos.add_loss_info("smoothness_loss", self.smoothness_loss)
        infos = infos.add_loss_info("dispersion_loss", self.dispersion_loss)
        infos = infos.add_loss_info("condensation_loss", self.condensation_loss)

        infos = infos.add_plain_info("forward_gate", forward_gate)
        infos = infos.add_plain_info("smoothness_gate", smoothness_gate)
        infos = infos.add_plain_info("dispersion_gate", dispersion_gate)
        infos = infos.add_plain_info("condensation_gate", condensation_gate)

        result_loss = Losses.init(
            state_reconstruction_loss=scaled_state_reconstruction_loss,
            action_reconstruction_loss=scaled_action_reconstruction_loss,
            forward_loss=scaled_forward_loss,
            smoothness_loss=scaled_smoothness_loss,
            dispersion_loss=scaled_dispersion_loss,
            condensation_loss=scaled_condensation_loss,
        )

        result_gates = Losses.init(
            state_reconstruction_loss=1.0,
            action_reconstruction_loss=1.0,
            forward_loss=forward_gate,
            smoothness_loss=smoothness_gate,
            dispersion_loss=dispersion_gate,
            condensation_loss=condensation_gate,
        )

        return result_loss, result_gates, infos

    def to_list(self):
        return [
            self.state_reconstruction_loss,
            self.action_reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
        ]

    def replace(self, **kwargs):
        return Losses.init(
            state_reconstruction_loss=kwargs.get(
                "state_reconstruction_loss", self.state_reconstruction_loss
            ),
            action_reconstruction_loss=kwargs.get(
                "action_reconstruction_loss", self.action_reconstruction_loss
            ),
            forward_loss=kwargs.get("forward_loss", self.forward_loss),
            smoothness_loss=kwargs.get("smoothness_loss", self.smoothness_loss),
            dispersion_loss=kwargs.get("dispersion_loss", self.dispersion_loss),
            condensation_loss=kwargs.get("condensation_loss", self.condensation_loss),
        )

    @classmethod
    def from_list(cls, self):
        return cls.init(
            state_reconstruction_loss=self[0],
            action_reconstruction_loss=self[1],
            forward_loss=self[2],
            smoothness_loss=self[3],
            dispersion_loss=self[4],
            condensation_loss=self[5],
        )

    def total(self):
        return jnp.sum(jnp.array(self.to_list()))


def make_gate_value(x, sharpness, center):
    sgx = jax.lax.stop_gradient(x)
    return (1 + jnp.exp(sharpness * (sgx - center))) ** -1


def make_log_gate_value(x, sharpness, center):
    sgx = jax.lax.stop_gradient(x)
    base = sgx / center
    den = 1 + base**sharpness
    return 1 / den

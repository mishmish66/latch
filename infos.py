from learning.train_state import TrainState

import jax

from jax import numpy as jnp

from jax.tree_util import register_pytree_node_class, Partial
from jax.experimental.host_callback import id_tap

import wandb

from einops import rearrange

from dataclasses import dataclass, field


@register_pytree_node_class
@dataclass
class Infos:
    loss_infos: any = field(default_factory=dict)
    plain_infos: any = field(default_factory=dict)

    @classmethod
    def init(cls, loss_infos={}, plain_infos={}):
        """Initializes an infos object.

        Args:
            loss_infos (dict, optional): The infos about losses. Defaults to {}.
            plain_infos (dict, optional): General infos to log. Defaults to {}.

        Returns:
            Infos: The new infos object.
        """

        return cls(
            loss_infos=loss_infos,
            plain_infos=plain_infos,
        )

    @classmethod
    def merge(cls, *infs) -> "Infos":
        """Merges multiple infos objects into one.

        Returns:
            Infos: The meged infos object.
        """

        return cls.init(
            loss_infos={k: v for inf in infs for k, v in inf.loss_infos.items()},
            plain_infos={k: v for inf in infs for k, v in inf.plain_infos.items()},
        )

    def add_loss_info(self, name, value):
        """Adds a loss info to the infos object.

        Args:
            name (str): The name of the info.
            value (Any): The value of the info.

        Returns:
            Infos: The updated infos object.
        """

        return Infos.init(
            loss_infos={**self.loss_infos, name: value},
            plain_infos=self.plain_infos,
        )

    def add_plain_info(self, name, value):
        """Adds a plain info to the infos object.

        Args:
            name (str): The name of the info.
            value (Any): The value of the info.

        Returns:
            Infos: The updated infos object.
        """

        return Infos.init(
            loss_infos=self.loss_infos,
            plain_infos={**self.plain_infos, name: value},
        )

    def tree_flatten(self):
        loss_info_names = list(self.loss_infos.keys())
        loss_info_values = list(self.loss_infos.values())

        plain_info_names = list(self.plain_infos.keys())
        plain_info_values = list(self.plain_infos.values())

        return (
            loss_info_values,
            plain_info_values,
        ), (
            loss_info_names,
            plain_info_names,
        )

    @classmethod
    def tree_unflatten(cls, aux, data):
        loss_info_names, plain_info_names = aux
        loss_info_values, plain_info_values = data

        loss_infos = {
            name: value for name, value in zip(loss_info_names, loss_info_values)
        }
        plain_infos = {
            name: value for name, value in zip(plain_info_names, plain_info_values)
        }
        return cls.init(
            loss_infos=loss_infos,
            plain_infos=plain_infos,
        )

    def host_get_dict(self, prefix=None):
        """Turns the Infos object into a single dict"""

        result_dict = {
            **self.plain_infos,
            **self.loss_infos,
        }

        if prefix is not None:
            result_dict = {f"{prefix}/{k}": v for k, v in result_dict.items()}

        return result_dict

    def condense(self, method="mean"):
        """Condenses the infos object along the zero axis.

        Args:
            method (str, optional): Either "mean" or "unstack" which will either flatten that axis or mean across it. Defaults to "mean".

        Returns:
            Infos: The condensed infos object.
        """

        condenser = None
        if method == "mean":
            condenser = lambda x: jnp.mean(x, axis=0)
        if method == "unstack":
            # Let's get the shape without the axis
            condenser = lambda x: jnp.reshape(x, (-1, *x.shape[2:]))
        return Infos.init(
            loss_infos=jax.tree_map(
                condenser,
                self.loss_infos,
            ),
            plain_infos=jax.tree_map(
                condenser,
                self.plain_infos,
            ),
        )

    def host_dump_to_wandb(self, step=None, prefix=None):
        """Dumps the infos object to wandb (to be called from the host)."""
        if step is not None:
            print(f"Logging 🪵 for step {step}")
            wandb.log(self.host_get_dict(prefix), step=step)
        else:
            wandb.log(self.host_get_dict(prefix))

    @Partial(jax.jit, static_argnames=["prefix"])
    def dump_to_wandb(self, train_state: TrainState, prefix=None):
        """Dumps the infos object to wandb (to be called from the device)."""
        step = train_state.step

        def dump_to_wandb_for_tap(tap_pack, transforms):
            self, step = tap_pack
            Infos.host_dump_to_wandb(self, step, prefix)

        id_tap(
            dump_to_wandb_for_tap,
            (self, step),
        )

    def host_get_str(self, step=None):
        """Gets a string representation of the infos object (to be called from the host)."""
        if step is not None:
            return f"Step {step}:" + self.host_get_str()

        loss_msg = "\nLosses:" + "".join(
            [f"\n\t\t{name}: {value}" for name, value in self.loss_infos.items()]
        )
        info_msg = "\nInfos:" + "".join(
            [f"\n\t\t{name}: {value}" for name, value in self.plain_infos.items()]
        )

        return loss_msg + "\n" + info_msg

    def host_dump_to_console(self, step=None):
        """Prints the infos object to the console (to be called from the host)."""
        print(self.host_get_str(step))

    def dump_to_console(self, train_state: TrainState):
        """Prints the infos object to the console (to be called from the device)."""
        id_tap(
            lambda arg, _: Infos.host_dump_to_console(*arg), (self, train_state.step)
        )

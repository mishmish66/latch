from learning.train_state import TrainState

import jax

from jax import numpy as jnp

from jax.tree_util import register_pytree_node_class
from jax.experimental.host_callback import id_tap

import wandb

from dataclasses import dataclass, field

import copy


@register_pytree_node_class
class Infos:
    infos: dict

    def __init__(self, infos: dict = {}):
        self.infos = infos

    @classmethod
    def merge(cls, *infs) -> "Infos":
        """Merges multiple infos objects into one.

        Returns:
            Infos: The merged infos object.
        """

        merged_infos = {}
        for inf in infs:
            merged_infos.update(inf.infos)

        return Infos(infos=merged_infos)

    def add_info(self, name, value):
        """Adds an info to the infos object.

        Args:
            name (str): The name of the info.
            value (Any): The value of the info.

        Returns:
            Infos: The updated infos object.
        """

        new_infos = copy.deepcopy(self.infos)
        new_infos[name] = value

        return Infos(infos=new_infos)

    def condense(self, method="mean"):
        """Condenses the infos object along the zero axis.

        Args:
            method (str, optional): Either "mean" or "unstack" which will either flatten the last axis axis or mean across it. Defaults to "mean".

        Returns:
            Infos: The condensed infos object.
        """
        methods_dict = {
            "mean": lambda x: jnp.mean(x, axis=0),
            "unstack": lambda x: jnp.reshape(x, (-1, *x.shape[2:])),
        }

        condenser = methods_dict[method]

        return Infos(infos=jax.tree_map(condenser, self.infos))

    def host_dump_to_wandb(self, step=None):
        """Dumps the infos object to wandb (to be called from the host)."""

        if step is not None:
            print(f"Logging 🪵 for step {step}")
            wandb.log(self.infos, step=step)
        else:
            print("Logging 🪵")
            wandb.log(self.infos)

    def dump_to_wandb(self, train_state: TrainState):
        """Dumps the infos object to wandb (to be called from the device)."""
        step = train_state.step

        def dump_to_wandb_for_tap(tap_pack, transforms):
            self, step = tap_pack
            Infos.host_dump_to_wandb(self, step)

        id_tap(
            dump_to_wandb_for_tap,
            (self, step),
        )

    def host_flatten_info_dict(self):
        """Flattens the infos to a unnested dict (to be called from the host)."""

        def flatten_infos_dict(infos):
            flat_infos = {}

            # Recursively flatten the infos dict
            for k, v in infos.items():
                # If the value is a dict, flatten it and add it to the flat infos dict
                if isinstance(v, dict):
                    # Flatten the dict
                    flat_sub_dict = flatten_infos_dict(v)
                    # Add prefixes to the keys in the sub dict
                    prefixed_flat_sub_dict = {
                        f"{k}/{sub_k}": sub_v for sub_k, sub_v in flat_sub_dict.items()
                    }
                    # Add the sub dict to the result flat infos dict
                    flat_infos.update(prefixed_flat_sub_dict)
                else:
                    # Otherwise, add the value to the flat infos dict
                    flat_infos[k] = v

            return flat_infos

        return flatten_infos_dict(self.infos)

    def host_get_str(self, step=None):
        """Gets a string representation of the infos object (to be called from the host)."""
        if step is not None:
            result_msg = f"Infos for step {step}:"
        else:
            result_msg = "Infos:"

        for k, v in self.host_flatten_info_dict().items():
            result_msg += f"\n    {k}: {v}"

        return result_msg

    def host_dump_to_console(self, step=None):
        """Prints the infos object to the console (to be called from the host)."""
        print(self.host_get_str(step))

    def dump_to_console(self, train_state: TrainState):
        """Prints the infos object to the console (to be called from the device)."""
        id_tap(
            lambda arg, _: Infos.host_dump_to_console(*arg),
            (self, train_state.step),
        )

    def tree_flatten(self):
        """Flattens the infos object into a tuple of arrays."""
        return (self.infos,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens the infos object from a tuple of arrays."""
        return cls(infos=children[0])

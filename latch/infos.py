import jax
from jax import numpy as jnp

from jax.tree_util import register_pytree_node_class
from jax.experimental.host_callback import id_tap

import wandb

from dataclasses import dataclass, field

from copy import deepcopy

from typing import Optional, Union


@register_pytree_node_class
class Infos:
    infos: dict

    def __init__(self, infos: dict = {}):
        self.infos = infos

    @classmethod
    def merge(cls, *infos_to_merge: "Infos") -> "Infos":
        """Merges multiple infos objects into one."""

        merged_infos = Infos()
        for info in infos_to_merge:
            # Add each key value pair from the info to the merged infos
            for info_name, info_value in info.infos.items():
                merged_infos = merged_infos.add_info(
                    info_name,
                    info_value,
                )

        return merged_infos

    def add_info(self, name: str, value: Union["Infos", jax.Array, dict, float, int]):
        """Adds an info to the infos object.

        Args:
            name (str): The name of the info.
            value (Union[jax.Array, float, int, Infos, dict]): The value of the info.

        Returns:
            Infos: The updated infos object.
        """

        new_value = value

        key_collision = name in self.infos

        if key_collision:
            # If there is a key collision, we might need to merge the new value with the old value
            old_value = self.infos[name]
            old_value_is_collection = isinstance(old_value, dict)
            add_value_is_collection = isinstance(value, dict) or isinstance(
                value, Infos
            )
            if old_value_is_collection and add_value_is_collection:
                # If the old and added values are collections, we need to merge them
                # First we make sure they are both infos objects
                old_value: Infos = Infos(infos=old_value)  # type: ignore
                add_value_is_dict = isinstance(value, dict)
                if add_value_is_dict:
                    value = Infos(infos=value)

                # Then we merge the old value with the new value
                new_value = Infos.merge(old_value, value).infos  # type: ignore

        new_value_is_infos = isinstance(new_value, Infos)
        if new_value_is_infos:
            # If the new value is an infos object, we need to extract the dict
            new_value = new_value.infos

        new_infos = deepcopy(self.infos)
        new_infos[name] = new_value

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

    def dump_to_wandb(self, step: int):
        """Dumps the infos object to wandb (to be called from the device)."""

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

    def dump_to_console(self, step: Optional[int]):
        """Prints the infos object to the console (to be called from the device)."""
        id_tap(
            lambda arg, _: Infos.host_dump_to_console(*arg),
            (self, step),
        )

    def __getitem__(self, key):
        if key in self.infos:
            return self.infos[key]
        else:
            raise KeyError(f"Key {key} not found in infos.")

    def tree_flatten(self):
        """Flattens the infos object into a tuple of arrays."""
        return (self.infos,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflattens the infos object from a tuple of arrays."""
        return cls(infos=children[0])

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

from dataclasses import dataclass


@register_pytree_node_class
@dataclass
class EnvConfig:
    action_bounds: any
    state_dim: any
    act_dim: any
    dt: any
    substep: any

    @classmethod
    def init(
        cls,
        action_bounds,
        state_dim,
        act_dim,
        dt,
        substep,
        env_cls,
    ):
        return cls(
            action_bounds=action_bounds,
            state_dim=state_dim,
            act_dim=act_dim,
            dt=dt,
            substep=substep,
            env_cls=env_cls,
        )

    def tree_flatten(self):
        return (
            self.action_bounds,
            self.state_dim,
            self.act_dim,
            self.dt,
            self.substep,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def random_action(self, key):
        rng, key = jax.random.split(key)
        random_nums = jax.random.uniform(rng, (self.action_bounds.shape[0],))
        scaled = random_nums * (self.action_bounds[:, 1] - self.action_bounds[:, 0])
        scaled_and_shifted = scaled + self.action_bounds[:, 0]

        return scaled_and_shifted

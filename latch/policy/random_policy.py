from learning.train_state import TrainState
from policy.policy import Policy

from infos import Infos

import jax
from jax import numpy as jnp
from jax.tree_util import Partial, register_pytree_node_class

from dataclasses import dataclass, replace


@register_pytree_node_class
@dataclass
class RandomPolicy(Policy):
    def make_init_carry(self, key, start_state, train_state: TrainState):
        return None, Infos.init()

    def init(**_):
        return RandomPolicy()

    def __call__(self, key, state, i, carry, train_state: TrainState):
        return (
            jax.random.uniform(
                key, shape=train_state.train_config.env_config.action_bounds.shape[:-1]
            ),
            None,
            Infos.init(),
        )

    def tree_flatten(self):
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls.init()

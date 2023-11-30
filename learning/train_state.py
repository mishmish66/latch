from learning.train_config import TrainConfig

import optax

import jax
import jax.numpy as jnp
import jax.lax
from jax.tree_util import register_pytree_node_class

from dataclasses import dataclass, replace


@register_pytree_node_class
@dataclass
class NetState:
    state_encoder_params: any
    action_encoder_params: any
    transition_model_params: any
    state_decoder_params: any
    action_decoder_params: any

    def tree_flatten(self):
        return (
            self.state_encoder_params,
            self.action_encoder_params,
            self.transition_model_params,
            self.state_decoder_params,
            self.action_decoder_params,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def to_list(self):
        return [
            self.state_encoder_params,
            self.action_encoder_params,
            self.transition_model_params,
            self.state_decoder_params,
            self.action_decoder_params,
        ]


@register_pytree_node_class
@dataclass
class TrainState:
    step: int
    net_state: NetState
    opt_state: any
    train_config: TrainConfig

    def tree_flatten(self):
        return (
            self.step,
            self.net_state,
            self.opt_state,
            self.train_config,
        ), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    @classmethod
    def init(cls, key, train_config: TrainConfig):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 6)

        state_encoder_params = train_config.state_encoder.init(
            rngs[0],
            jnp.ones(train_config.env_config.state_dim),
        )
        action_encoder_params = train_config.action_encoder.init(
            rngs[1],
            jnp.ones(train_config.env_config.act_dim),
            jnp.ones(train_config.latent_state_dim),
        )
        transition_model_params = train_config.transition_model.init(
            rngs[3],
            jnp.ones([train_config.latent_state_dim]),
            jnp.ones([16, train_config.latent_action_dim]),
            jnp.ones(16),
            10,
        )
        state_decoder_params = train_config.state_decoder.init(
            rngs[4],
            jnp.ones(train_config.latent_state_dim),
        )
        action_decoder_params = train_config.action_decoder.init(
            rngs[5],
            jnp.ones(train_config.latent_action_dim),
            jnp.ones(train_config.latent_state_dim),
        )

        net_state = NetState(
            state_encoder_params=state_encoder_params,
            action_encoder_params=action_encoder_params,
            transition_model_params=transition_model_params,
            state_decoder_params=state_decoder_params,
            action_decoder_params=action_decoder_params,
        )

        opt_state = train_config.optimizer.init(net_state)

        return cls(
            step=0,
            net_state=net_state,
            opt_state=opt_state,
            train_config=train_config,
        )

    def apply_gradients(self, grads: NetState):
        updates, new_opt_state = self.train_config.optimizer.update(
            grads,
            self.opt_state,
            self.net_state,
        )
        new_net_state = optax.apply_updates(self.net_state, updates)
        return self.replace(
            step=self.step + 1,
            opt_state=new_opt_state,
            net_state=new_net_state,
        )

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

    def replace(self, **kwargs):
        return replace(self, **kwargs)

    def tree_flatten(self):
        return (
            self.state_encoder_params,
            self.action_encoder_params,
            self.transition_model_params,
            self.state_decoder_params,
            self.action_decoder_params,
        ), None

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

        return cls(
            state_encoder_params=state_encoder_params,
            action_encoder_params=action_encoder_params,
            transition_model_params=transition_model_params,
            state_decoder_params=state_decoder_params,
            action_decoder_params=action_decoder_params,
        )

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

    @classmethod
    def from_list(cls, l):
        return cls(*l)


@register_pytree_node_class
@dataclass
class TrainState:
    key: jax.random.PRNGKey
    step: int
    rollout: int
    target_net_state: NetState
    primary_net_state: NetState
    opt_state: any
    train_config: TrainConfig

    def tree_flatten(self):
        return (
            self.key,
            self.step,
            self.rollout,
            self.target_net_state,
            self.primary_net_state,
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

        target_net_state = NetState.init(rng, train_config)
        primary_net_state = target_net_state

        opt_state = train_config.optimizer.init(primary_net_state)

        return cls(
            key=key,
            step=0,
            rollout=0,
            target_net_state=target_net_state,
            primary_net_state=primary_net_state,
            opt_state=opt_state,
            train_config=train_config,
        )

    def apply_gradients(self, grads: NetState):
        updates, new_opt_state = self.train_config.optimizer.update(
            grads,
            self.opt_state,
            self.primary_net_state,
        )

        new_primary_net_state = optax.apply_updates(self.primary_net_state, updates)

        return self.replace(
            step=self.step + 1,
            opt_state=new_opt_state,
            primary_net_state=new_primary_net_state,
        )

    def pull_target(self):
        target_factor = self.train_config.target_net_tau
        primary_factor = 1 - target_factor

        def leaf_interpolate(target, primary):
            return target_factor * target + primary_factor * primary

        new_target_net_state = jax.tree_map(
            leaf_interpolate,
            self.target_net_state,
            self.primary_net_state,
        )

        return self.replace(
            target_net_state=new_target_net_state,
        )

    def split_key(self):
        """This is a method that splits the key and returns the new key and the new train_state.

        Returns:
            (PRNGKey, TrainState): A new key and a new train_state.
        """
        rng, key = jax.random.split(self.key)
        return rng, self.replace(key=key)

    def is_done(self):
        return self.rollout >= self.train_config.rollouts

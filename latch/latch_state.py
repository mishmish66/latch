from latch import LatchConfig
from latch.nets import NetState
from latch import Infos

import optax

import jax_dataclasses as jdc

import jax
from jax import numpy as jnp

from typing import Any, Tuple


@jdc.pytree_dataclass(kw_only=True)
class LatchState:
    key: jax.Array
    step: int
    rollout: int
    target_net_state: NetState
    primary_net_state: NetState
    opt_state: Any
    config: LatchConfig

    @classmethod
    def init(cls, key, config: LatchConfig) -> "LatchState":
        rng, key = jax.random.split(key)

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 6)

        nets = config.nets

        # Initialize the network parameters
        state_encoder_params: Dict[str, Any] = nets.state_encoder.init(  # type: ignore
            rngs[0],
            jnp.ones(config.state_dim),
        )
        action_encoder_params: Dict[str, Any] = nets.action_encoder.init(  # type: ignore
            rngs[1],
            jnp.ones(config.action_dim),
            jnp.ones(config.latent_state_dim),
        )
        transition_model_params: Dict[str, Any] = nets.transition_model.init(  # type: ignore
            rngs[3],
            jnp.ones([config.latent_state_dim]),
            jnp.ones([16, config.latent_action_dim]),
            jnp.ones(16),
            10,
        )
        state_decoder_params: Dict[str, Any] = nets.state_decoder.init(  # type: ignore
            rngs[4],
            jnp.ones(config.latent_state_dim),
        )
        action_decoder_params: Dict[str, Any] = nets.action_decoder.init(  # type: ignore
            rngs[5],
            jnp.ones(config.latent_action_dim),
            jnp.ones(config.latent_state_dim),
        )

        target_net_state = NetState(
            state_encoder_params=state_encoder_params,
            action_encoder_params=action_encoder_params,
            transition_model_params=transition_model_params,
            state_decoder_params=state_decoder_params,
            action_decoder_params=action_decoder_params,
            nets=nets,
        )

        primary_net_state = target_net_state

        opt_state = train_config.optimizer.init(primary_net_state)  # type: ignore

        return cls(
            key=key,
            step=0,
            rollout=0,
            target_net_state=target_net_state,
            primary_net_state=primary_net_state,
            opt_state=opt_state,
            config=config,
        )

    def apply_gradients(self, grads) -> "LatchState":
        updates, new_opt_state = self.config.optimizer.update(
            grads,
            self.opt_state,
            self.primary_net_state,  # type: ignore
        )

        new_primary_net_state: NetState = optax.apply_updates(
            self.primary_net_state,  # type: ignore
            updates,
        )

        return self.replace(  # type: ignore
            step=self.step + 1,
            opt_state=new_opt_state,
            primary_net_state=new_primary_net_state,
        )

    def pull_target(self) -> "LatchState":
        target_factor = self.config.target_net_tau
        primary_factor = 1 - target_factor

        def leaf_interpolate(target, primary):
            return target_factor * target + primary_factor * primary

        new_target_net_state = jax.tree_map(
            leaf_interpolate,
            self.target_net_state,
            self.primary_net_state,
        )

        return self.replace(  # type: ignore
            target_net_state=new_target_net_state,
        )

    def split_key(self) -> Tuple[jax.Array, "LatchState"]:
        """This is a method that splits the key and returns the new key and the new train_state.

        Returns:
            (PRNGKey, TrainState): A new key and a new train_state.
        """
        rng, key = jax.random.split(self.key)
        return rng, self.replace(key=key)  # type: ignore

    def is_done(self):
        return self.rollout >= self.config.rollouts

    def train_step(
        self,
        key: jax.Array,
        states: jax.Array,
        actions: jax.Array,
    ) -> Tuple["LatchState", Infos]:
        """Train a single step of the network.

        Args:
            key (jax.Array): A random key.
            states (jax.Array): A (b x t x s) batch of states.
            actions (jax.Array): A (b x t-1 x a) batch of actions.

        Returns:
            (TrainState, Infos): The updated training state and the infos from the train step.
        """
        rng, key = jax.random.split(key)

        def loss_for_grad(net_state: NetState):
            """A small function that computes the loss only as a function of the net state for the gradient transformation."""

            loss, infos = self.config.latch_loss(
                key=rng,
                states=states,
                actions=actions,
                net_state=net_state,
            )

            return loss, infos

        loss_grad, loss_infos = jax.grad(loss_for_grad, has_aux=True)(
            train_state.primary_net_state  # type: ignore
        )

        infos = Infos()
        infos = infos.add_info("losses", loss_infos)

        # TODO: Log some gradient info potentially

        next_train_state = self.apply_gradients(loss_grad)

        return next_train_state, infos

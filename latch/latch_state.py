from latch import LatchConfig
from latch.models import ModelState, NetParams, Nets
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
    target_params: NetParams
    primary_params: NetParams
    opt_state: Any
    config: LatchConfig

    @classmethod
    def random_initial_state(cls, key, config: LatchConfig):
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

        target_params = NetParams(
            state_encoder_params=state_encoder_params,
            action_encoder_params=action_encoder_params,
            transition_model_params=transition_model_params,
            state_decoder_params=state_decoder_params,
            action_decoder_params=action_decoder_params,
        )

        primary_params = target_params

        opt_state = config.optimizer.init(primary_params)  # type: ignore

        return cls(
            key=key,
            step=0,
            rollout=0,
            target_params=target_params,
            primary_params=primary_params,
            opt_state=opt_state,
            config=config,
        )

    def apply_gradients(self, grads) -> "LatchState":
        updates, new_opt_state = self.config.optimizer.update(
            grads,
            self.opt_state,
            self.primary_net_state,  # type: ignore
        )

        new_primary_params: NetParams = optax.apply_updates(
            self.primary_net_state,  # type: ignore
            updates,
        )

        with jdc.copy_and_mutate(self) as new_state:
            new_state.primary_params = new_primary_params
            new_state.opt_state = new_opt_state
            new_state.step += 1

        return new_state

    def pull_target(self) -> "LatchState":
        target_factor = self.config.target_net_tau
        primary_factor = 1 - target_factor

        def leaf_interpolate(target, primary):
            return target_factor * target + primary_factor * primary

        new_target_net_state = jax.tree_map(
            leaf_interpolate,
            self.target_params,
            self.primary_params,
        )

        with jdc.copy_and_mutate(self) as new_state:
            new_state.target_params = new_target_net_state

        return new_state

    def split_key(self) -> Tuple[jax.Array, "LatchState"]:
        """This is a method that splits the key and returns the new key and the new train_state.

        Returns:
            (PRNGKey, TrainState): A new key and a new train_state.
        """
        rng, key = jax.random.split(self.key)

        with jdc.copy_and_mutate(self) as new_state:
            new_state.key = key

        return rng, new_state

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

        def loss_for_grad(net_params: NetParams):
            """A small function that computes the loss only as a function of the net state for the gradient transformation."""

            models = ModelState(net_params=net_params, nets=self.config.nets)
            loss, infos = self.config.latch_loss(
                key=rng,
                states=states,
                actions=actions,
                models=models,
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

    @property
    def primary_models(self):
        return ModelState(net_params=self.primary_params, nets=self.config.nets)

    @property
    def target_models(self):
        return ModelState(net_params=self.target_params, nets=self.config.nets)

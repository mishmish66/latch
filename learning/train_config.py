from nets.nets import (
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
)
from env.env import Env
from env.env_config import EnvConfig

from jax.tree_util import register_pytree_node_class

from dataclasses import dataclass


@register_pytree_node_class
@dataclass(frozen=True)
class TrainConfig:
    learning_rate: any
    optimizer: any

    state_encoder: StateEncoder
    action_encoder: ActionEncoder
    transition_model: TransitionModel
    state_decoder: StateDecoder
    action_decoder: ActionDecoder

    latent_state_dim: int
    latent_action_dim: int

    env_config: any
    env_cls: Env

    rollouts: int
    epochs: int
    batch_size: int
    every_k: int
    traj_per_rollout: int
    rollout_length: int
    seed: int

    target_net_tau: any

    state_radius: any
    action_radius: any

    reconstruction_weight: any
    forward_weight: any
    smoothness_weight: any
    dispersion_weight: any
    condensation_weight: any
    action_neighborhood_weight: any

    inverse_reconstruction_gate_sharpness: any
    inverse_forward_gate_sharpness: any

    inverse_reconstruction_gate_center: any
    inverse_forward_gate_center: any

    forward_blend_gate_sharpness: any
    forward_blend_gate_center: any

    forward_gate_sharpness: any
    smoothness_gate_sharpness: any
    dispersion_gate_sharpness: any
    condensation_gate_sharpness: any

    forward_gate_center: any
    smoothness_gate_center: any
    dispersion_gate_center: any
    condensation_gate_center: any

    @classmethod
    def init(
        cls,
        learning_rate,
        optimizer,
        state_encoder: StateEncoder,
        action_encoder: ActionEncoder,
        transition_model: TransitionModel,
        state_decoder: StateDecoder,
        action_decoder: ActionDecoder,
        latent_state_dim: any,
        latent_action_dim: any,
        env_config: EnvConfig,
        env_cls: Env,
        seed,
        target_net_tau=0.05,
        rollouts=1024,
        epochs=128,
        batch_size=256,
        every_k=1,
        traj_per_rollout=2048,
        rollout_length=250,
        state_radius=3.0,
        action_radius=2.0,
        reconstruction_weight=1.0,
        forward_weight=1.0,
        smoothness_weight=1.0,
        dispersion_weight=1.0,
        condensation_weight=1.0,
        action_neighborhood_weight=1.0,
        inverse_reconstruction_gate_sharpness=1.0,
        inverse_forward_gate_sharpness=1.0,
        inverse_reconstruction_gate_center=-5.0,
        inverse_forward_gate_center=-5.0,
        forward_blend_gate_sharpness=1.0,
        forward_blend_gate_center=0.0,
        forward_gate_sharpness=1.0,
        smoothness_gate_sharpness=1.0,
        dispersion_gate_sharpness=1.0,
        condensation_gate_sharpness=1.0,
        forward_gate_center=-3.0,
        smoothness_gate_center=-5.0,
        dispersion_gate_center=-5.0,
        condensation_gate_center=-5.0,
    ):
        return cls(
            learning_rate=learning_rate,
            optimizer=optimizer,
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            transition_model=transition_model,
            state_decoder=state_decoder,
            action_decoder=action_decoder,
            latent_state_dim=latent_state_dim,
            latent_action_dim=latent_action_dim,
            env_config=env_config,
            env_cls=env_cls,
            seed=seed,
            target_net_tau=target_net_tau,
            rollouts=rollouts,
            epochs=epochs,
            batch_size=batch_size,
            every_k=every_k,
            traj_per_rollout=traj_per_rollout,
            rollout_length=rollout_length,
            state_radius=state_radius,
            action_radius=action_radius,
            reconstruction_weight=reconstruction_weight,
            forward_weight=forward_weight,
            smoothness_weight=smoothness_weight,
            dispersion_weight=dispersion_weight,
            condensation_weight=condensation_weight,
            action_neighborhood_weight=action_neighborhood_weight,
            inverse_reconstruction_gate_sharpness=inverse_reconstruction_gate_sharpness,
            inverse_forward_gate_sharpness=inverse_forward_gate_sharpness,
            inverse_reconstruction_gate_center=inverse_reconstruction_gate_center,
            inverse_forward_gate_center=inverse_forward_gate_center,
            forward_blend_gate_sharpness=forward_blend_gate_sharpness,
            forward_blend_gate_center=forward_blend_gate_center,
            forward_gate_sharpness=forward_gate_sharpness,
            smoothness_gate_sharpness=smoothness_gate_sharpness,
            dispersion_gate_sharpness=dispersion_gate_sharpness,
            condensation_gate_sharpness=condensation_gate_sharpness,
            forward_gate_center=forward_gate_center,
            smoothness_gate_center=smoothness_gate_center,
            dispersion_gate_center=dispersion_gate_center,
            condensation_gate_center=condensation_gate_center,
        )

    def make_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "latent_state_dim": self.latent_state_dim,
            "latent_action_dim": self.latent_action_dim,
            "traj_per_rollout": self.traj_per_rollout,
            "seed": self.seed,
            "target_net_tau": self.target_net_tau,
            "rollouts": self.rollouts,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "every_k": self.every_k,
            "rollout_length": self.rollout_length,
            "state_radius": self.state_radius,
            "action_radius": self.action_radius,
            "reconstruction_weight": self.reconstruction_weight,
            "forward_weight": self.forward_weight,
            "smoothness_weight": self.smoothness_weight,
            "dispersion_weight": self.dispersion_weight,
            "condensation_weight": self.condensation_weight,
            "action_neighborhood_weight": self.action_neighborhood_weight,
            "inverse_reconstruction_gate_sharpness": self.inverse_reconstruction_gate_sharpness,
            "inverse_forward_gate_sharpness": self.inverse_forward_gate_sharpness,
            "inverse_reconstruction_gate_center": self.inverse_reconstruction_gate_center,
            "inverse_forward_gate_center": self.inverse_forward_gate_center,
            "forward_blend_gate_sharpness": self.forward_blend_gate_sharpness,
            "forward_blend_gate_center": self.forward_blend_gate_center,
            "forward_gate_sharpness": self.forward_gate_sharpness,
            "smoothness_gate_sharpness": self.smoothness_gate_sharpness,
            "dispersion_gate_sharpness": self.dispersion_gate_sharpness,
            "condensation_gate_sharpness": self.condensation_gate_sharpness,
            "forward_gate_center": self.forward_gate_center,
            "smoothness_gate_center": self.smoothness_gate_center,
            "dispersion_gate_center": self.dispersion_gate_center,
            "condensation_gate_center": self.condensation_gate_center,
        }

    def tree_flatten(self):
        return [None], {
            "learning_rate": self.learning_rate,
            "latent_state_dim": self.latent_state_dim,
            "latent_action_dim": self.latent_action_dim,
            "optimizer": self.optimizer,
            "state_encoder": self.state_encoder,
            "action_encoder": self.action_encoder,
            "transition_model": self.transition_model,
            "state_decoder": self.state_decoder,
            "action_decoder": self.action_decoder,
            "env_config": self.env_config,
            "env_cls": self.env_cls,
            "seed": self.seed,
            "target_net_tau": self.target_net_tau,
            "rollouts": self.rollouts,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "every_k": self.every_k,
            "traj_per_rollout": self.traj_per_rollout,
            "rollout_length": self.rollout_length,
            "state_radius": self.state_radius,
            "action_radius": self.action_radius,
            "reconstruction_weight": self.reconstruction_weight,
            "forward_weight": self.forward_weight,
            "smoothness_weight": self.smoothness_weight,
            "dispersion_weight": self.dispersion_weight,
            "condensation_weight": self.condensation_weight,
            "action_neighborhood_weight": self.action_neighborhood_weight,
            "inverse_reconstruction_gate_sharpness": self.inverse_reconstruction_gate_sharpness,
            "inverse_forward_gate_sharpness": self.inverse_forward_gate_sharpness,
            "inverse_reconstruction_gate_center": self.inverse_reconstruction_gate_center,
            "inverse_forward_gate_center": self.inverse_forward_gate_center,
            "forward_blend_gate_sharpness": self.forward_blend_gate_sharpness,
            "forward_blend_gate_center": self.forward_blend_gate_center,
            "forward_gate_sharpness": self.forward_gate_sharpness,
            "smoothness_gate_sharpness": self.smoothness_gate_sharpness,
            "dispersion_gate_sharpness": self.dispersion_gate_sharpness,
            "condensation_gate_sharpness": self.condensation_gate_sharpness,
            "forward_gate_center": self.forward_gate_center,
            "smoothness_gate_center": self.smoothness_gate_center,
            "dispersion_gate_center": self.dispersion_gate_center,
            "condensation_gate_center": self.condensation_gate_center,
        }

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls.init(
            learning_rate=aux["learning_rate"],
            optimizer=aux["optimizer"],
            state_encoder=aux["state_encoder"],
            action_encoder=aux["action_encoder"],
            transition_model=aux["transition_model"],
            state_decoder=aux["state_decoder"],
            action_decoder=aux["action_decoder"],
            latent_state_dim=aux["latent_state_dim"],
            latent_action_dim=aux["latent_action_dim"],
            env_config=aux["env_config"],
            env_cls=aux["env_cls"],
            seed=aux["seed"],
            target_net_tau=aux["target_net_tau"],
            rollouts=aux["rollouts"],
            epochs=aux["epochs"],
            batch_size=aux["batch_size"],
            every_k=aux["every_k"],
            traj_per_rollout=aux["traj_per_rollout"],
            rollout_length=aux["rollout_length"],
            state_radius=aux["state_radius"],
            action_radius=aux["action_radius"],
            reconstruction_weight=aux["reconstruction_weight"],
            forward_weight=aux["forward_weight"],
            smoothness_weight=aux["smoothness_weight"],
            dispersion_weight=aux["dispersion_weight"],
            condensation_weight=aux["condensation_weight"],
            action_neighborhood_weight=aux["action_neighborhood_weight"],
            inverse_reconstruction_gate_sharpness=aux[
                "inverse_reconstruction_gate_sharpness"
            ],
            inverse_forward_gate_sharpness=aux["inverse_forward_gate_sharpness"],
            inverse_reconstruction_gate_center=aux[
                "inverse_reconstruction_gate_center"
            ],
            inverse_forward_gate_center=aux["inverse_forward_gate_center"],
            forward_blend_gate_sharpness=aux["forward_blend_gate_sharpness"],
            forward_blend_gate_center=aux["forward_blend_gate_center"],
            forward_gate_sharpness=aux["forward_gate_sharpness"],
            smoothness_gate_sharpness=aux["smoothness_gate_sharpness"],
            dispersion_gate_sharpness=aux["dispersion_gate_sharpness"],
            condensation_gate_sharpness=aux["condensation_gate_sharpness"],
            forward_gate_center=aux["forward_gate_center"],
            smoothness_gate_center=aux["smoothness_gate_center"],
            dispersion_gate_center=aux["dispersion_gate_center"],
            condensation_gate_center=aux["condensation_gate_center"],
        )

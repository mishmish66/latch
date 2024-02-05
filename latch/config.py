from dataclasses import dataclass
from typing import List

import hydra
import jax
import optax
from hydra.core.config_store import ConfigStore

from latch import LatchConfig, LatchState
from latch.env import Env
from latch.loss import (
    ActionCondensationLoss,
    ActionDispersionLoss,
    ActionReconstructionLoss,
    ForwardLoss,
    LatchLoss,
    SmoothnessLoss,
    StateCondensationLoss,
    StateDispersionLoss,
    StateReconstructionLoss,
)
from latch.models import (
    ActionDecoder,
    ActionEncoder,
    Nets,
    StateDecoder,
    StateEncoder,
    TransitionModel,
)


@dataclass
class NetConfig:
    latent_state_dim: int
    state_dim: int

    latent_action_dim: int
    action_dim: int

    state_encoder_layers: List[int]
    action_encoder_layers: List[int]

    state_decoder_layers: List[int]
    action_decoder_layers: List[int]

    temporal_encoder_min_freq: float
    temporal_encoder_max_freq: float

    transition_model_n_layers: int
    transition_model_latent_dim: int
    transition_model_n_heads: int

    latent_state_radius: float
    latent_action_radius: float


@dataclass
class WeightedLossConfig:
    weight: float


@dataclass
class SigmoidGatedLossConfig(WeightedLossConfig):
    sharpness: float
    center: float


@dataclass
class StateDispersionLossConfig(SigmoidGatedLossConfig):
    num_samples: int


@dataclass
class ActionDispersionLossConfig(SigmoidGatedLossConfig):
    num_samples: int


@dataclass
class LatchLossConfig:
    state_reconstruction_loss_config: WeightedLossConfig
    action_reconstruction_loss_config: WeightedLossConfig
    forward_loss_config: SigmoidGatedLossConfig
    state_condensation_loss_config: SigmoidGatedLossConfig
    action_condensation_loss_config: SigmoidGatedLossConfig
    smoothness_loss_config: SigmoidGatedLossConfig
    state_dispersion_loss_config: StateDispersionLossConfig
    action_dispersion_loss_config: ActionDispersionLossConfig


@dataclass
class TrainConfig:

    net_config: NetConfig

    rollouts: int
    epochs: int
    batch_size: int
    traj_per_rollout: int
    rollout_length: int
    target_net_tau: float
    learning_rate: float

    checkpoint_dir: str
    checkpoint_count: int
    save_every: int
    eval_every: int
    use_wandb: bool

    seed: int
    resume: bool

    loss_config: LatchLossConfig


cs = ConfigStore.instance()
cs.store(name="net_config", node=NetConfig)
cs.store(name="train_config", node=TrainConfig)


def configure_nets(net_config: NetConfig):
    return Nets(
        state_encoder=StateEncoder(net_config.latent_state_dim),
        action_encoder=ActionEncoder(net_config.latent_action_dim),
        transition_model=TransitionModel(
            latent_state_dim=net_config.latent_state_dim,
            n_layers=net_config.transition_model_n_layers,
            latent_dim=net_config.transition_model_latent_dim,
            heads=net_config.transition_model_n_heads,
        ),
        state_decoder=StateDecoder(net_config.state_dim),
        action_decoder=ActionDecoder(net_config.action_dim),
        latent_state_radius=net_config.latent_state_radius,
        latent_action_radius=net_config.latent_action_radius,
    )


def configure_loss(loss_config: LatchLossConfig):
    return LatchLoss(
        state_reconstruction_loss=StateReconstructionLoss(
            weight=loss_config.state_reconstruction_loss_config.weight,
        ),
        action_reconstruction_loss=ActionReconstructionLoss(
            weight=loss_config.action_reconstruction_loss_config.weight,
        ),
        forward_loss=ForwardLoss(
            weight=loss_config.forward_loss_config.weight,
            sharpness=loss_config.forward_loss_config.sharpness,
            center=loss_config.forward_loss_config.center,
        ),
        state_condensation_loss=StateCondensationLoss(
            weight=loss_config.state_condensation_loss_config.weight,
            sharpness=loss_config.state_condensation_loss_config.sharpness,
            center=loss_config.state_condensation_loss_config.center,
        ),
        action_condensation_loss=ActionCondensationLoss(
            weight=loss_config.action_condensation_loss_config.weight,
            sharpness=loss_config.action_condensation_loss_config.sharpness,
            center=loss_config.action_condensation_loss_config.center,
        ),
        smoothness_loss=SmoothnessLoss(
            weight=loss_config.smoothness_loss_config.weight,
            sharpness=loss_config.smoothness_loss_config.sharpness,
            center=loss_config.smoothness_loss_config.center,
        ),
        state_dispersion_loss=StateDispersionLoss(
            weight=loss_config.state_dispersion_loss_config.weight,
            sharpness=loss_config.state_dispersion_loss_config.sharpness,
            center=loss_config.state_dispersion_loss_config.center,
            num_samples=loss_config.state_dispersion_loss_config.num_samples,
        ),
        action_dispersion_loss=ActionDispersionLoss(
            weight=loss_config.action_dispersion_loss_config.weight,
            sharpness=loss_config.action_dispersion_loss_config.sharpness,
            center=loss_config.action_dispersion_loss_config.center,
            num_samples=loss_config.action_dispersion_loss_config.num_samples,
        ),
    )


def configure_state(train_config: TrainConfig, env: Env):
    latch_config = LatchConfig(
        optimizer=optax.chain(
            optax.zero_nans(),
            optax.adamw(learning_rate=train_config.learning_rate),
        ),
        env=env,
        nets=configure_nets(train_config.net_config),
        latch_loss=configure_loss(train_config.loss_config),
        rollouts=train_config.rollouts,
        epochs=train_config.epochs,
        batch_size=train_config.batch_size,
        traj_per_rollout=train_config.traj_per_rollout,
        rollout_length=train_config.rollout_length,
        target_net_tau=train_config.target_net_tau,
    )

    key = jax.random.PRNGKey(train_config.seed)
    latch_state = LatchState.random_initial_state(key=key, config=latch_config)

    return latch_state

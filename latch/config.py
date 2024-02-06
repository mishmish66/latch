from dataclasses import dataclass
from typing import List

import hydra
import jax
import optax
from hydra.core.config_store import ConfigStore
from loss_gate_graph import Edge, LossGateGraph, SigmoidGate, SpikeGate
from omegaconf import DictConfig, OmegaConf

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
class EdgeConfig:
    source: str
    target: str
    gate: DictConfig


@dataclass
class LatchLossConfig:
    loss_edges: List[EdgeConfig]

    state_reconstruction_loss_config: DictConfig
    action_reconstruction_loss_config: DictConfig
    forward_loss_config: DictConfig
    state_condensation_loss_config: DictConfig
    action_condensation_loss_config: DictConfig
    smoothness_loss_config: DictConfig
    state_dispersion_loss_config: DictConfig
    action_dispersion_loss_config: DictConfig


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
        state_encoder=StateEncoder(
            net_config.latent_state_dim, net_config.state_encoder_layers
        ),
        action_encoder=ActionEncoder(
            net_config.latent_action_dim, net_config.action_encoder_layers
        ),
        transition_model=TransitionModel(
            latent_state_dim=net_config.latent_state_dim,
            n_layers=net_config.transition_model_n_layers,
            latent_dim=net_config.transition_model_latent_dim,
            heads=net_config.transition_model_n_heads,
        ),
        state_decoder=StateDecoder(
            net_config.state_dim, net_config.state_decoder_layers
        ),
        action_decoder=ActionDecoder(
            net_config.action_dim, net_config.action_decoder_layers
        ),
        latent_state_radius=net_config.latent_state_radius,
        latent_action_radius=net_config.latent_action_radius,
    )


name_to_gate = {"spike": SpikeGate, "sigmoid": SigmoidGate}

name_to_loss_func = {
    "state_reconstruction_loss": StateReconstructionLoss,
    "action_reconstruction_loss": ActionReconstructionLoss,
    "forward_loss": ForwardLoss,
    "state_condensation_loss": StateCondensationLoss,
    "action_condensation_loss": ActionCondensationLoss,
    "smoothness_loss": SmoothnessLoss,
    "state_dispersion_loss": StateDispersionLoss,
    "action_dispersion_loss": ActionDispersionLoss,
}


def configure_loss(loss_config: LatchLossConfig):

    edges = []
    for edge_config in loss_config.loss_edges:
        source = name_to_loss_func[edge_config.source]
        target = name_to_loss_func[edge_config.target]
        gate_class = name_to_gate[edge_config.gate.type]
        gate_params = {
            param_name: param_val
            for param_name, param_val in OmegaConf.to_container(
                edge_config.gate
            ).items()  # type: ignore
            if param_name != "type"
        }
        gate = gate_class(**gate_params)
        edges.append(
            Edge(
                source=source,
                target=target,
                gate=gate,
            )
        )

    return LatchLoss(
        loss_gate_graph=LossGateGraph(edges=edges),
        state_reconstruction_loss=StateReconstructionLoss(
            weight=loss_config.state_reconstruction_loss_config.weight,
        ),
        action_reconstruction_loss=ActionReconstructionLoss(
            weight=loss_config.action_reconstruction_loss_config.weight,
        ),
        forward_loss=ForwardLoss(
            weight=loss_config.forward_loss_config.weight,
        ),
        state_condensation_loss=StateCondensationLoss(
            weight=loss_config.state_condensation_loss_config.weight,
        ),
        action_condensation_loss=ActionCondensationLoss(
            weight=loss_config.action_condensation_loss_config.weight,
        ),
        smoothness_loss=SmoothnessLoss(
            weight=loss_config.smoothness_loss_config.weight,
        ),
        state_dispersion_loss=StateDispersionLoss(
            weight=loss_config.state_dispersion_loss_config.weight,
            num_samples=loss_config.state_dispersion_loss_config.num_samples,
        ),
        action_dispersion_loss=ActionDispersionLoss(
            weight=loss_config.action_dispersion_loss_config.weight,
            num_samples=loss_config.action_dispersion_loss_config.num_samples,
        ),
    )


def configure_state(train_config: TrainConfig, env: Env):
    latch_config = LatchConfig(
        optimizer=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(1.0),
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

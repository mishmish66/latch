from dataclasses import dataclass, field
from typing import List, Optional

import hydra
import jax
import optax
from hydra.core.config_store import ConfigStore
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
from latch.loss.gates import Gate, gate_dict
from latch.loss.loss_func import LossFunc
from latch.loss.loss_graph import GateEdge, LossGateGraph
from latch.loss.loss_registry import loss_dict
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

    latent_state_radius: float
    latent_action_radius: float

    state_encoder_layers: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])  # type: ignore
    action_encoder_layers: List[int] = field(default_factory=lambda: [1024, 512, 256, 64])  # type: ignore

    state_decoder_layers: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])  # type: ignore
    action_decoder_layers: List[int] = field(default_factory=lambda: [1024, 512, 256, 64])  # type: ignore

    temporal_encoder_min_freq: float = 0.015625
    temporal_encoder_max_freq: float = 0.5

    transition_model_n_layers: int = 3
    transition_model_latent_dim: int = 128
    transition_model_n_heads: int = 4


@dataclass
class EdgeConfig:
    source: str
    target: str
    gate_config: Gate.Config


@dataclass
class LatchLossConfig:
    edge_config_list: List[EdgeConfig] = field(default_factory=lambda: [])

    loss_config_list: List[LossFunc.Config] = field(default_factory=lambda: [])

    # state_reconstruction_loss_config: StateReconstructionLoss.Config
    # action_reconstruction_loss_config: ActionReconstructionLoss.Config
    # forward_loss_config: ForwardLoss.Config
    # state_condensation_loss_config: StateCondensationLoss.Config
    # action_condensation_loss_config: ActionCondensationLoss.Config
    # smoothness_loss_config: SmoothnessLoss.Config
    # state_dispersion_loss_config: StateDispersionLoss.Config
    # action_dispersion_loss_config: ActionDispersionLoss.Config


@dataclass
class TrainConfig:

    net_config: NetConfig
    loss_config: LatchLossConfig = LatchLossConfig()

    rollouts: int = 512
    epochs: int = 64
    batch_size: int = 64
    traj_per_rollout: int = 1024
    rollout_length: int = 64
    target_net_tau: float = 0.05
    learning_rate: float = 1e-4

    checkpoint_dir: str = "checkpoints"
    checkpoint_count: int = 3
    save_every: int = 1
    eval_every: int = 1
    use_wandb: bool = False

    seed: int = 0
    resume: bool = False

    warm_start_path: Optional[str] = None


### Register Configs ####

cs = ConfigStore.instance()
cs.store(name="net_config", node=NetConfig)
cs.store(name="edge_config", node=EdgeConfig)
cs.store(name="latch_loss_config", node=LatchLossConfig)
cs.store(name="train_config", node=TrainConfig)


### Process Configs ####


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


def configure_loss(latch_loss_config: LatchLossConfig):

    losses = []
    for loss_config in latch_loss_config.loss_config_list:
        loss_class = loss_dict[loss_config.loss_type]
        loss_instance = loss_class.configure(loss_config)
        losses.append(loss_instance)

    edges = []
    for edge_config in latch_loss_config.edge_config_list:
        source = edge_config.source
        target = edge_config.target

        gate_config = edge_config.gate_config
        gate_class = gate_dict[gate_config.gate_type]
        gate_instance = gate_class.configure(gate_config)

        edges.append(
            GateEdge(
                source=source,
                target=target,
                gate=gate_instance,
            )
        )

    return LatchLoss(
        loss_list=losses,
        loss_gate_graph=LossGateGraph(edges=edges),
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

from .loss import LatchLoss
from latch.env import Env

from latch.models import Nets

import optax

import jax_dataclasses as jdc

import jax

from typing import Any


@jdc.pytree_dataclass
class LatchConfig:
    optimizer: jdc.Static[optax.GradientTransformation]

    env: Env

    nets: Nets
    latch_loss: LatchLoss = LatchLoss()

    # Declare anything that could possibly decide a shape as static
    rollouts: jdc.Static[int] = 1024
    epochs: jdc.Static[int] = 128
    batch_size: jdc.Static[int] = 256
    traj_per_rollout: jdc.Static[int] = 2048
    rollout_length: jdc.Static[int] = 250

    target_net_tau: float = 0.05

    @property
    def state_dim(self):
        return self.env.state_dim

    @property
    def action_dim(self):
        return self.env.action_dim

    @property
    def latent_state_dim(self):
        return self.nets.latent_state_dim

    @property
    def latent_action_dim(self):
        return self.nets.latent_action_dim

    @property
    def latent_state_radius(self):
        return self.nets.latent_state_radius

    @property
    def latent_action_radius(self):
        return self.nets.latent_action_radius

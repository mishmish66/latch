from abc import abstractmethod
from dataclasses import dataclass

import jax
import jax_dataclasses as jdc
from hydra.core.config_store import ConfigStore
from jax import numpy as jnp

cs = ConfigStore.instance()


class Gate:
    @abstractmethod
    def _compute_gate(self, raw_loss) -> jax.Array:
        raise NotImplementedError

    def compute(self, in_val) -> jax.Array:

        return jax.lax.stop_gradient(self._compute_gate(in_val))

    @dataclass
    class Config:
        gate_type: str

    @staticmethod
    @abstractmethod
    def configure(config: "Gate.Config") -> "Gate":
        raise NotImplementedError


@jdc.pytree_dataclass
class SigmoidGate(Gate):
    sharpness: float = 1.0
    center: float = 0.0

    def _compute_gate(self, raw_loss: jax.Array) -> jax.Array:
        gate_value: jax.Array = (
            1 + jnp.exp(self.sharpness * (raw_loss - self.center))
        ) ** -1

        return gate_value

    @dataclass
    class Config(Gate.Config):
        sharpness: float = 1.0
        center: float = 0.0
        gate_type: str = "sigmoid"

    @staticmethod
    def configure(config: "SigmoidGate.Config") -> "SigmoidGate":
        return SigmoidGate(config.sharpness, config.center)


cs.store(group="gate", name="sigmoid", node=SigmoidGate.Config)


@jdc.pytree_dataclass
class SpikeGate(Gate):
    sharpness: float = 10_000.0
    center: float = 0.0

    def _compute_gate(self, raw_loss: jax.Array) -> jax.Array:
        gate_value: jax.Array = (
            jnp.exp(self.sharpness * jnp.square(raw_loss - self.center))
        ) ** -1
        return gate_value

    @dataclass
    class Config(Gate.Config):
        sharpness: float = 10_000.0
        center: float = 0.0
        gate_type: str = "spike"

    @staticmethod
    def configure(config: "SpikeGate.Config") -> "SpikeGate":
        return SpikeGate(config.sharpness, config.center)


cs.store(group="gate", name="spike", node=SpikeGate.Config)


gate_dict = {
    "spike": SpikeGate,
    "sigmoid": SigmoidGate,
}

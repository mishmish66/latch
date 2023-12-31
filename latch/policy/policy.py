from latch.learning import TrainState
from latch.infos import Infos

import jax

from typing import Tuple
from abc import ABC, abstractmethod


class Policy[C](ABC):
    """Abstract base class for policies."""

    @abstractmethod
    def make_init_carry(
        self,
        key: jax.Array,
        start_state: jax.Array,
        train_state: TrainState,
    ) -> Tuple[C, Infos]:
        raise NotImplementedError("make_init_carry not implemented in base class")

    @abstractmethod
    def __call__(
        self,
        key: jax.Array,
        state: jax.Array,
        i: int,
        carry: C,
        train_state: TrainState,
    ) -> Tuple[jax.Array, C, Infos]:
        raise NotImplementedError("__call__ not implemented in base class")

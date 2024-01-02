from latch import LatchState, Infos

import jax

from typing import Tuple, Generic, TypeVar
from abc import ABC, abstractmethod

C = TypeVar("C")

class Policy(ABC, Generic[C]):
    """Abstract base class for policies."""

    @abstractmethod
    def make_init_carry(
        self,
        key: jax.Array,
        start_state: jax.Array,
        train_state: LatchState,
    ) -> Tuple[C, Infos]:
        raise NotImplementedError("make_init_carry not implemented in base class")

    @abstractmethod
    def __call__(
        self,
        key: jax.Array,
        state: jax.Array,
        i: int,
        carry: C,
        train_state: LatchState,
    ) -> Tuple[jax.Array, C, Infos]:
        raise NotImplementedError("__call__ not implemented in base class")

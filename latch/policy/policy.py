from learning.train_state import TrainState


class Policy:
    def make_init_carry(self, key, start_state, aux, train_state: TrainState):
        raise NotImplementedError

    def __call__(self, key, state, i, carry, aux, train_state: TrainState):
        raise NotImplementedError

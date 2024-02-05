from latch.training.trainer import Trainer

import hydra

from latch.config import TrainConfig, configure_state
from latch.env import Finger


@hydra.main(version_base=None, config_path="conf", config_name="train")
def train(cfg: TrainConfig) -> None:

    # Instantiate the environment
    env = Finger.init()

    # Initialize the train state
    train_state = configure_state(train_config=cfg, env=env)

    trainer = Trainer(train_config=cfg)

    trainer.train(train_state)

    print("Finished Training 🎉")


if __name__ == "__main__":
    train()

import copy
import importlib
import os
import time
from pathlib import Path
import numpy
import torch
from torch import multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter

import models
import replay_buffer
import self_play
import trainer
from games.base_classes import MuZeroConfigBase


def time_stamp_str():
    import datetime
    return datetime.datetime.now().strftime("%m-%d_%H-%M-%S")


class MuZero:
    """
    Main class to manage MuZero.

    Args:
        game_name (str): Name of the game module, it should match the name of a .py file
        in the "./games" directory.

    Example:
        >>> muzero = MuZero("cartpole")
        >>> muzero.train()
        >>> muzero.test()
    """

    def __init__(self, game_name):
        self.game_name = game_name

        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + self.game_name)
            self.config: MuZeroConfigBase = game_module.MuZeroConfig()
            self.Game = game_module.Game
        except Exception as err:
            print(
                '{} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'.format(
                    self.game_name
                )
            )
            raise err

        os.makedirs(os.path.join(self.config.results_path), exist_ok=True)

        # Fix random generator seed for reproductibility
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)

        # Initial weights used to initialize components
        self.muzero_weights = models.MuZeroExtendedNetwork(
            self.config.observation_shape,
            len(self.config.action_space),
            self.config.encoding_size,
            self.config.hidden_size,
        ).get_weights()

        self.config.results_path = (
            Path(self.config.results_path)
            / (self.game_name + "_summary")
            / time_stamp_str()
        )

    def train(self):
        writer = SummaryWriter(self.config.results_path)

        # Initialize workers
        training_worker = mp.Process(
            target=trainer.Trainer,
            args=(copy.deepcopy(self.muzero_weights), self.config),
        )
        training_worker.start()
        replay_buffer_worker = mp.Process(
            target=replay_buffer.ReplayBuffer, args=(self.config,)
        )
        replay_buffer_worker.start()
        self_play_workers = [
            mp.Process(
                target=self_play.SelfPlay,
                args=(
                    copy.deepcopy(self.muzero_weights),
                    self.Game(self.config.seed + seed, i),
                    self.config,
                ),
                kwargs={"idx": seed, "test": False},
            )
            for i,seed in enumerate(range(self.config.num_actors))
        ]
        [p.start() for p in self_play_workers]
        test_worker = mp.Process(
            target=self_play.SelfPlay,
            args=(copy.deepcopy(self.muzero_weights), self.Game(), self.config),
            kwargs={"test": True},
        )
        test_worker.start()

        # Loop for monitoring in real time the workers
        print(
            "\nTraining...\nRun tensorboard --logdir ./ and go to http://localhost:6006/ to see in real time the training performance.\n"
        )
        self.config: MuZeroConfigBase
        while self.config.v_training_step.value < self.config.training_steps:
            time.sleep(10)

        print("Training steps set in conf exceeded. Killing all processes")
        training_worker.kill()
        replay_buffer_worker.kill()
        [worker.kill() for worker in self_play_workers]
        test_worker.kill()

    def test(self, render=True):
        """
        Test the model in a dedicated thread.
        """
        print("\nTesting...")
        test_worker = mp.Process(
            target=self_play.SelfPlay,
            args=(copy.deepcopy(self.muzero_weights), self.Game(), self.config),
            kwargs={"test": True, "idx": 0, "render": True},
        ).start()

        print("sleeping. Keyboard interrupt me when ready.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("stopping!")

        # for _ in range(self.config.test_episodes):
        #     history = ray.get(self_play_workers.play_game.remote(0, render))
        #     test_rewards.append(sum(history.rewards))
        # return test_rewards

    def load_model(self, path=None):
        if not path:
            path = os.path.join(self.config.results_path, self.game_name)
        try:
            self.muzero_weights = torch.load(path)
            print("Using weights from {}".format(path))
        except FileNotFoundError:
            print("There is no model saved in {}.".format(path))


if __name__ == "__main__":
    zero = MuZero('bpp')
    # zero = MuZero('connect4')
    muzero = zero
    muzero.train()
    muzero.load_model()
    muzero.test()

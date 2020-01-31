import gym
import torch
from torch import multiprocessing as mp


class MuZeroConfigBase:
    def __init__(self):
        self.seed = 0  # Seed for numpy, torch and the game

        ### Inter Process Communication
        self.v_self_play_count = mp.Value("i", 0)
        self.v_training_step = mp.Value("i", 0)
        self.v_total_reward = mp.Value("i", 0)
        self.q_save_game = mp.Queue()
        self.q_replay_batch = mp.Queue(maxsize=10)
        self.q_weights = mp.Queue(maxsize=1)

        ### Game
        self.observation_shape = 4  # Dimensions of the game observation
        self.action_space = [i for i in range(2)]  # Fixed list of all possible actions
        self.players = [i for i in range(1)]  # List of players

        ### Self-Play
        self.num_actors = (
            10  # Number of simultaneous threads self-playing to feed the replay buffer
        )
        self.max_moves = 500  # Maximum number of moves if game is not finished before
        self.num_simulations = 80  # Number of futur moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.self_play_delay = 0  # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid over/underfitting

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        ### Network
        self.encoding_size = 32
        self.hidden_size = 64

        ### Training
        self.results_path = "./pretrained"  # Path to store the model weights
        self.training_steps = 10000  # Total number of training steps (ie weights update according to a batch)
        self.batch_size = (
            128  # Number of parts of games to train on at each training step
        )
        self.num_unroll_steps = (
            5  # Number of game moves to keep for every batch element
        )
        self.checkpoint_interval = (
            10  # Number of training steps before using the model for sef-playing
        )
        self.window_size = (
            1000  # Number of self-play games to keep in the replay buffer
        )
        self.td_steps = 10  # Number of steps in the futur to take into account for calculating the target value
        self.training_delay = 0  # Number of seconds to wait after each training to adjust the self play / training ratio to avoid over/underfitting
        self.training_device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        )  # Train on GPU if available

        self.weight_decay = 1e-4  # L2 weights regularization
        self.momentum = 0.9

        # Exponential learning rate schedule
        self.lr_init = 0.008  # Initial learning rate
        self.lr_decay_rate = 0.01
        self.lr_decay_steps = 10000

        ### Test
        self.test_episodes = 2  # Number of game played to evaluate the network

    def visit_softmax_temperature_fn(self, trained_steps):
        """
        Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
        The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

        Returns:
            Positive float.
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25

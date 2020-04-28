import gym
import torch
from .base_classes import MuZeroConfigBase
from environment import PalleteWorld
import env_config as env_cfg
import numpy as np


class MuZeroConfig(MuZeroConfigBase):

    def __init__(self):
        super(MuZeroConfig, self).__init__()
        self.seed = 0  # Seed for numpy, torch and the game

        # Game
        # Dimensions of the game observation
        self.observation_shape = (env_cfg.ENV.ROW_COUNT, env_cfg.ENV.COL_COUNT, 1 + env_cfg.ENV.BIN_MAX_COUNT)
        self.action_space = [i for i in range(env_cfg.ENV.BIN_MAX_COUNT)]  # Fixed list of all possible actions
        self.players = [i for i in range(1)]  # List of players

        # Self-Play
        # Number of simultaneous threads self-playing to feed the replay buffer
        self.num_actors = (10)
        self.max_moves = env_cfg.ENV.EPISODE_MAX_STEP  # Maximum number of moves if game is not finished before
        self.num_simulations = 80  # Number of future moves self-simulated
        self.discount = 0.997  # Chronological discount of the reward
        self.self_play_delay = 0  # Number of seconds to wait after each played game to adjust the self play / training ratio to avoid over/underfitting

        # Root prior exploration noise
        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Network
        self.encoding_size = 200
        self.hidden_size = 128

        # Training
        self.results_path = "./pretrained"  # Path to store the model weights
        self.training_steps = 500000000  # Total number of training steps (ie weights update according to a batch)
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
        self.td_steps = 1  # Number of steps in the future to take into account for calculating the target value
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


class Game:
    """
    Game wrapper.
    """

    def __init__(self, seed=None, id=0):
        self.id = id

        # import pickle
        # l = []
        # with open('env_data.npy', 'rb') as f:
        #     l = pickle.load(f)
        # self.env = PalleteWorld(datasets=l)

        self.env = PalleteWorld(n_random_fixed=1, env_id=id)
        self.episode_step = 0

        if seed is not None:
            self.env.seed(seed)
        print('env {} is created with seed {}'.format(self.id, seed))
            
    def step(self, action):
        """
        Apply action to the game.
        
        Args:
            action : action of the action_space to take.
        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        self.episode_step = self.episode_step + 1
        a = env_cfg.Action(bin_index=action, priority=1, rotate=1)
        o, reward, done, _ = self.env.step(a)

        # this is for visual state representation.
        import torch
        boxes = torch.zeros(10, 10, env_cfg.ENV.BIN_MAX_COUNT)  # x, y, box count
        for i, box in enumerate(o[0]):  # box = x,y
            boxes[0:box[1], 0:box[0], i] = 1
        o = np.concatenate((o[1],boxes), axis=-1)

        print('A agent is taking action {} in {} step in game {}.'.format(action, self.episode_step, self.id))

        # this is for visual state representation.
        # import torch
        # embedding = torch.nn.Embedding(num_embeddings=env_cfg.ENV.BIN_MAX_X_SIZE*env_cfg.ENV.BIN_MAX_X_SIZE+env_cfg.ENV.BIN_MAX_Y_SIZE+1, embedding_dim=1)
        # i = [b[0] * env_cfg.ENV.BIN_MAX_X_SIZE + b[1] for b in o[0]]
        # boxlist = embedding(torch.Tensor(i).long()).squeeze(1).detach().numpy()
        # o = np.concatenate((o[1][:,:,0].flatten(), boxlist))

        return o, reward, done

    def to_play(self):
        """
        Return the current player.

        Returns:
            The current player, it should be an element of the players list in the config. 
        """
        return 0

    def reset(self):
        """
        Reset the game for a new game.
        Returns:
            Initial observation of the game.
        """
        o = self.env.reset()
        print('game {} is reset.'.format(self.id))
        self.episode_step = 0

        # import torch
        # embedding = torch.nn.Embedding(num_embeddings=env_cfg.ENV.BIN_MAX_X_SIZE*env_cfg.ENV.BIN_MAX_X_SIZE+env_cfg.ENV.BIN_MAX_Y_SIZE+1, embedding_dim=1)
        # i = [b[0] * env_cfg.ENV.BIN_MAX_X_SIZE + b[1] for b in o[0]]
        # boxlist = embedding(torch.Tensor(i).long()).squeeze(1).detach().numpy()
        # o = np.concatenate((o[1][:,:,0].flatten(), boxlist))

        # this is for visual state representation.
        import torch
        boxes = torch.zeros(10, 10, env_cfg.ENV.BIN_MAX_COUNT)  # x, y, box count
        for i, box in enumerate(o[0]):  # box = x,y
            boxes[0:box[1], 0:box[0], i] = 1
        o = np.concatenate((o[1],boxes), axis=-1)

        return o

    def close(self):
        """
        Properly close the game.
        """
        self.env.close()

    def render(self):
        """
        Display the game observation.
        """
        self.env.render()
        input("Press enter to take a step ")

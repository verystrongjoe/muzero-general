import copy
import os
import math
import time
from games.base_classes import MuZeroConfigBase
from torch.utils.tensorboard import SummaryWriter
import numpy
import torch

import models


class SelfPlay:
    """
    Class which run in a dedicated thread to play games and save them to the replay-buffer.
    """

    def __init__(self, initial_weights, game, config, test=False, idx=-1, render=False):
        self.config: MuZeroConfigBase = config
        self.game = game
        self.idx = idx
        self.episode = 0
        self.render = render
        self.writer = SummaryWriter(self.config.results_path / f"self_play_{idx}")

        # Initialize the network
        self.model = models.MuZeroExtendedNetwork(
            self.config.observation_shape,
            len(self.config.action_space),
            self.config.encoding_size,
            self.config.hidden_size,
        )
        self.model.set_weights(initial_weights)
        self.model.to(torch.device("cpu"))
        self.model.eval()

        self.continuous_self_play(test)

    def continuous_self_play(self, test_mode=False):
        while True:
            if self.config.v_self_play_count.value > 0:
                # Update the model if the trianer is running
                self.model.set_weights(self.config.q_weights.get())

            # Take the best action (no exploration) in test mode
            temperature = (
                0
                if test_mode
                else self.config.visit_softmax_temperature_fn(
                    trained_steps=self.config.v_training_step.value
                )
            )
            game_history = self.play_game(temperature, False)

            # Save to the shared storage
            score = sum(game_history.rewards)
            self.writer.add_scalars(
                f"1.Total reward/{'test' if test_mode else 'train'}",
                {f"env_{self.idx}": score},
                global_step=self.episode,
            )
            self.episode += 1
            if test_mode:
                self.config.v_total_reward.value = int(score)

            if not test_mode:
                self.config.q_save_game.put(game_history)

            if not test_mode and self.config.self_play_delay:
                time.sleep(self.config.self_play_delay)

    def play_game(self, temperature, render: bool = None):
        """
        Play one game with actions based on the Monte Carlo tree search at each moves.
        """
        if render is None:
            render = self.render
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.observation_history.append(observation)
        done = False

        with torch.no_grad():
            while not done and len(game_history.action_history) < self.config.max_moves:

                root = MCTS(self.config).run(
                    self.model,
                    observation,
                    self.game.to_play(),
                    True if temperature else False,
                    self.game
                )

                action = select_action(root, temperature, self.game)

                observation, reward, done = self.game.step(action)

                if render:
                    self.game.render()

                game_history.observation_history.append(observation)
                game_history.rewards.append(reward)
                game_history.action_history.append(action)
                game_history.store_search_statistics(root, self.config.action_space)

        self.game.close()
        return game_history


def select_action(node, temperature, game):
    """
    Select action according to the visit count distribution and the temperature.
    The temperature is changed dynamically with the visit_softmax_temperature function 
    in the config.
    """
    visit_counts = numpy.array(
        [[child.visit_count, action] for action, child in node.children.items()]
    ).T
    if temperature == 0:
        action_pos = numpy.argmax(visit_counts[0])
    else:
        # See paper appendix Data Generation
        visit_count_distribution = visit_counts[0] ** (1 / temperature)
        visit_count_distribution = visit_count_distribution / sum(
            visit_count_distribution
        )

        # ----------------------------- added -----------------------------
        # def softmax(a):
        #     exp_a = np.exp(a)
        #     sum_exp_a = np.sum(exp_a)
        #     y = exp_a / sum_exp_a
        #
        #     return y

        import numpy as np
        visit_count_distribution[game.env.previous_actions] = -100

        odds = np.exp(visit_count_distribution)
        act_probs = odds / np.sum(odds)

        action_pos = numpy.random.choice(
            len(visit_counts[1]), p=act_probs
        )
        # ----------------------------- added -----------------------------

        # action_pos = numpy.random.choice(
        #     len(visit_counts[1]), p=visit_count_distribution
        # )

    if temperature == float("inf"):
        action_pos = numpy.random.choice(len(visit_counts[1]))

    return visit_counts[1][action_pos]


# Game independant
class MCTS:
    """
    Core Monte Carlo Tree Search algorithm.
    To decide on an action, we run N simulations, always starting at the root of
    the search tree and traversing the tree according to the UCB formula until we
    reach a leaf node.
    """

    def __init__(self, config):
        self.config = config

    def run(self, model, observation, to_play, add_exploration_noise, game):
        """
        At the root of the search tree we use the representation function to obtain a
        hidden state given the current observation.
        We then run a Monte Carlo Tree Search using only action sequences and the model
        learned by the network.
        """
        root = Node(0)
        observation = (
            torch.from_numpy(observation)
            .float()
            .unsqueeze(0)
            .to(next(model.parameters()).device)
        )

        # initial inference
        _, expected_reward, policy_logits, hidden_state = model.initial_inference(
            observation
        )

        root.expand(
            self.config.action_space,
            to_play,
            expected_reward,
            policy_logits,
            hidden_state,
            game.env.previous_actions
        )
        if add_exploration_noise:
            root.add_exploration_noise(
                dirichlet_alpha=self.config.root_dirichlet_alpha,
                exploration_fraction=self.config.root_exploration_fraction,
            )

        min_max_stats = MinMaxStats()

        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]

            ######## Simulation ##########
            while node.expanded():
                action, node = self.select_child(node, min_max_stats)
                last_action = action
                search_path.append(node)

                # Players play turn by turn
                if virtual_to_play + 1 < len(self.config.players):
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            # Inside the search tree we use the dynamics function to obtain the next hidden
            # state given an action and the previous hidden state
            parent = search_path[-2]

            ######## Expansion ##########
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state,
                torch.tensor([[last_action]]).to(parent.hidden_state.device),
                game.env.previous_actions
            )

            node.expand(
                self.config.action_space,
                virtual_to_play,
                reward,
                policy_logits,
                hidden_state,
                game.env.previous_actions
            )

            self.backpropagate(search_path, value.item(), to_play, min_max_stats)

        return root

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        _, action, child = max(
            (self.ucb_score(node, child, min_max_stats), action, child)
            for action, child in node.children.items()
        )
        return action, child

    def ucb_score(self, parent, child, min_max_stats):
        """
        The score for a node is based on its value, plus an exploration bonus based on the prior.
        """
        pb_c = (
            math.log(
                (parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base
            )
            + self.config.pb_c_init
        )
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior
        value_score = min_max_stats.normalize(child.value())

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree to the root.
        """
        for node in search_path:
            node.value_sum += value if node.to_play == to_play else -value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + self.config.discount * value


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state, prev_actions):
    # def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        We expand a node using the value, reward and policy prediction obtained from the neural network.
        """
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy = {a: math.exp(policy_logits[0][a]) for a in actions}

        for p_a in prev_actions:
            policy[p_a] = -100

        policy_sum = sum(policy.values())
        for action, p in policy.items():
            node = Node(p / policy_sum)
            self.children[action] = node

        return action

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction
        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * (1 - frac) + n * frac


class GameHistory:
    """
    Store only usefull information of a self-play game.
    """

    def __init__(self):
        self.observation_history = []
        self.action_history = []
        self.rewards = []
        self.child_visits = []
        self.root_values = []

    def store_search_statistics(self, root, action_space):
        sum_visits = sum(child.visit_count for child in root.children.values())
        # TODO: action could be of any type, not only integers
        self.child_visits.append(
            [
                root.children[a].visit_count / sum_visits if a in root.children else 0
                for a in action_space
            ]
        )
        self.root_values.append(root.value())


class MinMaxStats:
    """
    A class that holds the min-max values of the tree.
    """

    def __init__(self):
        self.maximum = -float("inf")
        self.minimum = float("inf")

    def update(self, value):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value):
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

import numpy
from games.base_classes import MuZeroConfigBase
import threading
import time
from torch.utils.tensorboard import SummaryWriter


class ReplayBuffer:
    """
    Class which run in a dedicated thread to store played games and generate batch.
    """

    def __init__(self, config):
        self.config: MuZeroConfigBase = config
        self.buffer = []
        self.writer = SummaryWriter(self.config.results_path / "self_play")
        self.global_step = 0

        def save_gave_listener():
            while True:
                game_history = self.config.q_save_game.get()
                self.save_game(game_history)

        threading.Thread(target=save_gave_listener).start()

        def batch_pusher():
            while True:
                if self.config.v_self_play_count.value < 1:
                    time.sleep(1)
                else:
                    batch = self.get_batch()
                    self.config.q_replay_batch.put(batch)

        threading.Thread(target=batch_pusher).start()

    def save_game(self, game_history):
        if len(self.buffer) > self.config.window_size:
            self.buffer.pop(0)
        self.buffer.append(game_history)
        self.config.v_self_play_count.value += 1
        self.writer.add_scalar(
            "2.Workers/Self played games",
            self.config.v_self_play_count.value,
            self.global_step,
        )
        self.global_step += 1

    def get_self_play_count(self):
        return self.config.v_self_play_count.value

    def get_batch(self):
        observation_batch, action_batch, reward_batch, value_batch, policy_batch = (
            [],
            [],
            [],
            [],
            [],
        )
        for _ in range(self.config.batch_size):
            game_history = sample_game(self.buffer)
            game_pos = sample_position(game_history)

            value, reward, policy, actions = make_target(
                game_history,
                game_pos,
                self.config.num_unroll_steps,
                self.config.td_steps,
                self.config.discount,
            )

            observation_batch.append(game_history.observation_history[game_pos])
            action_batch.append(actions)
            value_batch.append(value)
            reward_batch.append(reward)
            policy_batch.append(policy)

        return observation_batch, action_batch, value_batch, reward_batch, policy_batch


def sample_game(buffer):
    """
    Sample game from buffer either uniformly or according to some priority.
    """
    # TODO: sample with probability link to the highest difference between real and
    # predicted value (See paper appendix Training)
    batch_size = len(buffer)
    choice = numpy.random.randint(0, batch_size)
    return buffer[choice]


def sample_position(game_history):
    """
    Sample position from game either uniformly or according to some priority.
    """
    # TODO: sample according to some priority
    return numpy.random.choice(range(len(game_history.rewards)))


def make_target(game_history, state_index, num_unroll_steps, td_steps, discount):
    """
    The value target is the discounted root value of the search tree td_steps into the
    future, plus the discounted sum of all rewards until then.
    """
    target_values, target_rewards, target_policies, actions = [], [], [], []
    for current_index in range(state_index, state_index + num_unroll_steps + 1):
        bootstrap_index = current_index + td_steps
        if bootstrap_index < len(game_history.root_values):
            value = game_history.root_values[bootstrap_index] * discount ** td_steps
        else:
            value = 0

        for i, reward in enumerate(game_history.rewards[current_index:bootstrap_index]):
            value += reward * discount ** i

        if current_index < len(game_history.root_values):
            # TODO: Scale and transform the reward and the value, don't forget the invert transform in inference time (See paper appendix Network Architecture)
            # Value target could be scaled by 0.25 (See paper appendix Reanalyze)
            target_values.append(value)
            target_rewards.append(game_history.rewards[current_index])
            target_policies.append(game_history.child_visits[current_index])
            actions.append(game_history.action_history[current_index])
        else:
            # States past the end of games are treated as absorbing states
            target_values.append(0)
            target_rewards.append(0)
            # Uniform policy to give the tensor a valid dimension
            target_policies.append(
                [
                    1 / len(game_history.child_visits[0])
                    for _ in range(len(game_history.child_visits[0]))
                ]
            )
            actions.append(game_history.action_history[-1])

    return target_values, target_rewards, target_policies, actions

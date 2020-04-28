from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from games.bpp import *


# class RedisualLayer(nn.Module):
#
# 	def __init__(self, in_channel, n_filters, kernel_size):
# 		super(RedisualLayer, self).__init__()
#
# 		self.conv2d = ConvLayer(in_channel, n_filters, kernel_size, 2)
# 		self.conv2d_residual = ConvLayer(in_channel, n_filters, kernel_size, 2, False)
# 		self.leaky_relu_final = nn.LeakyReLU()
#
# 	def forward(self, x):
# 		residual = self.conv2d(x)
# 		x = self.conv2d_residual(x)
# 		x = residual + x
# 		x = self.leaky_relu_final(x)
# 		return x
#
#
# class ConvLayer(nn.Module):
#
#     def __init__(self, in_channel, n_filters, kernel_size, padding, use_relu=True):
#         super(ConvLayer, self).__init__()
#         self.in_channel = in_channel
#         self.n_filters = n_filters
#         self.kernel_size = kernel_size
#         self.use_relu = use_relu
#         self.padding = padding
#         # https://ezyang.github.io/convolution-visualizer/index.html
#         # o = [i + 2 * p - k] / s + 1
#         # (o + k) / 2  = p   o = i
#         self.conv2d = nn.Conv2d(in_channel, n_filters, kernel_size, padding=padding, bias=False)
#         if use_relu:
#             self.leaky_relu = nn.LeakyReLU()
#         self.bn_2d = nn.BatchNorm2d(n_filters)
#         # todo : add l2 regularizer with reg_const (weight_decay) in adam optimizer
#
#     def forward(self, x):
#         x = self.conv2d(x)
#
# 		if self.padding == 2:
# 			x = x[:,:,:-1,:-1]
#
# 		x = self.bn_2d(x)
# 		if self.use_relu:
# 			x = self.leaky_relu(x)
# 		return x


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 30, 3, 1)

    def forward(self, x):
        bin = x.permute(0, 3, 1, 2)
        bin = F.relu(self.conv1(bin))
        bin = F.max_pool2d(bin, 2, 2)
        bin = F.relu(self.conv2(bin))
        bin = F.max_pool2d(bin, 2, 2)
        return bin.flatten(start_dim=1)


class FullyConnectedNetwork(nn.Module):
    def __init__(
        self, input_size, layers_sizes, output_size, activation=nn.Tanh()
    ):
        super(FullyConnectedNetwork, self).__init__()
        layers_sizes.insert(0, input_size)
        layers = []
        if 1 < len(layers_sizes):
            for i in range(len(layers_sizes) - 1):
                layers.extend(
                    [
                        nn.Linear(layers_sizes[i], layers_sizes[i + 1]),
                        nn.ReLU(),
                    ]
                )
        layers.append(nn.Linear(layers_sizes[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# TODO: unified residual network
class MuZeroNetwork(nn.Module):
    def __init__(self, observation_size, action_space_size, encoding_size, hidden_size):
        super().__init__()
        self.action_space_size = action_space_size

        # h (representation) : # s,s,s,s,... -> embedding
        self.representation_network = FullyConnectedNetwork(
            observation_size, [], encoding_size
        )
        # g (dynamic function) state part - 1
        # s + a -> next state
        self.dynamics_encoded_state_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size, [hidden_size], encoding_size
        )
        # Gradient scaling (See paper appendix Training)
        self.dynamics_encoded_state_network.register_backward_hook(
            lambda module, grad_i, grad_o: (grad_i[0] * 0.5,)
        )
        # g (dynamic function) reward part - 2
        # s + a -> reward
        self.dynamics_reward_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size, [hidden_size], 1
        )

        # f(prediction) actor part - 1
        self.prediction_policy_network = FullyConnectedNetwork(
            encoding_size, [], self.action_space_size, activation=None
        )
        # f(prediction) critic part - 2
        self.prediction_value_network = FullyConnectedNetwork(
            encoding_size, [], 1, activation=None
        )

    def prediction(self, encoded_state):
        # todo : it doesn't have shared network
        policy_logit = self.prediction_policy_network(encoded_state)
        value = self.prediction_value_network(encoded_state)
        return policy_logit, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        # TODO: Try to remov e tanh activation
        # Scale encoded state between [0, 1] (See appendix paper Training)
        encoded_state = (encoded_state - torch.min(encoded_state)) / (
            torch.max(encoded_state) - torch.min(encoded_state)
        )
        return encoded_state

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with one hot action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)
        # TODO: Try to remove tanh activation
        # Scale encoded state between [0, 1] (See paper appendix Training)
        next_encoded_state = (next_encoded_state - torch.min(next_encoded_state)) / (
            torch.max(next_encoded_state) - torch.min(next_encoded_state)
        )
        reward = self.dynamics_reward_network(x)
        return next_encoded_state, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logit, value = self.prediction(encoded_state)
        return (
            value,
            torch.zeros(len(observation)).to(observation.device),
            policy_logit,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logit, value = self.prediction(next_encoded_state)
        return value, reward, policy_logit, next_encoded_state

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)


import env_config as env_cfg
import numpy as np


class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 3, 1)
        self.conv2 = nn.Conv2d(10, 30, 3, 1)

    def forward(self, x):
        bin = x.permute(0, 3, 1, 2)
        bin = F.relu(self.conv1(bin))
        bin = F.max_pool2d(bin, 2, 2)
        bin = F.relu(self.conv2(bin))
        bin = F.max_pool2d(bin, 2, 2)
        return bin.flatten(start_dim=1)


class StateRepresentation(nn.Module):

    def __init__(self):
        super(StateRepresentation, self).__init__()
        self.conv_item = Convolution()
        self.conv_bin = Convolution()
        self.fc1 = nn.Linear(60, 200)
        # self.fc2 = nn.Linear(200, MuZeroConfig.encoding_size)
        self.fc2 = nn.Linear(200, 200)

        # self.fc3 = nn.Linear(200, env_cfg.ENV.BIN_MAX_COUNT)
        # self.fc4 = nn.Linear(200, 1)
        # self.saved_log_probs = []
        # self.rewards = []

    def forward(self, x):
        batch_dim = x.shape[0]
        n_item = x.shape[3] - 1
        bin = x[:, :, :, :1]
        item = x[:, :, :, 1:]

        bin = self.conv_bin(bin)
        item = item.permute(0, 3, 1, 2).reshape(-1, 10, 10, 1)
        item = self.conv_item(item)
        item = item.reshape(batch_dim, n_item, -1)
        item = torch.sum(item, axis=1)
        concat = torch.cat([item, bin], dim=1)
        x = F.relu(self.fc1(concat))
        x = F.relu(self.fc2(x))

        return x


class RepresentationNetwork(nn.Module):
    def __init__(self):
        super(RepresentationNetwork, self).__init__()
        self.sr = StateRepresentation()
        # self.fc = nn.Linear(200, MuZeroConfig.encoding_size)
        self.fc = nn.Linear(200, 200)

    def forward(self, x):
        x = self.sr(x)
        # todo : activation tanh?
        x = self.fc(x)
        return nn.Tanh()(x)


class PredictionNetwork(nn.Module):
    def __init__(self):
        super(PredictionNetwork, self).__init__()
        # self.sr = StateRepresentation()
        self.fc1 = nn.Linear(200, 200)
        self.fc2 = nn.Linear(200, 200)
        self.fc_a = nn.Linear(200, env_cfg.ENV.BIN_MAX_COUNT)
        self.fc_c = nn.Linear(200, 1)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x, masks, step):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_probs = self.fc_a(x)
        v = self.fc_c(x)

        for i, p in enumerate(masks):
            # for m in p[:step]:
            # todo : check this dimension!!
            # print('{} th action is masked..'.format(i))
            action_probs[0, p] = -np.inf
        return v, F.softmax(action_probs, dim=1)


# class DynamicFunctionNetwork(nn.Module):
#     def __init__(self):
#         super(DynamicFunctionNetwork, self).__init__()
#         self.fc1 = nn.Linear(MuZeroConfig.encoding_size,200)
#         self.fc_next_state = nn.Linear(200, MuZeroConfig.encoding_size)
#         self.fc_reward = nn.Linear(200, 1)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         ns = self.fc_next_state(x)
#         r = self.fc_reward(x)
#
#         return ns, r


# TODO: unified residual network
class MuZeroExtendedNetwork(nn.Module):
    def __init__(self, observation_size, action_space_size, encoding_size, hidden_size):
        super().__init__()
        self.action_space_size = action_space_size

        # h (representation) : # s -> embedding
        self.representation_network = RepresentationNetwork()

        # g (dynamic function) state part - 1
        # s + a -> next state
        self.dynamics_encoded_state_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size, [hidden_size], encoding_size)

        # Gradient scaling (See paper appendix Training)
        self.dynamics_encoded_state_network.register_backward_hook(lambda module, grad_i, grad_o: (grad_i[0] * 0.5,))

        # g (dynamic function) reward part - 2
        # s + a -> reward
        self.dynamics_reward_network = FullyConnectedNetwork(
            encoding_size + self.action_space_size, [hidden_size], 1
        )

        self.prediction_network = PredictionNetwork()

        # # f(prediction) actor part - 1
        # # s -> policy
        # self.prediction_policy_network = FullyConnectedNetwork(
        #     encoding_size, [], self.action_space_size, activation=None
        # )
        # # f(prediction) critic part - 2
        # # s -> value
        # self.prediction_value_network = FullyConnectedNetwork(
        #     encoding_size, [], 1, activation=None
        # )

    def prediction(self, encoded_state, step, previous_actions):
        # todo : it doesn't have shared network
        # policy_logit = self.prediction_policy_network(encoded_state)
        # value = self.prediction_value_network(encoded_state)
        value, policy_logit, = self.prediction_network(encoded_state, previous_actions, step)
        return policy_logit, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        # TODO: Try to remove tanh activation
        # Scale encoded state between [0, 1] (See appendix paper Training)
        encoded_state = (encoded_state - torch.min(encoded_state)) / (
            torch.max(encoded_state) - torch.min(encoded_state)
        )
        return encoded_state

    def dynamics(self, encoded_state, action):
        # Stack encoded_state with one hot action (See paper appendix Network Architecture)
        action_one_hot = (
            torch.zeros((action.shape[0], self.action_space_size))
            .to(action.device)
            .float()
        )
        action_one_hot.scatter_(1, action.long(), 1.0)
        x = torch.cat((encoded_state, action_one_hot), dim=1)

        next_encoded_state = self.dynamics_encoded_state_network(x)
        # TODO: Try to remove tanh activation
        # Scale encoded state between [0, 1] (See paper appendix Training)
        next_encoded_state = (next_encoded_state - torch.min(next_encoded_state)) / (
            torch.max(next_encoded_state) - torch.min(next_encoded_state)
        )
        reward = self.dynamics_reward_network(x)
        return next_encoded_state, reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logit, value = self.prediction(encoded_state, 0, [])
        return (
            value,
            torch.zeros(len(observation)).to(observation.device),
            policy_logit,
            encoded_state,
        )

    def recurrent_inference(self, encoded_state, action, actions):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logit, value = self.prediction(next_encoded_state, len(actions), actions)
        return value, reward, policy_logit, next_encoded_state

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def set_weights(self, weights):
        self.load_state_dict(weights)

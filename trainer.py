import time
from games.base_classes import MuZeroConfigBase
from threading import Thread
from torch.utils.tensorboard import SummaryWriter
import numpy
import torch

import models


class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(self, initial_weights, config):
        self.config: MuZeroConfigBase = config
        self.training_step = 0
        self.writer = SummaryWriter(self.config.results_path / "trainer")

        # Initialize the network
        self.model = models.MuZeroNetwork(
            self.config.observation_shape,
            len(self.config.action_space),
            self.config.encoding_size,
            self.config.hidden_size,
        )
        self.model.set_weights(initial_weights)
        self.model.to(torch.device(config.training_device))
        self.model.train()

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.lr_init,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay,
        )

        def async_put_weights():
            last_idx = None
            while True:
                if self.config.q_weights.empty():
                    if self.training_step != last_idx:
                        weights = self.model.get_weights()
                        last_idx = self.training_step
                    self.config.q_weights.put(weights)
                else:
                    time.sleep(0.1)

        Thread(target=async_put_weights).start()
        self.continuous_update_weights()

    def continuous_update_weights(self):
        # Wait for the replay buffer to be filled
        while self.config.v_self_play_count.value < 1:
            time.sleep(1)

        # Training loop
        while True:
            batch = self.config.q_replay_batch.get()
            total_loss, value_loss, reward_loss, policy_loss = self.update_weights(
                batch
            )

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                self.config.q_weights.put(self.model.get_weights())
            self.config.v_training_step.value = self.training_step
            self.writer.add_scalar(
                "2.Workers/Training steps", self.training_step, self.training_step
            )
            self.writer.add_scalar(
                "3.Loss/1.Total loss", total_loss, self.training_step
            )
            self.writer.add_scalar("3.Loss/Value loss", value_loss, self.training_step)
            self.writer.add_scalar(
                "3.Loss/Reward loss", reward_loss, self.training_step
            )
            self.writer.add_scalar(
                "3.Loss/Policy loss", policy_loss, self.training_step
            )

            if self.config.training_delay:
                time.sleep(self.config.training_delay)

    def update_weights(self, batch):
        """
        Perform one training step.
        """
        self.update_lr()

        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
        ) = batch

        device = next(self.model.parameters()).device
        observation_batch = torch.tensor(observation_batch).float().to(device)
        action_batch = torch.tensor(action_batch).float().to(device).unsqueeze(-1)
        target_value = torch.tensor(target_value).float().to(device)
        target_reward = torch.tensor(target_reward).float().to(device)
        target_policy = torch.tensor(target_policy).float().to(device)

        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )
        predictions = [(value, reward, policy_logits)]
        for action_i in range(self.config.num_unroll_steps):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state, action_batch[:, action_i]
            )
            predictions.append((value, reward, policy_logits))

        # Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        for i, prediction in enumerate(predictions):
            value, reward, policy_logits = prediction
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i, :],
            )
            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

        loss = (value_loss + reward_loss + policy_loss).mean()

        # Scale gradient by number of unroll steps (See paper Training appendix)
        loss.register_hook(lambda grad: grad * 1 / self.config.num_unroll_steps)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def loss_function(
    value, reward, policy_logits, target_value, target_reward, target_policy
):
    # TODO: paper promotes cross entropy instead of MSE
    value_loss = torch.nn.MSELoss()(value, target_value)
    reward_loss = torch.nn.MSELoss()(reward, target_reward)
    policy_loss = torch.mean(
        torch.sum(-target_policy * torch.nn.LogSoftmax(dim=1)(policy_logits), 1)
    )
    return value_loss, reward_loss, policy_loss

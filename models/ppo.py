
import copy

import torch
import torch.nn as nn

class PPO:
    def __init__(
        self,
        policy,
        clip_param,
        update_epoch,
        optimizer,
        actor_loss_coef=1.0,
        value_loss_coef=1.0,
        entropy_coef=0.01,
        max_grad_norm=None,
        clip_value_loss=False,
        optimizer_scheduler=None,
        device=None,
    ):
        """
        PPO算法实现
        Args:
            policy: 神经网络模型
            clip_param (float): clip predicted value range
            update_epoch (int): 更新次数
            optimizer: 优化器，建议默认使用Adam
            actor_loss_coef (float): actor损失函数系数
            value_loss_coef (float): value损失函数系数
            entropy_coef (float): entropy损失函数系数
            max_grad_norm (float or int): max norm of the gradients
            clip_value_loss (bool):
        """

        self.policy = policy
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.clip_param = clip_param
        self.update_epoch = update_epoch
        self.optimizer = optimizer
        self.optimizer_scheduler = optimizer_scheduler

        self.actor_loss_coef = actor_loss_coef
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.clip_value_loss = clip_value_loss
        self.device = device

    def update(self, memories):
        value_loss_record = 0
        action_loss_record = 0
        dist_entropy_record = 0

        for e in range(self.update_epoch):
            loss_sum = 0
            for memory in memories:
                (
                    obs_batch,
                    actions_batch,
                    old_value_preds_batch,
                    return_batch,
                    old_action_log_probs_batch,
                ) = memory.sample(self.device)

                # entropy loss
                p, values = self.policy(obs_batch)
                advantages = return_batch - values.view(-1).detach()
                action_log_probs, dist_entropy = self.evaluate_actions(p, actions_batch)

                # action loss
                ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                action_loss = -torch.min(surr1, surr2).mean()

                # value loss
                if self.clip_value_loss:
                    value_pred_clipped = old_value_preds_batch + (values - old_value_preds_batch).clamp(
                        -self.clip_param, self.clip_param
                    )
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                loss_sum = (
                    value_loss * self.value_loss_coef
                    + action_loss * self.actor_loss_coef
                    - dist_entropy * self.entropy_coef
                )

                value_loss_record += value_loss.item() * self.value_loss_coef
                action_loss_record += action_loss.item() * self.actor_loss_coef
                dist_entropy_record += dist_entropy.item() * self.entropy_coef

            # 求所有数据的loss均值，然后更新网络权重
            self.optimizer.zero_grad()
            loss_sum.mean().backward()
            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        if self.optimizer_scheduler is not None:
            self.optimizer_scheduler.step()

        num_updates = self.update_epoch * len(memories)
        value_loss_record /= num_updates
        action_loss_record /= num_updates
        dist_entropy_record /= num_updates

        return value_loss_record, action_loss_record, -dist_entropy_record

    def evaluate_actions(self, p, actions):
        raise NotImplementedError("请实现该函数")

    def sample_action(self, p, candidate):
        raise NotImplementedError("请实现该函数")

    def greedy_select_action(self, p, candidate):
        raise NotImplementedError("请实现该函数")

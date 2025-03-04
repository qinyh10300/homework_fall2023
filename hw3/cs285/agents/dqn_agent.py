from typing import Sequence, Callable, Tuple, Optional

import torch
from torch import nn, distributions

import numpy as np

import cs285.infrastructure.pytorch_util as ptu


class DQNAgent(nn.Module):
    def __init__(
        self,
        observation_shape: Sequence[int],
        num_actions: int,
        make_critic: Callable[[Tuple[int, ...], int], nn.Module],
        make_optimizer: Callable[[torch.nn.ParameterList], torch.optim.Optimizer],
        make_lr_schedule: Callable[
            [torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler
        ],
        discount: float,
        target_update_period: int,
        use_double_q: bool = False,
        clip_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.critic = make_critic(observation_shape, num_actions)
        self.target_critic = make_critic(observation_shape, num_actions)
        self.critic_optimizer = make_optimizer(self.critic.parameters())
        self.lr_scheduler = make_lr_schedule(self.critic_optimizer)

        self.observation_shape = observation_shape
        self.num_actions = num_actions
        self.discount = discount
        self.target_update_period = target_update_period
        self.clip_grad_norm = clip_grad_norm
        self.use_double_q = use_double_q

        self.critic_loss = nn.MSELoss()

        self.update_target_critic()

    def get_action(self, observation: np.ndarray, epsilon: float = 0.02) -> int:
        """
        Used for evaluation.
        """
        # [None]在张量observation前加入一个新的维度，表示批次batch
        observation = ptu.from_numpy(np.asarray(observation))[None]

        # TODO(student): get the action from the critic using an epsilon-greedy strategy
        # epsilon-greedy策略
        q_values = self.critic(observation)
        best_action_index = torch.argmax(q_values, dim=-1)  # 这里dim是1和-1是一样的
        n, m = q_values.shape  # n = 1
        probs = torch.ones(n, m) * epsilon / (m - 1)   # 全1的张量
        probs[torch.arange(n), best_action_index] = 1 - epsilon
        # print(q_values.shape, torch.arange(n), best_action_index)
        # torch.arange(n)生成一个从0到n-1的张量
        action_distribution = distributions.Categorical(probs=probs)
        action = action_distribution.sample()  # 离散型只有sample方法

        return ptu.to_numpy(action).squeeze(0).item()
        # squeeze(0)去除张量(tensor)中所有维度大小为1的维度；这里是第一个维度batch_size
        # item()把单一的张量变成标量（一个数字）

    def update_critic(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
    ) -> dict:
        """Update the DQN critic, and return stats（统计数据） for logging."""
        (batch_size,) = reward.shape

        # Compute target values
        # 如果不适用Double DQN, 则target网络不参与迭代
        with torch.no_grad():
            # TODO(student): compute target values
            next_qa_values = self.critic(next_obs)

            if self.use_double_q:
                # critic在线网络选择最优动作，target网络计算Q值
                next_action = torch.argmax(next_qa_values, dim=-1)
                next_q_values = self.target_critic(next_obs)[torch.arange(batch_size), next_action]
                # 和下面的写法等价，下面使用了gather函数
                # next_best_action_index = torch.argmax(next_qa_values,-1,keepdim=True)
                # target_next_q_values = self.target_critic(next_obs)
                # next_q_values = torch.gather(target_next_q_values, -1, next_best_action_index).squeeze()
            else:
                next_action = torch.argmax(next_qa_values, dim=-1)
                next_q_values = next_qa_values[torch.arange(batch_size), next_action]
            
            done = done.float()
            target_values = reward + self.discount * (1 - done) * next_q_values

        # TODO(student): train the critic with the target values
        qa_values = self.critic(obs)
        q_values = torch.gather(qa_values, -1, torch.argmax(qa_values, dim=-1, keepdim=True)).squeeze() # Compute from the data actions; see torch.gather
        loss = self.critic_loss(q_values, target_values)


        self.critic_optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.clip_grad_norm or float("inf")
        )
        self.critic_optimizer.step()

        # 更新学习率规划器
        self.lr_scheduler.step()

        return {
            "critic_loss": loss.item(),
            "q_values": q_values.mean().item(),
            "target_values": target_values.mean().item(),
            "grad_norm": grad_norm.item(),
        }

    def update_target_critic(self):
        # 每隔一段步长，将评估网络的参数值给目标网络
        self.target_critic.load_state_dict(self.critic.state_dict())

    def update(
        self,
        obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_obs: torch.Tensor,
        done: torch.Tensor,
        step: int,
    ) -> dict:
        """
        Update the DQN agent, including both the critic and target.
        """
        # TODO(student): update the critic, and the target if needed
        critic_stats = self.update_critic(obs, action, reward, next_obs, done)

        if step % self.target_update_period == 0:
            self.update_target_critic()

        return critic_stats

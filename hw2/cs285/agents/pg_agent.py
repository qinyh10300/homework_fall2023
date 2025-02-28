from typing import Optional, Sequence
import numpy as np
import torch

from cs285.networks.policies import MLPPolicyPG
from cs285.networks.critics import ValueCritic
from cs285.infrastructure import pytorch_util as ptu
from torch import nn


class PGAgent(nn.Module):
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        gamma: float,
        learning_rate: float,
        use_baseline: bool,
        use_reward_to_go: bool,
        baseline_learning_rate: Optional[float],
        baseline_gradient_steps: Optional[int],
        gae_lambda: Optional[float],
        normalize_advantages: bool,
    ):
        super().__init__()

        # create the actor (policy) network
        self.actor = MLPPolicyPG(
            ac_dim, ob_dim, discrete, n_layers, layer_size, learning_rate
        )

        # create the critic (baseline) network, if needed
        if use_baseline:
            self.critic = ValueCritic(
                ob_dim, n_layers, layer_size, baseline_learning_rate
            )
            self.baseline_gradient_steps = baseline_gradient_steps
        else:
            self.critic = None

        # other agent parameters
        self.gamma = gamma
        self.use_reward_to_go = use_reward_to_go
        self.gae_lambda = gae_lambda
        self.normalize_advantages = normalize_advantages

    def update(
        self,
        obs: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[np.ndarray],
        terminals: Sequence[np.ndarray],
    ) -> dict:
        """The train step for PG involves updating its actor using the given observations/actions and the calculated
        qvals/advantages that come from the seen rewards.

        Each input is a list of NumPy arrays, where each array corresponds to a single trajectory. The batch size is the
        total number of samples across all trajectories (i.e. the sum of the lengths of all the arrays).
        """

        # step 1: calculate Q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values: Sequence[np.ndarray] = self._calculate_q_vals(rewards)

        # flatten the lists of arrays into single arrays, so that the rest of the code can be written in a vectorized
        # way. obs, actions, rewards, terminals, and q_values should all be arrays with a leading dimension of `batch_size`
        # beyond this point.
        # 全部变为一维向量，维度为batch_size
        obs = np.concatenate(obs)
        actions = np.concatenate(actions)
        rewards = np.concatenate(rewards)
        terminals = np.concatenate(terminals)
        q_values = np.concatenate(q_values)

        # step 2: calculate advantages from Q values
        advantages: np.ndarray = self._estimate_advantage(
            obs, rewards, q_values, terminals
        )

        # step 3: use all datapoints (s_t, a_t, adv_t) to update the PG actor/policy
        # TODO: update the PG actor/policy network once using the advantages
        info: dict = self.actor.update(obs, actions, advantages)

        # step 4: if needed, use all datapoints (s_t, a_t, q_t) to update the PG critic/baseline
        if self.critic is not None:
            # TODO: perform `self.baseline_gradient_steps` updates to the critic/baseline network
            # 表示每一轮更新几次baseline(critics)网络
            critic_info: dict = {}
            for _ in range(self.baseline_gradient_steps):
                critic_info = self.critic.update(obs, q_values)
                # info['critic_info'] = critic_info

            info.update(critic_info)

        return info

    def _calculate_q_vals(self, rewards: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
        """Monte Carlo estimation of the Q function."""

        if not self.use_reward_to_go:
            # Case 1: in trajectory-based PG, we ignore the timestep and instead use the discounted return for the entire
            # trajectory at each point.
            # In other words: Q(s_t, a_t) = sum_{t'=0}^T gamma^t' r_{t'}
            # TODO: use the helper function self._discounted_return to calculate the Q-values
            q_values: Sequence[np.ndarray] = self._discounted_return(rewards)
        else:
            # Case 2: in reward-to-go PG, we only use the rewards after timestep t to estimate the Q-value for (s_t, a_t).
            # In other words: Q(s_t, a_t) = sum_{t'=t}^T gamma^(t'-t) * r_{t'}
            # TODO: use the helper function self._discounted_reward_to_go to calculate the Q-values
            q_values: Sequence[np.ndarray] = self._discounted_reward_to_go(rewards)
            # 更简单的实现方式是传入每个rewards的reward，得到结果再拼接成list给q_values

        return q_values

    def _estimate_advantage(
        self,
        obs: np.ndarray,
        rewards: np.ndarray,
        q_values: np.ndarray,
        terminals: np.ndarray,
    ) -> np.ndarray:
        """Computes advantages by (possibly) subtracting a value baseline from the estimated Q-values.

        Operates on flat 1D NumPy arrays.
        """
        if self.critic is None:
            # TODO: if no baseline, then what are the advantages?
            # DONE: if no baseline, use q_values as advantages
            advantages = q_values
        else:
            # TODO: run the critic and use it as a baseline
            # 直接调用类实例相当于调用forward方法，因为继承自nn.Module
            values = ptu.to_numpy(self.critic(ptu.from_numpy(obs)).squeeze())
            assert values.shape == q_values.shape

            if self.gae_lambda is None:
                # TODO: if using a baseline, but not GAE, what are the advantages?
                advantages = q_values - values
                # 直接用Q值减去V值就可以了
            else:
                # TODO: implement GAE
                batch_size = obs.shape[0]

                # HINT: append a dummy T+1 value for simpler recursive calculation
                values = np.append(values, [0])
                advantages = np.zeros(batch_size + 1)

                # reversed生成反转序列
                for i in reversed(range(batch_size)):
                    # TODO: recursively compute advantage estimates starting from timestep T.
                    # HINT: use terminals to handle edge cases. terminals[i] is 1 if the state is the last in its
                    # trajectory, and 0 otherwise.
                    # delta_i也有edge_case
                    delta_i = rewards[i] + self.gamma * values[i + 1] * (1 - terminals[i]) - values[i]
                    advantages[i] = delta_i + self.gamma * self.gae_lambda * (1 - terminals[i]) * advantages[i + 1]

                # remove dummy advantage
                advantages = advantages[:-1]

        # TODO: normalize the advantages to have a mean of zero and a standard deviation of one within the batch
        if self.normalize_advantages:
            # 归一化模板
            mean = np.mean(advantages)
            std = np.std(advantages) + 1e-8   #防止除以0
            advantages = (advantages - mean) / std

        return advantages

    def _discounted_return(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns
        a list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}

        Note that all entries of the output list should be the exact same because each sum is from 0 to T (and doesn't
        involve t)!
        """
        # 没有reward_to_go的情况下，一个trajectory上的所有Q-function都是一样的值，和过去、现在、未来有关
        q_values = []
        for k in range(len(rewards)):
            assert len(rewards[k].shape) == 1
            # 断言rewards为一维向量
            single_traj_q_values = np.zeros_like(rewards[k])
            multiply_factor = 1  # 引入乘积因子，方便迭代计算
            single_traj_q_value = 0
            for i in range(rewards[k].shape[0]):
                single_traj_q_value += multiply_factor * rewards[k][i]
                multiply_factor *= self.gamma  # 更新乘积因子
            single_traj_q_values.fill(single_traj_q_value)
            q_values.append(single_traj_q_values)
        return q_values


    def _discounted_reward_to_go(self, rewards: Sequence[float]) -> Sequence[float]:
        """
        Helper function which takes a list of rewards {r_0, r_1, ..., r_t', ... r_T} and returns a list where the entry
        in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}.
        """
        # 有reward_to_go的情况下，同一个trajectory上的Q-function值不一样，只和和现在、未来有关
        q_values = []
        for k in range(len(rewards)):
            assert len(rewards[k].shape) == 1
            # 断言rewards为一维向量
            single_traj_q_values = np.zeros_like(rewards[k])
            multiply_factor = self.gamma  # 引入乘积因子，方便迭代计算
            single_traj_q_values[rewards[k].shape[0]-1] = rewards[k][rewards[k].shape[0]-1]
            for i in range(rewards[k].shape[0]-1-1, -1, -1):
                # 从后往前遍历
                single_traj_q_values[i] = single_traj_q_values[i+1] * multiply_factor + rewards[k][i]
            q_values.append(single_traj_q_values)
        return q_values

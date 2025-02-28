import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu


class MLPPolicy(nn.Module):
    """Base MLP policy, which can take an observation and output a distribution over actions.

    This class should implement the `forward` and `get_action` methods. The `update` method should be written in the
    subclasses, since the policy update rule differs for different algorithms.
    """

    def __init__(
        self,
        ac_dim: int,
        ob_dim: int,
        discrete: bool,
        n_layers: int,
        layer_size: int,
        learning_rate: float,
    ):
        super().__init__()

        if discrete:
            self.logits_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            parameters = self.logits_net.parameters()
        else:
            self.mean_net = ptu.build_mlp(
                input_size=ob_dim,
                output_size=ac_dim,
                n_layers=n_layers,
                size=layer_size,
            ).to(ptu.device)
            self.logstd = nn.Parameter(
                torch.zeros(ac_dim, dtype=torch.float32, device=ptu.device)
            )
            parameters = itertools.chain([self.logstd], self.mean_net.parameters())

        self.optimizer = optim.Adam(
            parameters,
            learning_rate,
        )
        # 优化器一般会放置学习率参数，其中Adam优化器能够在训练中自适应更新学习率

        self.discrete = discrete

    @torch.no_grad()
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        """Takes a single observation (as a numpy array) and returns a single action (as a numpy array)."""
        # TODO: implement get_action
        action_distribution = self.forward(ptu.from_numpy(obs))
        # Using rsample to allow gradients to pass through the sample
        # 离散分布只能使用sample()，通过与one-hot向量进行mse误差。
        # 连续分布可以使用rsample()，也可以使用sample()。因为连续分布无法获取one-hot向量，只能采样action进行反向传播，因此有rsample函数
        # 但是rsample()能够反向传播
        # 在策略梯度PG算法，使用sample和rsample都可以，因为不需要对采样的动作做反向传播
        # 而是对于动作分布进行反向传播
        if self.discrete:
            # 离散动作空间
            action = action_distribution.sample()
        else:
            # 连续动作空间
            action = action_distribution.rsample()

        return ptu.to_numpy(action)

    def forward(self, obs: torch.FloatTensor):
        """
        This function defines the forward pass of the network.  You can return anything you want, but you should be
        able to differentiate through it. For example, you can return a torch.FloatTensor. You can also return more
        flexible objects, such as a `torch.distributions.Distribution` object. It's up to you!
        """
        # forward函数并不需要直接输出采样后的action，这样一来update函数的loss就没法计算了
        # forword函数应该输出一个action_distribution
        if self.discrete:
            # TODO: define the forward pass for a policy with a discrete action space.
            logits = self.logits_net(obs)
            action_distribution = distributions.Categorical(logits=logits)
            # Categorical为用于离散情况下的分类分布
            # action =  action_distribution.rsample()
            # 离散化的分类分布，传入参数probs（未归一化概率），会将其归一化为归一化概率
            # 传入参数logits，先加e指数上变成probs，再变成归一化概率
        else:
            # TODO: define the forward pass for a policy with a continuous action space.
            action_mean = self.mean_net(obs)
            action_std = torch.exp(self.logstd) 
            action_distribution = distributions.Normal(action_mean, action_std)
            # action = action_distribution.rsample()
            # sample()为随机采样，不能反向传播；rsample()能够反向传播
        return action_distribution

    def update(self, obs: np.ndarray, actions: np.ndarray, *args, **kwargs) -> dict:
        """Performs one iteration of gradient descent on the provided batch of data."""
        raise NotImplementedError


class MLPPolicyPG(MLPPolicy):
    """Policy subclass for the policy gradient algorithm."""

    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        advantages: np.ndarray,
    ) -> dict:
        """Implements the policy gradient actor update."""
        obs = ptu.from_numpy(obs)
        actions = ptu.from_numpy(actions)
        advantages = ptu.from_numpy(advantages)

        # TODO: implement the policy gradient actor update.
        action_distribution = self.forward(obs)

        if self.discrete:
            # log_prob方法返回给定动作的对数概率
            # 该方法返回的是一个张量，其中包含每个动作的对数概率值（离散情况下）
            log_prob = action_distribution.log_prob(actions)
        else:
            # 连续分布中，log_prob得到给定动作action的对数概率（概率密度）
            # 由于连续分布，得到的是每个动作维度的对数概率，求和（log里面是乘积）得到整个动作的对数概率
            # 最后一个维度应该就表示每个动作维度
            log_prob = action_distribution.log_prob(actions).sum(dim=-1)

        # 对应位置相乘（向量，矩阵），依然得到一个向量，矩阵；矩阵乘积用@
        loss = -(log_prob * advantages).mean()
        # 计算损失函数，使用负的log_prob乘以优势函数，求平均值.mean()才能得到一个标量损失

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "Actor Loss": ptu.to_numpy(loss),
        }

import torch

# 创建一个示例张量，形状为 (3, 4)
action_probs = torch.tensor([
    [0.1, 0.3, 0.4, 0.2],
])

# 找到每个样本的最大值所在的索引
best_action_index = torch.argmax(action_probs, dim=1)
print(best_action_index, action_probs.shape)  # 输出: tensor([2, 2, 1])
import torch

# 创建一个形状为 (5, 1) 的张量
tensor = torch.tensor([[1], [2], [3], [4], [5]])
tensor = torch.tensor([1,2,3,4,5])
print("Original tensor shape:", tensor.shape)

# 使用 .squeeze() 移除大小为 1 的维度
squeezed_tensor = tensor.squeeze()
print("Squeezed tensor shape:", squeezed_tensor.shape)
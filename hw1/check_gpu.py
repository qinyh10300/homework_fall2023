import torch

if torch.cuda.is_available():
    print("Yes")
else:
    print("False")

print(torch.cuda.is_available(), torch.cuda.current_device())
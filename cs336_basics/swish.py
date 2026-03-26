
import torch
import torch.nn as nn

def standard_swish(x: torch.tensor):
    # 容易溢出，不稳定，用不到 torch 的算子融合；反向传播效率低
    # return x * (1 / (1 + torch.exp(-x)))
    return x * torch.sigmoid(x)

# 3. 稳定版手动实现 (类似底层逻辑)
def silu_stable(x):
    # 当 x < 0 时，使用另一种形式防止 exp(-x) 溢出
    mask = x >= 0
    z = torch.exp(-torch.abs(x))
    # 逻辑略显复杂，这就是为什么我们要用内置函数
    return torch.where(mask, x / (1 + z), x * z / (1 + z))

class TrainableSwish(nn.Module):
    def __init__(self, beta_init: float = 1.0):
        super(TrainableSwish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([beta_init]))
    
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(self.beta * x);
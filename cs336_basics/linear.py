
import torch.nn as nn
import torch

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()
        # 创建权重参数，形状为 (out_features, in_features)
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        # 用截断正态分布初始化: N(0, 2/(d_in + d_out))，截断于 [-3σ, 3σ]
        std = (2.0 / (in_features + out_features)) ** 0.5
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor):
        # y = x @ W^T，等价于 nn.functional.linear(x, W) 但不使用它
        return x @ self.weight.T


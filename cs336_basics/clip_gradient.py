from collections.abc import Iterable

import torch.nn as nn
import torch
from torch import Tensor


def clip_gradient(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    # 收集有梯度的参数（需要遍历两次，所以先存成 list）
    grads = [p.grad.data for p in parameters if p.grad is not None]

    # 计算所有梯度的 combined L2 norm: sqrt(Σ ||g_i||^2)
    total_norm = torch.sqrt(sum(torch.sum(g * g) for g in grads))

    # 只在超过阈值时裁剪：g *= max_norm / total_norm
    if total_norm > max_l2_norm:
        scale = max_l2_norm / total_norm
        for g in grads:
            g.mul_(scale)




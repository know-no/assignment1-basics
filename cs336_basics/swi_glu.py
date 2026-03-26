

import torch
import torch.nn as nn

from swish import TrainableSwish
from glu import GLU

from cs336_basics.swish import standard_swish 

class SwiGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.randn(in_features, out_features))
        self.u = nn.Parameter(torch.randn(in_features, out_features))
        
        nn.init.xavier_normal(self.w)
        nn.init.xavier_normal(self.u)

    def forward(self, x: torch.Tensor):
        # 1. 计算左半部分：Swish(xW)
        gate_linear = x @ self.w
        gate_activated = standard_swish(gate_linear)
        # 2. 计算右半部分：xV
        content_linear = x @ self.u

        #GLU 核心（门控相乘）组合：SwiGLU = Swish(xW) * xV
        return gate_activated * content_linear

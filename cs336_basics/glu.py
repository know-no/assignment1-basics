
import math
import torch
import torch.nn as nn


class GLU(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.weight_gate = nn.Parameter(torch.randn(in_features, out_features))
        self.weight_up = nn.Parameter(torch.randn(in_features, out_features))

        # llm 里经常设置为 none
        self.bias_gate = nn.Parameter(torch.zeros(out_features))
        self.bias_up = nn.Parameter(torch.zeros(out_features))

        # 手动初始化， randn 容易梯度爆炸
        nn.init.xavier_uniform(self.weight_gate)
        nn.init.xavier_uniform(self.weight_up)

    def forward(self, x):
        gate_logit = x @ self.weight_gate + self.bias_gate
        up_logit = x @ self.weight_gate + self.bias_up

        return torch.sigmoid(gate_logit) * up_logit

# 这个实现每次都会启用两个 parameter， 会增大显存
# 每次矩阵乘法都会启动一个独立的 CUDA Kernel。两个小矩阵乘法的 Kernel Launch Overhead（启动开销）和显存访问开销较大。
# 在 vLLM 这种追求极致吞吐的项目中，我们会把 weight_gate 和 weight_up 拼接成一个大的 Parameter
# 用一次大的矩阵乘法（GEMM）解决，这被称为 Weight Concatenation。

class OptimizedGLU(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__
        self.combined_weight = nn.Parameter(torch.empty(in_features, 2* out_features))

        # kaiming or xavier
        nn.init.kaiming_uniform(self.combined_weight, a = math.sqrt(5))

    def forward(self, x):
        combined_output = x @ self.combined_weight

        gate_act, content = torch.chunk(combined_output, 2 ,dim = -1)
        return torch.sigmoid(gate_act, content)

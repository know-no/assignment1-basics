
import math
import torch
import torch.nn as nn

class StrictRoPE(nn.Module):
    def __init__(self, theta:float, d_k:int, max_seq_len: int, device: torch.device | None =None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.device = device
        # 1. 计算 d k / 2 个衰减频率
        inv_freq = 1 /(self.theta ** (torch.arange(0, self.d_k, 2, dtype=torch.float32, device=device) / self.d_k))

        t = torch.arange(0, self.max_seq_len, 1, dtype=torch.float32, device=device)

        # 频率矩阵, 本质是一种多分辨率编码
        freqs = torch.outer(t, inv_freq)

        freqs_emb = torch.repeat_interleave(freqs, 2, dim=-1)

        self.register_buffer("cos_cached", freqs_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_emb.sin(), persistent=False)

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        while cos.dim() < x. dim(): # 不出意外 ，cos 是两个维度， 而x 可能是三个维度，batch，seq len， dim
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        def rotate_adjacent(tensor: torch.Tensor) -> torch.Tensor:
            x_even = tensor[..., 0::2]
            x_odd = tensor[..., 1::2]

            rotated_pairs = torch.stack((-x_odd, x_even), dim = -1)

            return rotated_pairs.reshape(tensor.shape)
        
        x_rotated = (x * cos) + (rotate_adjacent(x) * sin)
        return x_rotated


# todo 过不了测试用例， 需要修改；因为测试用例里是： 是“相邻偶奇位成对旋转”的实现约定。
class OptimizedRoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # 1. 计算维度相关的衰减频率: \theta^{-2(k-1)/d_k}
        # k 取值对应 [0, 2, 4, ..., d_k-2]，共有 d_k / 2 个频率
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        
        # 2. 生成绝对位置序列 i
        t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        
        # 3. 外积计算 i * \theta^{-2(k-1)/d_k}，形状为 (max_seq_len, d_k / 2)
        freqs = torch.outer(t, inv_freq)
        
        # 4. 拼接复制成 (max_seq_len, d_k) 的形状，一半用于前 d_k/2 特征，一半用于后 d_k/2
        freqs_emb = torch.cat((freqs, freqs), dim=-1)
        
        # 5. 预先计算 cos 和 sin，并将其注册为缓冲区 (Buffer)
        # persistent=False 表示这些变量不需要保存在模型 state_dict (权重) 中
        self.register_buffer("cos_cached", freqs_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_emb.sin(), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: 形状通常为 (batch_size, [num_heads], seq_len, d_k)
        token_positions: 形状通常为 (batch_size, seq_len)
        """
        # 第一步：切片/查找 (Slicing)
        # 利用 PyTorch 的高级索引机制，直接使用 token_positions 提取所需的 cos/sin
        # 提取后的张量形状将变为 (*token_positions.shape, d_k)
        # 例如：如果 token_positions 是 (B, seq_len)，提取结果就是 (B, seq_len, d_k)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]
        
        # 第二步：维度对齐 (Broadcasting Alignment)
        # 如果 x 包含额外的维度（例如 Multi-Head Attention 中的 num_heads 维度），
        # x 的形状可能是 (B, num_heads, seq_len, d_k)。
        # 我们需要在 cos 和 sin 的 seq_len (倒数第2维) 之前插入占位维度 (unsqueeze)。
        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
            
        # 第三步：应用旋转 (Apply Rotation)
        # 辅助函数：将 x 沿着最后一个维度切成两半，并进行互换取反操作
        def rotate_half(tensor: torch.Tensor) -> torch.Tensor:
            x1 = tensor[..., : self.d_k // 2]
            x2 = tensor[..., self.d_k // 2 :]
            return torch.cat((-x2, x1), dim=-1)
            
        # 根据 RoPE 核心公式: x_rotated = (x * cos) + (rotate_half(x) * sin)
        x_rotated = (x * cos) + (rotate_half(x) * sin)
        
        return x_rotated


class OptimizedRoPEFixed(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device: torch.device | None = None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError(f"d_k must be even for RoPE, got {d_k}")

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(positions, inv_freq)

        # Match snapshot behavior: each (2i, 2i+1) pair shares the same phase.
        freqs_emb = torch.repeat_interleave(freqs, repeats=2, dim=-1)
        self.register_buffer("cos_cached", freqs_emb.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs_emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        while cos.dim() < x.dim():
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated = torch.empty_like(x)
        rotated[..., 0::2] = -x_odd
        rotated[..., 1::2] = x_even
        return (x * cos) + (rotated * sin)

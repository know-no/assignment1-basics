import torch
import torch.nn as nn
from cs336_basics.rope import StrictRoPE, OptimizedRoPE


# 这个命令可以只跑不带 rope 的测试，uv run pytest -k "test_multihead_self_attention and not with_rope" -q
class CasualMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.wq = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wk = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wv = nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.wo = nn.Parameter(torch.randn(self.d_model, self.d_model))
        # 已经用 randn 这个有初始分布的方法创建
        # 还可以用torch empty，再用xavier 或者 trunc normal 初始化


    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        *batch_like, seq_len, _ = x.shape
        x = x.view(*batch_like, seq_len, self.num_heads, self.head_dim)
        return x.transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        *batch_like, _, seq_len, _ = x.shape
        x = x.transpose(-3, -2).contiguous()
        return x.view(*batch_like, seq_len, self.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        *batch_like, seq_len, _ = x.shape

        q = x @ self.wq.T
        k = x @ self.wk.T
        v = x @ self.wv.T

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = attn_probs @ v

        out = self._merge_heads(context)
        return out @ self.wo.T


class CasualMultiHeadSelfAttentionRoPE(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float, max_seq_len: int, context_length: int | None =None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = self.d_model // self.num_heads
        self.context_length = context_length
        self.rope = StrictRoPE(theta=theta, d_k = self.head_dim, max_seq_len=max_seq_len)

        self.wq = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wk = nn.Parameter(torch.randn(self.d_model, self.d_model))
        self.wv = nn.Parameter(torch.randn(self.d_model, self.d_model))

        self.wo = nn.Parameter(torch.randn(self.d_model, self.d_model))
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., seq_len, d_model) -> (..., num_heads, seq_len, head_dim)
        *batch_like, seq_len, _ = x.shape
        x = x.view(*batch_like, seq_len, self.num_heads, self.head_dim)
        return x.transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (..., num_heads, seq_len, head_dim) -> (..., seq_len, d_model)
        *batch_like, _, seq_len, _ = x.shape
        x = x.transpose(-3, -2).contiguous()
        return x.view(*batch_like, seq_len, self.d_model)

    
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, token_positions: torch.Tensor | None = None):
        *batch_like, seq_len, _ = x.shape

        q = x @ self.wq.T
        k = x @ self.wk.T
        v = x @ self.wv.T

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        else:
            token_positions = token_positions.to(device=x.device, dtype=torch.long)

        q = self.rope.forward(q, token_positions)
        k = self.rope.forward(k, token_positions)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)

        if mask is None:
            mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        if self.context_length is not None and self.context_length < q.shape[-2] \
            and self.context_length > 0: # > 0, 是某些人的实现会把 none 变成负数传递进来
            q_len = q.shape[-2]
            k_len = k.shape[-2]
            row_idx = (k_len - q_len) + torch.arange(q_len, device=q.device)
            col_idx = torch.arange(k.shape[-2], device=q.device)
            dist = row_idx[:, None] - col_idx[None, :] # (q_len, k_len)
            local_mask = (dist >= 0) & (dist < self.context_length)
            # dist = row_idx.unsqueeze(1) - col_idx.unsqueeze(0)
            # local_mask = (dist >= 0) & (dist < self.context_length)
            mask = mask & local_mask
        attn_scores = attn_scores.masked_fill(~mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        context = attn_probs @ v

        out = self._merge_heads(context)
        return out @ self.wo.T

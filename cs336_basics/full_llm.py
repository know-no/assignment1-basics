
import torch
import torch.nn as nn

from cs336_basics.ffn import FFN
from cs336_basics.casual_multi_head_self_attention import CasualMultiHeadSelfAttentionRoPE
from cs336_basics.rms_norm import RMSNorm
from cs336_basics.embedding import Embedding
from cs336_basics.linear import Linear

class PreNormTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, context_length: int | None = None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = self.d_model // self.num_heads
        self.theta = theta
        self.max_seq_len = max_seq_len
        if context_length is None:
            self.context_length = self.max_seq_len
        else:
            self.context_length = context_length
        self.casual_attention = CasualMultiHeadSelfAttentionRoPE(
            self.d_model, self.num_heads, self.theta, self.max_seq_len , self.context_length
        )
        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)
        self.ffn = FFN(self.d_model, self.d_ff)

    def forward(self, x: torch.Tensor):
        h = x + self.casual_attention(self.ln1(x))
        return h + self.ffn(self.ln2(h))

class SimpleTransformer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, theta: float, max_seq_len: int, vocab_size: int, context_length: int, num_layers: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.head_dim = self.d_model // self.num_heads
        self.theta = theta
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.embedding = Embedding(self.vocab_size, self.d_model)

        self.attention_layers = nn.ModuleList([PreNormTransformerBlock(self.d_model, self.num_heads, self.d_ff, self.theta, self.max_seq_len, self.context_length) for i in range(self.num_layers)])

        self.lm_final = RMSNorm(self.d_model)
        self.lm_head = Linear(self.d_model, self.vocab_size)


    def forward(self, x: torch.Tensor):
        if x.shape[-1] > self.context_length:
            raise ValueError(
                f"input sequence_length={x.shape[-1]} exceeds context_length={self.context_length}"
            )

        embed = self.embedding.forward(x)

        h = embed
        for block in self.attention_layers:
            h = block(h)

        rms = self.lm_final.forward(h)
        logits = self.lm_head.forward(rms)

        return logits

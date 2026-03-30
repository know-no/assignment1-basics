
import torch.nn as nn
import torch

class Embedding(nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim: int, device=None, dtype=None):
        super(Embedding, self).__init__()
        self.num_embeddings = num_embeddings;
        self.embedding_dim =  embedding_dim;
        # 没必要这样初始化
        self.weights = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        # self.weights = nn.Parameter(torch.randn(num_embeddings, embedding_dim, device=device, dtype=dtype))
        std = (2.0 / (num_embeddings + embedding_dim)) ** 0.5
        torch.nn.init.trunc_normal_(self.weights, mean=0.0, std=std, a=-3.0*std, b= 3.0*std)

    def forward(self, token_ids: torch.Tensor):
        return self.weights[token_ids]
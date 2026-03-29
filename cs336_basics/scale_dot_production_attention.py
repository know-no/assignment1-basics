
import math
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int


def soft_max_stable(v: torch.Tensor):
    v_max = torch.max(v, dim=-1, keepdim=True).values
    exp_v = torch.exp(v - v_max)
    return exp_v / torch.sum(exp_v, dim=-1, keepdim=True)

# 过不了测试，测视里就要求要使用 减去最大值 来增加稳定性;  所以，我把 float 转成 64 了
# test_softmax_matches_pytorch 里面 v+100, torch.exp(v+100) 就会变成 nan
def soft_max_normal(v: torch.Tensor):
    x = v.to(torch.float64)
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=-1, keepdim=True)


def soft_max_stable_dim(v: torch.Tensor, dim:int):
    v_max = torch.max(v, dim=dim, keepdim=True).values
    exp_v = torch.exp(v - v_max)
    return exp_v / torch.sum(exp_v, dim=dim, keepdim=True)

# 过不了测试，测视里就要求要使用 减去最大值 来增加稳定性;  所以，我把 float 转成 64 了
# test_softmax_matches_pytorch 里面 v+100, torch.exp(v+100) 就会变成 nan
def soft_max_normal_dim(v: torch.Tensor, dim:int):
    x = v.to(torch.float64)
    exp_x = torch.exp(x)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def scale_dot_production_attention(
    Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... values d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None):
    print("\n")
    print("Q: ", Q.shape)
    print("K: ", K.shape)
    print("V: ", V.shape)
    if mask is not None:
        print("mask: ", mask.shape)


    d_k = Q.shape[-1]

    attn = Q @ K.transpose(-2,-1) / math.sqrt(d_k)

    print("attn1: ", attn.shape)
    sp = attn.shape
    if mask is not None:
        attn = attn.masked_fill(~mask, float("-inf"))
    print("attn2c: ", attn.shape)

    attn = torch.softmax(attn, dim = -1)
    v = attn @ V
    print("v :", v.shape)
    return v


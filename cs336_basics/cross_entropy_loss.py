
import torch
import torch.nn as nn
from torch import Tensor
from jaxtyping import Bool, Float, Int




# def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
#     # log-softmax trick: log(softmax(x)_i) = x_i - max(x) - log(Σ exp(x_j - max(x)))
#     max_vals = inputs.max(dim=-1, keepdim=True).values    # 数值稳定
#     shifted = inputs - max_vals                           # (batch_size, vocab_size)
#     log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))  # (batch_size,)
#     target_logits = inputs[torch.arange(inputs.shape[0]), targets]  # 真实类别的 logit
#     loss_per_example = -target_logits + max_vals.squeeze(-1) + log_sum_exp
#     return loss_per_example.mean()

def cross_entropy_loss(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]):
    # print()
    # print(inputs.shape)
    max_vals = inputs.max(dim = -1, keepdim=True).values
    shifted = inputs - max_vals

    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim = -1))
    # print()
    # print(log_sum_exp.shape)

    targets_logits = inputs[torch.arange(inputs.shape[-2]), targets]
    # print(targets_logits.shape)

    # print(max_vals.shape)
    # print(max_vals.squeeze(-1).shape)
    loss_per_example = -targets_logits + max_vals.squeeze(-1) + log_sum_exp

    return loss_per_example.mean()


import torch.nn as nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, d_model:int ,eps:float=1e-5, device: torch.device|None=None, dtype: torch.dtype | None=None):
        super(RMSNorm,self).__init__()
        self.d_model:int = d_model
        self.eps:float = eps
        self.device = device
        self.dtype = dtype

        # gi 是可学习的参数
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True) + self.eps)
        rms_norm = (x / rms ) * self.weight
        return rms_norm.to(in_dtype)





        
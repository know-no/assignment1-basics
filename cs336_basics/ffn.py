
import torch.nn as nn
import torch

from cs336_basics.swish import standard_swish 

class FFN(nn.Module):
    def __init__(self, d_model:int, dff:int | None):
        super().__init__()
        self.d_model = d_model
        self.dff = dff if dff else self.d_model / 3 * 8
        
        self.w1 = nn.Parameter(torch.randn(self.d_model, self.dff))
        self.w3 = nn.Parameter(torch.randn(self.d_model, self.dff))

        self.w2 = nn.Parameter(torch.randn(self.dff, self.d_model))


    def forward(self, x:torch.Tensor):
        
        gate_linear = x @ self.w1
        gate = standard_swish(gate_linear)

        up = x @ self.w3

        intermediate = gate * up

        return intermediate @ self.w2
        
        


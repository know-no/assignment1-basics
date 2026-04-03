import torch.nn as nn
import torch
from torch import Tensor
import math


def schedule_learning_rate(t:int, a_max:float, a_min:float, Tw: float, Tc: float):
    if t < Tw:
        return a_max * (t / Tw)
    elif t >= Tw and t <= Tc:
        second = 0.5 * (math.cos(math.pi * ((t- Tw)/ (Tc-Tw))) + 1)
        return a_min + (a_max - a_min) * second
    else:
        return a_min


        




import torch
import torch.nn as nn

from torch import Tensor
import os
from typing import IO, Any, BinaryIO
import mmap


def save(model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)




def load(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    if isinstance(src, (str, os.PathLike)):
        with open(src, "rb") as f:
            with mmap.mmap(f.fileno(), length=0, access=mmap.ACCESS_READ) as mm:
                checkpoint = torch.load(mm, map_location="cpu")
    else:
        checkpoint = torch.load(src, map_location="cpu")

    model_state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    optimizer_state = checkpoint.get("optimizer_state_dict", checkpoint.get("optimizer"))
    if model_state is None or optimizer_state is None:
        raise KeyError("Checkpoint must contain model and optimizer states.")

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return int(checkpoint["iteration"])

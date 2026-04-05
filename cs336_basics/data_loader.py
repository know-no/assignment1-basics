import torch


def load_data(x, batch_size: int, context_length: int, device: str):
    """Sample language-modeling training batches from a 1D token dataset.

    Returns:
        x_batch: (batch_size, context_length)
        y_batch: (batch_size, context_length), shifted by +1 token from x_batch
    """
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    data = torch.as_tensor(x, dtype=torch.long, device=device)
    if data.ndim != 1:
        raise ValueError("x must be a 1D sequence of token IDs")
    if data.numel() <= context_length:
        raise ValueError("dataset length must be greater than context_length")

    max_start = data.numel() - context_length
    starts = torch.randint(0, max_start, (batch_size,), device=device)
    offsets = torch.arange(context_length, device=device)
    # print(offsets)
    # 先将
    positions = starts.unsqueeze(1) + offsets.unsqueeze(0)
    # print(positions[0])
    x_batch = data[positions]
    y_batch = data[positions + 1]
    return x_batch, y_batch

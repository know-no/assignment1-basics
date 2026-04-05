from __future__ import annotations

from collections.abc import Sequence
import importlib

import torch

# Toggle switch:
# - True: prefer your custom implementation; fallback to torch if import fails.
# - False: force torch implementation.
USE_CUSTOM_SOFTMAX = True


def _import_custom_softmax():
    for module_name in ["cs336_basics.scale_dot_production_attention", "scale_dot_production_attention"]:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, "soft_max_stable", None)
        except Exception:
            continue
    return None


_custom_softmax = _import_custom_softmax() if USE_CUSTOM_SOFTMAX else None


def _sample_from_logits(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    generator: torch.Generator | None = None,
) -> int:
    """Sample one token id from 1D logits with temperature and top-p sampling."""
    if logits.ndim != 1:
        raise ValueError(f"logits must be 1D, got shape={tuple(logits.shape)}")
    if temperature < 0:
        raise ValueError("temperature must be >= 0")
    if not (0 < top_p <= 1.0):
        raise ValueError("top_p must be in (0, 1]")

    # tau -> 0 becomes greedy decoding.
    if temperature == 0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature
    if _custom_softmax is not None:
        probs = _custom_softmax(scaled_logits)
    else:
        probs = torch.softmax(scaled_logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Keep largest tokens until reaching top_p (inclusive of crossing token).
        keep_mask = (cumulative_probs - sorted_probs) < top_p
        keep_mask[0] = True

        filtered_probs = torch.zeros_like(probs)
        kept_probs = sorted_probs[keep_mask]
        kept_indices = sorted_indices[keep_mask]
        filtered_probs[kept_indices] = kept_probs
        probs = filtered_probs / filtered_probs.sum()

    next_token = torch.multinomial(probs, num_samples=1, generator=generator)
    return int(next_token.item())


@torch.no_grad()
def decode(
    model: torch.nn.Module,
    prompt_token_ids: Sequence[int],
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    """
    Autoregressively decode from a language model.

    Args:
        model: Causal LM mapping (batch, seq_len) -> (batch, seq_len, vocab_size).
        prompt_token_ids: Prompt token IDs x_1...x_t.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: If generated, stop decoding early.
        temperature: Softmax temperature; 0 means greedy decoding.
        top_p: Nucleus sampling threshold in (0, 1].
        device: Device to run model on. Defaults to model parameter device.
        generator: Optional torch.Generator for reproducible sampling.

    Returns:
        Full token list: prompt + generated continuation.
    """
    if len(prompt_token_ids) == 0:
        raise ValueError("prompt_token_ids must not be empty")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    # Use model context length if available.
    context_length = getattr(model, "context_length", None)
    if context_length is not None and context_length <= 0:
        context_length = None

    output_ids = [int(t) for t in prompt_token_ids]
    was_training = model.training
    model.eval()

    try:
        for _ in range(max_new_tokens):
            if context_length is None:
                input_ids = output_ids
            else:
                input_ids = output_ids[-context_length:]

            x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
            logits = model(x)  # (1, seq_len, vocab_size)
            next_token_logits = logits[0, -1, :]

            next_token_id = _sample_from_logits(
                next_token_logits,
                temperature=temperature,
                top_p=top_p,
                generator=generator,
            )
            output_ids.append(next_token_id)

            if eos_token_id is not None and next_token_id == eos_token_id:
                break
    finally:
        if was_training:
            model.train()

    return output_ids


@torch.no_grad()
def decode_text(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    eos_token: str = "<|endoftext|>",
    temperature: float = 1.0,
    top_p: float = 1.0,
    device: str | torch.device | None = None,
    generator: torch.Generator | None = None,
) -> str:
    """
    Text wrapper around `decode`.

    The tokenizer is expected to expose:
      - encode(str) -> list[int]
      - decode(list[int]) -> str
      - bytes_to_id: dict[bytes, int] (for eos_token lookup)
    """
    prompt_ids = tokenizer.encode(prompt)
    eos_token_id = tokenizer.bytes_to_id.get(eos_token.encode("utf-8"))

    out_ids = decode(
        model=model,
        prompt_token_ids=prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        top_p=top_p,
        device=device,
        generator=generator,
    )
    return tokenizer.decode(out_ids)

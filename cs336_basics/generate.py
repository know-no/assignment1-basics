from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

try:
    from cs336_basics.full_llm import SimpleTransformer
    from cs336_basics.generation import decode_text
    from cs336_basics.tokenizer import Tokenizer
except Exception:  # pragma: no cover
    from full_llm import SimpleTransformer
    from generation import decode_text
    from tokenizer import Tokenizer

# .venv/bin/python cs336_basics/generate.py \
#   --prompt "Once upon a time" \
#   --checkpoint-path checkpoints/tinystories.pt \
#   --config-path checkpoints/train_config.json \
#   --tokenizer byte

# .venv/bin/python cs336_basics/generate.py \
#   --interactive \
#   --checkpoint-path checkpoints/tinystories.pt \
#   --config-path checkpoints/train_config.json \
#   --tokenizer byte

# .venv/bin/python cs336_basics/generate.py \
#   --checkpoint-path checkpoints/latest.pt \
#   --config-path checkpoints/train_config.json \
#   --prompt "Once upon a time" \
#   --tokenizer auto




class ByteTokenizer:
    """Byte-level tokenizer for vocab_size=256 style models."""

    def __init__(self):
        self.bytes_to_id = {bytes([i]): i for i in range(256)}

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def decode(self, token_ids: list[int]) -> str:
        return bytes(int(t) % 256 for t in token_ids).decode("utf-8", errors="replace")


def gpt2_bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, [chr(n) for n in cs]))


def load_bpe_tokenizer(vocab_path: Path, merges_path: Path, special_tokens: list[str]) -> Tokenizer:
    gpt2_byte_decoder = {v: k for k, v in gpt2_bytes_to_unicode().items()}

    with vocab_path.open(encoding="utf-8") as f:
        gpt2_vocab = json.load(f)

    gpt2_bpe_merges: list[tuple[str, str]] = []
    with merges_path.open(encoding="utf-8") as f:
        for line in f:
            cleaned = line.rstrip()
            parts = cleaned.split(" ")
            if cleaned and len(parts) == 2:
                gpt2_bpe_merges.append((parts[0], parts[1]))

    vocab = {
        token_id: bytes([gpt2_byte_decoder[token] for token in token_text])
        for token_text, token_id in gpt2_vocab.items()
    }

    if special_tokens:
        existing = set(vocab.values())
        for token in special_tokens:
            b = token.encode("utf-8")
            if b not in existing:
                vocab[len(vocab)] = b
                existing.add(b)

    merges = [
        (
            bytes([gpt2_byte_decoder[t] for t in t1]),
            bytes([gpt2_byte_decoder[t] for t in t2]),
        )
        for t1, t2 in gpt2_bpe_merges
    ]

    return Tokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained SimpleTransformer checkpoint.")

    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/tinystories.pt"))
    parser.add_argument("--config-path", type=Path, default=Path("checkpoints/train_config.json"))
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--tokenizer", choices=["auto", "byte", "bpe"], default="auto")
    parser.add_argument("--vocab-path", type=Path, default=None)
    parser.add_argument("--merges-path", type=Path, default=None)
    parser.add_argument("--special-tokens", nargs="*", default=None)
    parser.add_argument("--eos-token", type=str, default="<|endoftext|>")

    # Optional model overrides (if not provided, read from config JSON).
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--num-heads", type=int, default=None)
    parser.add_argument("--d-ff", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--theta", type=float, default=None)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)

    args = parser.parse_args()

    if args.max_new_tokens < 0:
        parser.error("--max-new-tokens must be >= 0")
    if not args.interactive and not args.prompt:
        parser.error("Either provide --prompt for one-shot generation, or use --interactive.")

    return args


def _get_param(args: argparse.Namespace, cfg: dict[str, Any], cli_key: str, cfg_key: str) -> Any:
    val = getattr(args, cli_key)
    if val is not None:
        return val
    if cfg_key in cfg:
        return cfg[cfg_key]
    raise ValueError(f"Missing required model parameter: {cli_key} (or {cfg_key} in config)")


def load_config(config_path: Path) -> dict[str, Any]:
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def build_model(args: argparse.Namespace, cfg: dict[str, Any]) -> SimpleTransformer:

    model = SimpleTransformer(
        d_model=int(_get_param(args, cfg, "d_model", "d_model")),
        num_heads=int(_get_param(args, cfg, "num_heads", "num_heads")),
        d_ff=int(_get_param(args, cfg, "d_ff", "d_ff")),
        theta=float(_get_param(args, cfg, "theta", "theta")),
        max_seq_len=int(_get_param(args, cfg, "max_seq_len", "max_seq_len")),
        vocab_size=int(_get_param(args, cfg, "vocab_size", "vocab_size")),
        context_length=int(_get_param(args, cfg, "context_length", "context_length")),
        num_layers=int(_get_param(args, cfg, "num_layers", "num_layers")),
    ).to(args.device)

    checkpoint = torch.load(args.checkpoint_path, map_location=args.device)
    model_state = checkpoint.get("model_state_dict", checkpoint.get("model"))
    if model_state is None:
        raise KeyError("Checkpoint does not contain model_state_dict/model")
    model.load_state_dict(model_state)
    model.eval()
    return model


def build_tokenizer(args: argparse.Namespace, cfg: dict[str, Any]):
    tokenizer_mode = args.tokenizer
    if tokenizer_mode == "auto":
        tokenizer_mode = cfg.get("tokenizer_mode", "byte")

    if tokenizer_mode == "byte":
        return ByteTokenizer()
    if tokenizer_mode != "bpe":
        raise ValueError(f"Unsupported tokenizer mode: {tokenizer_mode}")

    vocab_path = args.vocab_path or (Path(cfg["tokenizer_vocab_path"]) if cfg.get("tokenizer_vocab_path") else None)
    merges_path = args.merges_path or (Path(cfg["tokenizer_merges_path"]) if cfg.get("tokenizer_merges_path") else None)
    special_tokens = args.special_tokens if args.special_tokens is not None else cfg.get("tokenizer_special_tokens", ["<|endoftext|>"])

    if vocab_path is None or merges_path is None:
        raise ValueError(
            "BPE tokenizer requires vocab/merges paths. "
            "Provide --vocab-path/--merges-path or ensure train_config.json has tokenizer_vocab_path/tokenizer_merges_path."
        )
    return load_bpe_tokenizer(vocab_path, merges_path, special_tokens)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = load_config(args.config_path)
    model = build_model(args, cfg)
    tokenizer = build_tokenizer(args, cfg)

    if args.interactive:
        print("Interactive mode. Type prompt and press enter. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                prompt = input("> ")
            except EOFError:
                break
            if prompt.strip().lower() in {"exit", "quit"}:
                break
            if not prompt.strip():
                continue
            text = decode_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                eos_token=args.eos_token,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )
            print(text)
    else:
        text = decode_text(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            eos_token=args.eos_token,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

        print("=== Prompt ===")
        print(args.prompt)
        print("=== Completion ===")
        print(text)


if __name__ == "__main__":
    main()

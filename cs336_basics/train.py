from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import importlib

import numpy as np
import torch

# python3 cs336_basics/train.py \
#   --train-text data/TinyStoriesV2-GPT4-train.txt \
#   --val-text data/TinyStoriesV2-GPT4-valid.txt \
#   --vocab-size 256 \
#   --batch-size 32 \
#   --context-length 256 \
#   --total-iters 2000 \
#   --eval-every 200 \
#   --save-every 200 \
#   --checkpoint-path checkpoints/tinystories.pt

# python3 cs336_basics/train.py \
#   --train-data /path/to/train.bin \
#   --val-data /path/to/val.bin \
#   --data-dtype uint16 \
#   --vocab-size 50257 \
#   --total-iters 5000

# .venv/bin/python cs336_basics/train.py \
#   --tokenizer bpe \
#   --train-text data/TinyStoriesV2-GPT4-train.txt \
#   --val-text data/TinyStoriesV2-GPT4-valid.txt \
#   --vocab-path tests/fixtures/gpt2_vocab.json \
#   --merges-path tests/fixtures/gpt2_merges.txt \
#   --vocab-size 50257



# Toggle switches:
# - True: prefer your custom implementation; fallback to torch if import fails.
# - False: force torch implementation.
USE_CUSTOM_CROSS_ENTROPY = True
USE_CUSTOM_CLIP_GRADIENT = True


try:
    from cs336_basics.adamw import AdamWOptimizer
    from cs336_basics.full_llm import SimpleTransformer
    from cs336_basics.learn_rate_scheduling import schedule_learning_rate
    from cs336_basics.tokenizer import Tokenizer
    from cs336_basics.save_and_load import load as load_checkpoint
    from cs336_basics.save_and_load import save as save_checkpoint
except Exception:  # pragma: no cover
    # Allows running as: python cs336_basics/train.py (without package installation).
    from adamw import AdamWOptimizer
    from full_llm import SimpleTransformer
    from learn_rate_scheduling import schedule_learning_rate
    from tokenizer import Tokenizer
    from save_and_load import load as load_checkpoint
    from save_and_load import save as save_checkpoint


def _import_custom_attr(module_names: list[str], attr_name: str):
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name, None)
        except Exception:
            continue
    return None


_custom_cross_entropy = None
if USE_CUSTOM_CROSS_ENTROPY:
    _custom_cross_entropy = _import_custom_attr(
        ["cs336_basics.cross_entropy_loss", "cross_entropy_loss"],
        "cross_entropy_loss",
    )

_custom_clip_gradient = None
if USE_CUSTOM_CLIP_GRADIENT:
    _custom_clip_gradient = _import_custom_attr(
        ["cs336_basics.clip_gradient", "clip_gradient"],
        "clip_gradient",
    )


def _cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    if _custom_cross_entropy is not None:
        return _custom_cross_entropy(inputs, targets)
    return torch.nn.functional.cross_entropy(inputs, targets)


def _clip_gradients(parameters, max_l2_norm: float) -> None:
    if _custom_clip_gradient is not None:
        _custom_clip_gradient(parameters, max_l2_norm)
    else:
        torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SimpleTransformer on tokenized data with np.memmap")

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--train-data", type=Path, help="Path to training token data (.bin raw array or .npy).")
    data_group.add_argument("--val-data", type=Path, help="Path to validation token data (.bin raw array or .npy).")
    data_group.add_argument("--train-text", type=Path, help="Optional raw text file. Will be converted to memmap bytes.")
    data_group.add_argument("--val-text", type=Path, help="Optional raw text file. Will be converted to memmap bytes.")
    data_group.add_argument("--tokenizer", choices=["byte", "bpe"], default="byte")
    data_group.add_argument("--vocab-path", type=Path, default=None, help="Required for bpe text mode.")
    data_group.add_argument("--merges-path", type=Path, default=None, help="Required for bpe text mode.")
    data_group.add_argument("--special-tokens", nargs="*", default=["<|endoftext|>"])
    data_group.add_argument(
        "--data-dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "uint32", "int32", "int64"],
        help="Dtype for raw binary token files.",
    )
    data_group.add_argument(
        "--generated-data-dir",
        type=Path,
        default=Path("data/tokenized"),
        help="Where to write memmap files generated from --train-text/--val-text.",
    )

    model_group = parser.add_argument_group("model")
    model_group.add_argument("--vocab-size", type=int, default=256)
    model_group.add_argument("--d-model", type=int, default=256)
    model_group.add_argument("--num-heads", type=int, default=8)
    model_group.add_argument("--d-ff", type=int, default=1024)
    model_group.add_argument("--num-layers", type=int, default=6)
    model_group.add_argument("--theta", type=float, default=10000.0)
    model_group.add_argument("--max-seq-len", type=int, default=1024)
    model_group.add_argument("--context-length", type=int, default=256)

    optim_group = parser.add_argument_group("optimizer")
    optim_group.add_argument("--learning-rate", type=float, default=3e-4)
    optim_group.add_argument("--min-learning-rate", type=float, default=3e-5)
    optim_group.add_argument("--warmup-iters", type=int, default=200)
    optim_group.add_argument("--cosine-cycle-iters", type=int, default=2000)
    optim_group.add_argument("--weight-decay", type=float, default=0.1)
    optim_group.add_argument("--beta1", type=float, default=0.9)
    optim_group.add_argument("--beta2", type=float, default=0.95)
    optim_group.add_argument("--eps", type=float, default=1e-8)
    optim_group.add_argument("--grad-clip", type=float, default=1.0)

    train_group = parser.add_argument_group("training")
    train_group.add_argument("--batch-size", type=int, default=32)
    train_group.add_argument("--total-iters", type=int, default=5000)
    train_group.add_argument("--seed", type=int, default=1337)
    train_group.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    io_group = parser.add_argument_group("io/logging")
    io_group.add_argument("--log-every", type=int, default=20)
    io_group.add_argument("--eval-every", type=int, default=200)
    io_group.add_argument("--eval-iters", type=int, default=20)
    io_group.add_argument("--save-every", type=int, default=500)
    io_group.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/latest.pt"))
    io_group.add_argument("--resume", action="store_true")
    io_group.add_argument("--config-out", type=Path, default=Path("checkpoints/train_config.json"))

    wandb_group = parser.add_argument_group("wandb")
    wandb_group.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging.")
    wandb_group.add_argument("--wandb-project", type=str, default="cs336-basics")
    wandb_group.add_argument("--wandb-run-name", type=str, default=None)

    args = parser.parse_args()

    using_token_data = args.train_data is not None and args.val_data is not None
    using_text_data = args.train_text is not None and args.val_text is not None
    if not (using_token_data or using_text_data):
        parser.error("Provide either (--train-data and --val-data) or (--train-text and --val-text).")
    if args.tokenizer == "bpe" and using_text_data and (args.vocab_path is None or args.merges_path is None):
        parser.error("When --tokenizer bpe with text input, both --vocab-path and --merges-path are required.")

    return args


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


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


def convert_text_to_memmap_bytes(text_path: Path, out_path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> Path:
    ensure_parent(out_path)
    total_size = text_path.stat().st_size

    mem = np.memmap(out_path, dtype=np.uint8, mode="w+", shape=(total_size,))
    pos = 0
    with text_path.open("rb") as f:
        while True:
            block = f.read(chunk_bytes)
            if not block:
                break
            block_arr = np.frombuffer(block, dtype=np.uint8)
            mem[pos : pos + block_arr.size] = block_arr
            pos += block_arr.size

    mem.flush()
    del mem
    return out_path


def convert_text_to_token_bin(
    text_path: Path,
    out_path: Path,
    tokenizer: Tokenizer,
    dtype: str = "uint32",
    flush_tokens: int = 1_000_000,
) -> Path:
    ensure_parent(out_path)
    np_dtype = np.dtype(dtype)

    with text_path.open("r", encoding="utf-8", errors="ignore") as src, out_path.open("wb") as dst:
        buffer: list[int] = []
        for token_id in tokenizer.encode_iterable(src):
            buffer.append(int(token_id))
            if len(buffer) >= flush_tokens:
                np.asarray(buffer, dtype=np_dtype).tofile(dst)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=np_dtype).tofile(dst)
    return out_path


def open_token_array(path: Path, dtype: str) -> np.ndarray:
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, dtype=np.dtype(dtype), mode="r")

    if data.ndim != 1:
        raise ValueError(f"Expected 1D token array, got shape={data.shape} from {path}")
    if data.size < 2:
        raise ValueError(f"Dataset too small: {path}")
    return data


def sample_batch(data: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = int(data.shape[0]) - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Dataset length {data.shape[0]} is too short for context_length={context_length}. Need at least {context_length + 1}."
        )

    starts = np.random.randint(0, max_start + 1, size=(batch_size,), dtype=np.int64)
    offsets = np.arange(context_length + 1, dtype=np.int64)
    window_idx = starts[:, None] + offsets[None, :]

    windows = np.asarray(data[window_idx], dtype=np.int64)
    x = torch.from_numpy(windows[:, :-1]).to(device=device, dtype=torch.long, non_blocking=True)
    y = torch.from_numpy(windows[:, 1:]).to(device=device, dtype=torch.long, non_blocking=True)
    return x, y


def eval_loss(
    model: torch.nn.Module,
    data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_iters: int,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(eval_iters):
            x, y = sample_batch(data, batch_size, context_length, device)
            logits = model(x)
            loss = _cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def make_datasets(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, str, str, dict[str, Any]]:
    if args.train_data is not None and args.val_data is not None:
        train = open_token_array(args.train_data, args.data_dtype)
        val = open_token_array(args.val_data, args.data_dtype)
        dataset_desc = f"{args.tokenizer}-memmap train={args.train_data} val={args.val_data} dtype={args.data_dtype}"
        tokenizer_meta = {
            "tokenizer_mode": args.tokenizer,
            "tokenizer_vocab_path": str(args.vocab_path) if args.vocab_path else None,
            "tokenizer_merges_path": str(args.merges_path) if args.merges_path else None,
            "tokenizer_special_tokens": args.special_tokens,
        }
        return train, val, dataset_desc, args.data_dtype, tokenizer_meta

    if args.tokenizer == "byte":
        train_bin = args.generated_data_dir / f"{args.train_text.stem}.byte.uint8.bin"
        val_bin = args.generated_data_dir / f"{args.val_text.stem}.byte.uint8.bin"

        if not train_bin.exists():
            print(f"[data] converting (byte) {args.train_text} -> {train_bin}")
            convert_text_to_memmap_bytes(args.train_text, train_bin)
        if not val_bin.exists():
            print(f"[data] converting (byte) {args.val_text} -> {val_bin}")
            convert_text_to_memmap_bytes(args.val_text, val_bin)

        token_dtype = "uint8"
        train = open_token_array(train_bin, token_dtype)
        val = open_token_array(val_bin, token_dtype)
        dataset_desc = f"byte-memmap train={train_bin} val={val_bin}"
        tokenizer_meta = {
            "tokenizer_mode": "byte",
            "tokenizer_vocab_path": None,
            "tokenizer_merges_path": None,
            "tokenizer_special_tokens": args.special_tokens,
        }
        return train, val, dataset_desc, token_dtype, tokenizer_meta

    # BPE tokenizer mode.
    tokenizer = load_bpe_tokenizer(args.vocab_path, args.merges_path, args.special_tokens)
    token_dtype = "uint32"
    train_bin = args.generated_data_dir / f"{args.train_text.stem}.bpe.{token_dtype}.bin"
    val_bin = args.generated_data_dir / f"{args.val_text.stem}.bpe.{token_dtype}.bin"

    if not train_bin.exists():
        print(f"[data] converting (bpe) {args.train_text} -> {train_bin}")
        convert_text_to_token_bin(args.train_text, train_bin, tokenizer=tokenizer, dtype=token_dtype)
    if not val_bin.exists():
        print(f"[data] converting (bpe) {args.val_text} -> {val_bin}")
        convert_text_to_token_bin(args.val_text, val_bin, tokenizer=tokenizer, dtype=token_dtype)

    train = open_token_array(train_bin, token_dtype)
    val = open_token_array(val_bin, token_dtype)
    dataset_desc = f"bpe-memmap train={train_bin} val={val_bin}"
    tokenizer_meta = {
        "tokenizer_mode": "bpe",
        "tokenizer_vocab_path": str(args.vocab_path),
        "tokenizer_merges_path": str(args.merges_path),
        "tokenizer_special_tokens": args.special_tokens,
    }
    return train, val, dataset_desc, token_dtype, tokenizer_meta


def maybe_init_wandb(args: argparse.Namespace, config: dict[str, Any]):
    if not args.wandb:
        return None

    try:
        import wandb
    except Exception as exc:  # pragma: no cover
        print(f"[wandb] disabled: import failed ({exc})")
        return None

    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=config)
    return wandb


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = lr


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, val_data, dataset_desc, token_dtype, tokenizer_meta = make_datasets(args)

    if args.vocab_size <= int(np.max(train_data)):
        raise ValueError(
            f"vocab_size={args.vocab_size} must be greater than max token id in train data ({int(np.max(train_data))})."
        )

    model = SimpleTransformer(
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
    ).to(args.device)

    optimizer = AdamWOptimizer(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume and args.checkpoint_path.exists():
        print(f"[ckpt] loading from {args.checkpoint_path}")
        start_iter = load_checkpoint(args.checkpoint_path, model=model, optimizer=optimizer)
        print(f"[ckpt] resumed at iteration {start_iter}")

    config = {
        "dataset": dataset_desc,
        "token_dtype": token_dtype,
        "vocab_size": args.vocab_size,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "num_layers": args.num_layers,
        "theta": args.theta,
        "max_seq_len": args.max_seq_len,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "total_iters": args.total_iters,
        "learning_rate": args.learning_rate,
        "min_learning_rate": args.min_learning_rate,
        "warmup_iters": args.warmup_iters,
        "cosine_cycle_iters": args.cosine_cycle_iters,
        "weight_decay": args.weight_decay,
        "betas": [args.beta1, args.beta2],
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "device": args.device,
        "seed": args.seed,
        "checkpoint_path": str(args.checkpoint_path),
        "use_custom_cross_entropy_requested": USE_CUSTOM_CROSS_ENTROPY,
        "use_custom_cross_entropy_active": _custom_cross_entropy is not None,
        "use_custom_clip_gradient_requested": USE_CUSTOM_CLIP_GRADIENT,
        "use_custom_clip_gradient_active": _custom_clip_gradient is not None,
        **tokenizer_meta,
    }

    ensure_parent(args.config_out)
    with args.config_out.open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    wb = maybe_init_wandb(args, config)

    print(f"[init] dataset={dataset_desc}")
    print(f"[init] train_tokens={train_data.shape[0]} val_tokens={val_data.shape[0]} device={args.device}")
    print(f"[init] tokenizer_mode={tokenizer_meta['tokenizer_mode']} token_dtype={token_dtype}")
    print(
        "[init] impls "
        f"cross_entropy={'custom' if _custom_cross_entropy is not None else 'torch'} "
        f"clip_gradient={'custom' if _custom_clip_gradient is not None else 'torch'}"
    )

    model.train()
    for it in range(start_iter, args.total_iters):
        lr = schedule_learning_rate(
            it,
            args.learning_rate,
            args.min_learning_rate,
            args.warmup_iters,
            args.cosine_cycle_iters,
        )
        set_learning_rate(optimizer, lr)

        x, y = sample_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()

        logits = model(x)
        loss = _cross_entropy_loss(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        loss.backward()
        if args.grad_clip > 0:
            _clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        step = it + 1
        if step % args.log_every == 0 or step == 1:
            msg = f"[train] iter={step}/{args.total_iters} loss={loss.item():.4f} lr={lr:.3e}"
            print(msg)
            if wb is not None:
                wb.log({"iter": step, "train/loss": float(loss.item()), "lr": lr}, step=step)

        if step % args.eval_every == 0 or step == args.total_iters:
            train_eval = eval_loss(model, train_data, args.batch_size, args.context_length, args.device, args.eval_iters)
            val_eval = eval_loss(model, val_data, args.batch_size, args.context_length, args.device, args.eval_iters)
            print(f"[eval] iter={step} train_loss={train_eval:.4f} val_loss={val_eval:.4f}")
            if wb is not None:
                wb.log({"iter": step, "eval/train_loss": train_eval, "eval/val_loss": val_eval}, step=step)

        if step % args.save_every == 0 or step == args.total_iters:
            ensure_parent(args.checkpoint_path)
            save_checkpoint(model=model, optimizer=optimizer, iteration=step, out=args.checkpoint_path)
            print(f"[ckpt] saved iter={step} -> {args.checkpoint_path}")

    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()

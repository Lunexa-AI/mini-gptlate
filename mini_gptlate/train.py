"""Training script for Mini‑GPTLate.

Add‑ons over the original 60‑line loop:
* **Hugging Face streaming** – pass `--hf wikitext` (or any dataset id) and an
  optional `--split` to pull data via the `datasets` library.
* **Torch 2 compiler** – `--compile` wraps the model in `torch.compile(...)` for
  ~20‑30 % CPU speed‑up when PyTorch ≥ 2.1.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from rich.progress import Progress, BarColumn, TimeElapsedColumn

from .model import GPTLate, GPTConfig
from .tokeniser import get_tokeniser

try:
    from datasets import load_dataset  # type: ignore
except ImportError:
    load_dataset = None  # optional; guarded

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def _seq_chunks(texts: Sequence[str], ctx: int):
    tok = get_tokeniser()
    ids: List[int] = []
    for txt in texts:
        ids.extend(tok.encode(txt))
    step = ctx + 1  # +1 so last token is label
    for i in range(0, len(ids) - step, step):
        yield ids[i : i + step]


class LocalText(Dataset):
    def __init__(self, patterns: List[str], ctx: int):
        files = [f for p in patterns for f in glob.glob(p)]
        self.samples = list(_seq_chunks((Path(f).read_text() for f in files), ctx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


class HFText(Dataset):
    def __init__(self, name: str, split: str, ctx: int):
        if load_dataset is None:
            raise ImportError("pip install datasets to use --hf flag")
        ds = load_dataset(name, split=split, streaming=True)
        self.samples = list(_seq_chunks((r["text"] for r in ds), ctx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq = torch.tensor(self.samples[idx], dtype=torch.long)
        return seq[:-1], seq[1:]


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------


def train(args):
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    cfg = GPTConfig(context=args.context)
    model = GPTLate(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    if args.resume and args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location=device))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    dataset: Dataset
    if args.hf:
        dataset = HFText(args.hf, args.split, args.context)
    else:
        dataset = LocalText(args.data, args.context)

    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    prog = Progress("{task.description}", BarColumn(), TimeElapsedColumn())
    task = prog.add_task("train", total=args.epochs * len(loader))

    step = 0
    model.train()
    with prog:
        for _ in range(args.epochs):
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss.backward()
                if (step + 1) % args.grad_acc == 0:
                    opt.step(); opt.zero_grad(set_to_none=True)
                if step % args.log_every == 0:
                    prog.console.print(f"step {step}  loss {loss.item():.3f}")
                if step % args.save_every == 0 and step > 0:
                    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), out / f"step_{step}.pt")
                prog.update(task, advance=1)
                step += 1

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "last.pt")


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="Train Mini‑GPTLate")
    p.add_argument("--data", nargs="+", help="Glob(s) of local .txt files")
    p.add_argument("--hf", type=str, help="HF dataset id (e.g. wikitext)")
    p.add_argument("--split", type=str, default="train")

    p.add_argument("--context", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--grad_acc", type=int, default=1)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=500)

    p.add_argument("--out", type=str, default="runs")
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument("--resume", action="store_true")

    p.add_argument("--cuda", action="store_true")
    p.add_argument("--compile", action="store_true", help="Enable torch.compile")

    args = p.parse_args(argv)
    if not args.data and not args.hf:
        p.error("Provide either --data or --hf dataset")
    train(args)


if __name__ == "__main__":
    main()

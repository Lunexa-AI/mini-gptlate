"""LoRA fine‑tuning script for Mini‑GPTLate (≈ 70 LoC).

Requires `peft` (pip install peft) which supports CPU LoRA starting PEFT v0.10.
By default it injects rank‑4 adapters into Q and V projection matrices.

Usage:
    python -m mini_gptlate.lora_finetune \
        --data data/*.txt \
        --ckpt gpt2.pt \
        --out lora_adapters.pt

The resulting adapter file is tiny (≈ <5 MB) and can be merged or loaded via
`GPTLate.load_state_dict(base_sd, strict=False); model.load_state_dict(lora_sd, strict=False)`.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from rich.progress import Progress, BarColumn

try:
    from peft import get_peft_model, LoraConfig, LoraModel  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError("`peft` is required: pip install peft") from e

from .model import GPTLate, GPTConfig
from .tokeniser import get_tokeniser
from .train import TextDataset  # reuse dataset class


# ---------------------------------------------------------------------------
# Fine‑tune
# ---------------------------------------------------------------------------

def finetune(args):
    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    cfg = GPTConfig(context=args.context)
    base = GPTLate(cfg).to(device)
    if args.ckpt and Path(args.ckpt).exists():
        base.load_state_dict(torch.load(args.ckpt, map_location=device), strict=False)
    base.eval()

    peft_cfg = LoraConfig(r=args.rank, lora_alpha=args.rank * 2, target_modules=["attn"], lora_dropout=0.05)
    model: LoraModel = get_peft_model(base, peft_cfg)  # type: ignore

    files = [f for pattern in args.data for f in glob.glob(pattern)]
    loader = DataLoader(TextDataset(files, args.context), batch_size=args.batch, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    progress = Progress("{task.description}", BarColumn())
    task = progress.add_task("LoRA", total=args.steps)
    with progress:
        step = 0
        while step < args.steps:
            for x, y in loader:
                if step >= args.steps:
                    break
                x, y = x.to(device), y.to(device)
                logits, _ = model(x)
                loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
                loss.backward()
                opt.step(); opt.zero_grad(set_to_none=True)
                if step % args.log_every == 0:
                    progress.console.print(f"step {step}  loss {loss.item():.3f}")
                progress.update(task, advance=1)
                step += 1

    # save only LoRA adapters
    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), args.out)
    print("Adapters saved to", args.out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None):
    p = argparse.ArgumentParser(description="LoRA fine‑tune Mini‑GPTLate")
    p.add_argument("--data", nargs="+", required=True, help="Glob(s) of .txt files")
    p.add_argument("--ckpt", type=str, default="", help="Base checkpoint to load")
    p.add_argument("--context", type=int, default=256)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--rank", type=int, default=4, help="LoRA rank")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--out", type=str, default="lora_adapters.pt")
    p.add_argument("--cuda", action="store_true")
    args = p.parse_args(argv)

    finetune(args)


if __name__ == "__main__":
    main()

"""Command-line interface for Mini-GPTLate.

Example:
    $ python -m mini_gptlate.cli --prompt "Hello world" --trace
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from rich.console import Console

from .model import GPTLate, GPTConfig
from .tokeniser import get_tokeniser
from .tracer import AttentionTracer

console = Console()


def build_model(args) -> GPTLate:
    cfg = GPTConfig(
        context=args.context,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        rope=args.rope,
        alibi=args.alibi,
    )
    model = GPTLate(cfg)
    if args.ckpt and Path(args.ckpt).exists():
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        console.print(f"Loaded weights from [green]{args.ckpt}[/]")
    model.eval()
    return model


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Mini-GPTLate text generator")
    parser.add_argument("--prompt", type=str, default="Hello",
                        help="Initial text prompt")
    parser.add_argument("--max_new", type=int, default=50,
                        help="Tokens to generate")
    parser.add_argument("--context", type=int, default=256,
                        help="Context length")
    parser.add_argument("--n_layer", type=int, default=6)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--rope", action="store_true")
    parser.add_argument("--alibi", action="store_true")
    parser.add_argument("--ckpt", type=str, default="",
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--trace", action="store_true",
                        help="Show live attention heat-maps")
    args = parser.parse_args(argv)

    tok = get_tokeniser()
    prompt_ids = tok.encode(args.prompt)

    model = build_model(args)

    ids = prompt_ids.copy()
    tracer = AttentionTracer(ids, tok.decode, pause=False) if args.trace else None

    console.print(f"\n[bold]Prompt:[/] {args.prompt}\n")
    with torch.no_grad():
        for step in range(args.max_new):
            inp = torch.tensor(ids[-args.context :], dtype=torch.long)[None]
            logits, attns = model(inp, return_attn=args.trace)
            next_id = int(logits[0, -1].argmax())
            ids.append(next_id)
            word = tok.decode([next_id])
            console.print(word, end="", style="bright_green")

            if args.trace and attns is not None:
                tracer.log(step % model.config.n_layer, attns[-1])

            sys.stdout.flush()
    console.print()  # newline


if __name__ == "__main__":  # pragma: no cover
    main()


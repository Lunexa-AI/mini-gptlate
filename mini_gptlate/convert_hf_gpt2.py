"""Convert a Hugging‑Face GPT‑2 checkpoint into a Mini‑GPTLate `state_dict.pt`.

We assume you want *GPT‑2 Small* (12 layers, 768‑dim).  If you trained
Mini‑GPTLate with matching config (n_layer=12, d_model=768, n_head=12, context=1024)
then you can reuse the weights without change.

For classroom demos you can also slice the first 6 layers by passing
`--layers 6`, but *embedding and head dimensions must still match*.

Usage:
    python -m mini_gptlate.convert_hf_gpt2 --out gpt2.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

try:
    from transformers import GPT2LMHeadModel  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "`transformers` must be installed: pip install transformers==4.*"
    ) from e


def load_hf(variant: str) -> Dict[str, torch.Tensor]:
    model = GPT2LMHeadModel.from_pretrained(variant)
    return model.state_dict()


def slice_layers(sd: Dict[str, torch.Tensor], keep: int):
    """Keep only the first `keep` transformer blocks (for 200‑line config)."""
    out = {}
    for k, v in sd.items():
        if k.startswith("transformer.h."):
            idx = int(k.split(".")[2])
            if idx >= keep:
                continue
            new_k = k.replace(f"h.{idx}", f"blocks.{idx}")
        else:
            new_k = k.replace("transformer.", "")
        out[new_k] = v
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Convert HF GPT‑2 to Mini‑GPTLate")
    p.add_argument("--variant", default="gpt2", help="HF model id (e.g. gpt2)")
    p.add_argument("--layers", type=int, default=12, help="How many layers to keep")
    p.add_argument("--out", default="gpt2.pt", help="Output .pt path")
    args = p.parse_args(argv)

    sd = load_hf(args.variant)
    if args.layers < 12:
        sd = slice_layers(sd, args.layers)
    torch.save(sd, args.out)
    print("Saved", args.out)


if __name__ == "__main__":  # pragma: no cover
    main()

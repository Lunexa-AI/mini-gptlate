"""Build a 20k‑vocab Byte‑Pair tokenizer from any UTF‑8 corpus.

Why? Schools in the Global South often have textbooks or Wikipedia dumps in
local languages but no ready‑made tokeniser.  This script turns *any* text file
(or folder / glob) into a **tokenizer.json** compatible with Mini‑GPTLate.

### Usage (one‑liner)
```bash
python tools/build_tokenizers.py --text data/*.txt --out swahili_tokenizer.json
```

* Requires `pip install tokenizers` (small, wheels available for ARM & x86). *
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

try:
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
except ImportError as e:  # pragma: no cover
    sys.exit(
        "[!] `tokenizers` library missing. Install with:  pip install tokenizers\n"
        f"Original error: {e}"
    )

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def gather_text(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for sub in p.rglob("*.txt"):
                yield sub.read_text(errors="ignore")
        else:
            for g in glob.glob(str(p)):
                yield Path(g).read_text(errors="ignore")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(paths, out, vocab_size):
    print(f"[+] Collecting texts from {len(paths)} path(s)…")
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=True)
    tokenizer.train_from_iterator(gather_text(paths), trainer=trainer)

    out = Path(out)
    out.write_text(tokenizer.to_str())
    print(f"[✓] Saved tokenizer to {out}  (size: {out.stat().st_size/1024:.1f} KB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description="Build a BPE tokenizer for Mini‑GPTLate")
    p.add_argument("--text", nargs="+", required=True, help=".txt files, folders, or globs")
    p.add_argument("--out", default="tokenizer.json", help="Output path (default: tokenizer.json)")
    p.add_argument("--vocab", type=int, default=20_000, help="Vocabulary size (default: 20k)")
    args = p.parse_args(argv)

    build(args.text, args.out, args.vocab)


if __name__ == "__main__":
    main()

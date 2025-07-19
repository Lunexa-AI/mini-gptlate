"""Offline BPE tokenizer for Mini-GPTLate.

This is *not* a full-featured Hugging-Face tokenizer; it's about 40 lines yet
handles exactly what we need: basic English plus Unicode fallback.

The tokenizer expects a `tokenizer.json` file (dict with `merges` & `vocab` keys)
next to this script.  A ready-made file can be downloaded or trained separately.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

# Regex to split on whitespace + keep punctuation (roughly GPT-2's)
PATTERN = re.compile(r"[\w']+|[^\w\s]", flags=re.UNICODE)
_CUR_DIR = Path(__file__).resolve().parent


class OfflineBPETokeniser:
    def __init__(self, path: str | Path | None = None):
        path = Path(path or _CUR_DIR / "tokenizer.json")
        if not path.exists():
            raise FileNotFoundError(
                f"tokenizer.json not found at {path}. Provide or train one."
            )
        data = json.loads(path.read_text())
        self.vocab: Dict[str, int] = data["vocab"]
        self.merges: Dict[Tuple[str, str], int] = {
            tuple(k.split()): rank for rank, k in enumerate(data["merges"])
        }
        self.unk = self.vocab.get("<unk>", len(self.vocab))

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def bpe(self, token: str) -> List[int]:
        """Apply merges until no pair is in the table; return list of ids."""
        if token in self.vocab:
            return [self.vocab[token]]
        # Start as a list of chars
        parts = list(token)
        while True:
            pairs = [(parts[i], parts[i + 1]) for i in range(len(parts) - 1)]
            ranked = [(self.merges.get(p, 1e9), i, p) for i, p in enumerate(pairs)]
            best_rank, idx, pair = min(ranked, key=lambda x: x[0])
            if best_rank == 1e9:
                break
            parts[idx : idx + 2] = ["".join(pair)]
        return [self.vocab.get(p, self.unk) for p in parts]

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for tok in PATTERN.findall(text):
            ids.extend(self.bpe(tok.lower()))
        return ids

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, ids: List[int]) -> str:
        inv_vocab = {i: s for s, i in self.vocab.items()}
        tokens = [inv_vocab.get(i, "<unk>") for i in ids]
        return "".join(tokens).replace("▁", " ").strip()


# ----------------------------------------------------------------------------
# Convenience singleton for CLI scripts
# ----------------------------------------------------------------------------

def get_tokeniser():
    global _tok_singleton  # noqa: PLW0603
    try:
        return _tok_singleton
    except NameError:
        _tok_singleton = OfflineBPETokeniser()
        return _tok_singleton

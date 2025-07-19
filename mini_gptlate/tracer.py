"""Live attention tracer for Mini‑GPTLate.

Goal: stay tiny, no external GUI – just ANSI art so it works over SSH or in
classroom terminals.  Each call to ``log()`` prints one heat‑map per layer.

Blocks are Unicode shades:  ▁▂▃▄▅▆▇█ (8‑level).  Averaged across heads.
"""
from __future__ import annotations

import math
from typing import List

import torch
from rich.console import Console
from rich.table import Table

BLOCKS = " ▁▂▃▄▅▆▇█"  # index 0‑7
console = Console()


class AttentionTracer:
    """Pretty‑print attention weights layer‑by‑layer."""

    def __init__(self, tokens: List[int], decode, pause: bool = False):
        """
        Args:
            tokens : the prompt token IDs so we can label rows/cols.
            decode : callable id‑>string.
            pause  : if True wait for Enter between layers.
        """
        self.words = [decode([t]) or "·" for t in tokens]
        self.pause = pause

    # ------------------------------------------------------------------
    # Helper – weight → shaded block
    # ------------------------------------------------------------------

    @staticmethod
    def _shade(val: float) -> str:
        idx = min(int(val * 7 + 0.5), 7)
        return BLOCKS[idx]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, layer: int, weights: torch.Tensor):
        """Print heat‑map for a single layer.

        Args:
            layer    : layer index (0‑based)
            weights  : [B, head, T, T] tensor (we assume B==1)
        """
        w = weights.mean(1)[0].cpu()  # [T, T] average heads
        T = w.size(0)

        table = Table(title=f"Layer {layer}", box=None, pad_edge=False)
        table.add_column("tok→tok", no_wrap=True)
        for col in range(T):
            table.add_column(self.words[col][:4])

        for r in range(T):
            row = [self.words[r][:4]]
            for c in range(T):
                row.append(self._shade(w[r, c].item()))
            table.add_row(*row)

        console.print(table)
        if self.pause:
            console.print("[bold yellow]Press Enter to continue…[/]")
            try:
                input()
            except EOFError:
                pass

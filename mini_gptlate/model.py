"""Mini-GPTLate – ultra-light GPT-2-class model (<200 LoC).

Core ideas:
*   Keep everything readable & hackable – no obscure helper layers.
*   Return attention weights when `return_attn=True`, enabling the live tracer.
*   Flags for RoPE / ALiBi handled via small helper; off by default for clarity.

Usage (in repo root):
    >>> import torch, mini_gptlate as mg
    >>> model = mg.GPTLate()
    >>> ids = torch.randint(0, model.config.vocab_size, (1, 10))
    >>> logits, attns = model(ids, return_attn=True)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Config – simple namespace so users can pass overrides easily
# -----------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 50_000
    context: int = 256
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 6
    rope: bool = False  # rotary positions flag
    alibi: bool = False  # alibi flag (mutually exclusive with rope)

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_head


# -----------------------------------------------------------------------------
# Positional helpers
# -----------------------------------------------------------------------------

def rotary_embed(dim: int, seq_len: int, device: torch.device):
    """Returns cos & sin tables for RoPE; see (Su et al. 2021)."""
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)  # [seq, dim]
    return emb.cos(), emb.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate last-dim pairs (x1, x2) by cos/sin tables."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return x_rot.flatten(-2)


# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------


class Block(nn.Module):
    """GPT-2 style block with pre-LayerNorm and casual self-attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.ln1(x)
        if cos_sin is not None:
            cos, sin = cos_sin
            qkv = apply_rope(qkv, cos, sin)
        a, w = self.attn(qkv, qkv, qkv, attn_mask=mask, need_weights=True)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, w  # return attention for tracer


# -----------------------------------------------------------------------------
# GPTLate model wrapper
# -----------------------------------------------------------------------------


class GPTLate(nn.Module):
    def __init__(self, config: GPTConfig | None = None):
        super().__init__()
        self.config = config or GPTConfig()
        cfg = self.config

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if not (cfg.rope or cfg.alibi):
            self.pos = nn.Parameter(torch.zeros(1, cfg.context, cfg.d_model))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # weight-tying
        self.head.weight = self.embed.weight

        self._mask = torch.triu(torch.ones(cfg.context, cfg.context), 1).bool()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        tokens: torch.Tensor,  # [B, T]
        *,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T = tokens.shape
        if T > self.config.context:
            raise ValueError("Sequence length exceeds model context")
        device = tokens.device

        x = self.embed(tokens)

        # Add positions unless we’re using RoPE / ALiBi which bake them into attention.
        if not (self.config.rope or self.config.alibi):
            x = x + self.pos[:, :T, :]

        attns: List[torch.Tensor] = []

        mask = self._mask[:T, :T].to(device)

        cos_sin = None
        if self.config.rope:
            cos, sin = rotary_embed(self.config.head_dim, T, device)
            cos_sin = (cos, sin)

        for blk in self.blocks:
            x, w = blk(x, mask, cos_sin)
            if return_attn:
                attns.append(w.detach())

        x = self.ln_f(x)
        logits = self.head(x)
        return logits, attns if return_attn else None

    # ------------------------------------------------------------------
    # Convenience: generate next token greedily (no KV cache for brevity)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 50):
        ids = torch.tensor(prompt_ids, dtype=torch.long)[None]
        for _ in range(max_new_tokens):
            logits, _ = self(ids)
            next_id = logits[0, -1].argmax(-1, keepdim=True)
            ids = torch.cat((ids, next_id[None]), dim=1)
            if ids.shape[1] >= self.config.context:
                ids = ids[:, -self.config.context :]
        return ids.squeeze().tolist()


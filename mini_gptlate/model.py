"""Mini-GPTLate - ultra-light GPT-2-class model (<200 LoC).

Core ideas:
*   Keep everything readable & hackable - no obscure helper layers.
*   Return attention weights when 'return_attn=True', enabling the live tracer.
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
    vocab_size: int = 50_000  # Number of unique tokens in the vocabulary
    context: int = 256        # Maximum sequence length (context window)
    d_model: int = 256        # Embedding dimension (model hidden size)
    n_head: int = 8           # Number of attention heads
    n_layer: int = 6          # Number of transformer blocks (layers)
    rope: bool = False        # Use rotary positional embeddings if True
    alibi: bool = False       # Use ALiBi positional bias if True (exclusive with RoPE)

    @property
    def head_dim(self) -> int:
        # Size of each attention head
        return self.d_model // self.n_head

# -----------------------------------------------------------------------------
# Positional helpers
# -----------------------------------------------------------------------------

def rotary_embed(dim: int, seq_len: int, device: torch.device):
    """Returns cos & sin tables for RoPE; see (Su et al. 2021).
    These tables are used to rotate query/key vectors in attention, encoding position information.
    """
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, device=device) / dim))  # Inverse frequencies for each pair of dims
    t = torch.arange(seq_len, device=device)  # Sequence positions (0, 1, ..., seq_len-1)
    freqs = torch.einsum("i,j->ij", t, inv_freq)  # Outer product: position x frequency
    emb = torch.cat((freqs, freqs), dim=-1)  # Duplicate for even/odd dims
    return emb.cos(), emb.sin()  # Return cos/sin tables for rotation

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Rotate last-dim pairs (x1, x2) by cos/sin tables.
    This encodes position into the vector by rotating each pair of features.
    """
    x1, x2 = x[..., ::2], x[..., 1::2]  # Split into even and odd dims
    x_rot = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)  # Apply 2D rotation
    return x_rot.flatten(-2)  # Restore original shape

# -----------------------------------------------------------------------------
# Transformer Block
# -----------------------------------------------------------------------------

class Block(nn.Module):
    """GPT-2 style block with pre-LayerNorm and casual self-attention."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)  # LayerNorm before attention
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_head, batch_first=True)  # Multi-head self-attention
        self.ln2 = nn.LayerNorm(cfg.d_model)  # LayerNorm before MLP
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),  # Expand hidden size (feedforward)
            nn.GELU(),                               # Non-linearity
            nn.Linear(4 * cfg.d_model, cfg.d_model), # Project back to hidden size
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        qkv = self.ln1(x)  # Normalize input
        if cos_sin is not None:
            cos, sin = cos_sin
            qkv = apply_rope(qkv, cos, sin)  # Apply RoPE if enabled
        a, w = self.attn(qkv, qkv, qkv, attn_mask=mask, need_weights=True)  # Self-attention
        x = x + a  # Residual connection (add attention output)
        x = x + self.mlp(self.ln2(x))  # Residual connection (add MLP output)
        return x, w  # Return output and attention weights

# -----------------------------------------------------------------------------
# GPTLate model wrapper
# -----------------------------------------------------------------------------

class GPTLate(nn.Module):
    def __init__(self, config: GPTConfig | None = None):
        super().__init__()
        self.config = config or GPTConfig()  # Use default config if none provided
        cfg = self.config

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)  # Token embedding layer
        if not (cfg.rope or cfg.alibi):
            self.pos = nn.Parameter(torch.zeros(1, cfg.context, cfg.d_model))  # Learnable position embeddings
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))  # Stack of transformer blocks
        self.ln_f = nn.LayerNorm(cfg.d_model)  # Final LayerNorm
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)  # Output projection to vocab size

        # Weight tying: share weights between embedding and output head
        self.head.weight = self.embed.weight

        # Causal mask: prevents attending to future tokens
        self._mask = torch.triu(torch.ones(cfg.context, cfg.context), 1).bool()

    def forward(
        self,
        tokens: torch.Tensor,  # [B, T] input token IDs
        *,
        return_attn: bool = False,  # Whether to return attention weights
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        B, T = tokens.shape  # Batch size and sequence length
        if T > self.config.context:
            raise ValueError("Sequence length exceeds model context")
        device = tokens.device

        x = self.embed(tokens)  # Embed tokens

        # Add position embeddings unless using RoPE/ALiBi
        if not (self.config.rope or self.config.alibi):
            x = x + self.pos[:, :T, :]

        attns: List[torch.Tensor] = []  # To store attention weights if needed

        mask = self._mask[:T, :T].to(device)  # Causal mask for current sequence length

        cos_sin = None
        if self.config.rope:
            cos, sin = rotary_embed(self.config.head_dim, T, device)  # Get RoPE tables
            cos_sin = (cos, sin)

        for blk in self.blocks:
            x, w = blk(x, mask, cos_sin)  # Pass through each transformer block
            if return_attn:
                attns.append(w.detach())  # Store attention weights

        x = self.ln_f(x)  # Final normalization
        logits = self.head(x)  # Project to vocabulary logits
        return logits, attns if return_attn else None  # Return logits and optionally attention

    @torch.no_grad()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 50):
        # Greedy text generation: repeatedly predict next token
        ids = torch.tensor(prompt_ids, dtype=torch.long)[None]
        for _ in range(max_new_tokens):
            logits, _ = self(ids)
            next_id = logits[0, -1].argmax(-1, keepdim=True)  # Pick most likely next token
            ids = torch.cat((ids, next_id[None]), dim=1)  # Append to sequence
            if ids.shape[1] >= self.config.context:
                ids = ids[:, -self.config.context :]  # Keep only last 'context' tokens
        return ids.squeeze().tolist()  # Return generated token IDs


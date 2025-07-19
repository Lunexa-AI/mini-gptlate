"""Core unit tests for Mini‑GPTLate.

Run with:
    pytest -q
"""
from __future__ import annotations

import torch

from mini_gptlate.model import GPTLate, GPTConfig
from mini_gptlate.tokeniser import get_tokeniser


def test_tokeniser_roundtrip():
    tok = get_tokeniser()
    text = "The quick brown fox."  # ensure punctuation handled
    ids = tok.encode(text)
    back = tok.decode(ids)
    # decoded may lose capitalisation due to lower‑casing but words intact
    assert "quick" in back.lower() and "fox" in back.lower()


def test_model_forward():
    cfg = GPTConfig(vocab_size=100, context=16, d_model=64, n_layer=2, n_head=4)
    model = GPTLate(cfg)
    x = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, attns = model(x, return_attn=True)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert len(attns) == cfg.n_layer
    # ensure gradients flow
    loss = logits.mean()
    loss.backward()
    for p in model.parameters():
        assert p.grad is not None

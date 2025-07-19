"""Numerical sanity: one‑step perplexity should stay reasonable (<200).

We use a toy corpus and a tiny random model so this test is lenient; as the
project matures you can tighten the threshold or load a real checkpoint.
"""
from __future__ import annotations

import math
import torch

from mini_gptlate.model import GPTLate, GPTConfig
from mini_gptlate.tokeniser import get_tokeniser


def test_one_batch_perplexity():
    tok = get_tokeniser()
    text = "the cat sat on the mat." * 4
    ids = torch.tensor(tok.encode(text)[:16])[None]  # [1, T]

    cfg = GPTConfig(vocab_size=len(tok.vocab), context=16, d_model=64, n_layer=2, n_head=4)
    model = GPTLate(cfg)
    logits, _ = model(ids)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), ids.view(-1))
    ppl = math.exp(loss.item())
    assert ppl < 200  # smoke‑check upper bound

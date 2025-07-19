# Mini‑GPTLate

**Ultra‑light, 200‑line GPT‑2‑class model with live attention tracing**

---

## 1 Vision & Goals

| Aspect | Target |
| --- | --- |
| **Mission** | Make transformer language models transparent, hackable, and runnable everywhere—especially in classrooms and on commodity laptops—with <200 lines of clearly‑commented code. |
| **Primary users** | • Students learning deep‑learning internals |
| • Boot‑camp instructors & bloggers |  |
| • Educational researchers analysing attention |  |
| **Success metrics** | • <5 min install, <60 s inference setup on a 2‑core CPU |
| • ≥500 GitHub ⭐ in 90 days |  |
| • ≥5 published blog posts or classroom adoptions in 6 months |  |

---

## 2 Design Principles

1. **Radical Simplicity** – One file, ~200 effective LoC (excluding blank lines & comments).
2. **Transparency First** – Every tensor, shape, and math operation is visible through the `-trace` flag; no hidden kernels.
3. **Pedagogical Defaults** – Defaults favour comprehension over speed (e.g. no fused QKV by default, plainly written loops), but opt‑in performance flags are provided.
4. **Backend Agnostic** – Reference implementation in PyTorch; drop‑in swap to tinygrad via two‑line change.
5. **Off‑Grid Friendly** – No internet, GPU, or AVX‑512 required; pure‑CPU and offline tokeniser.
6. **State‑of‑the‑Art Hooks** – Modern research tricks (RoPE, ALiBi, residual‑stream QKV) included behind feature flags so they can be studied.
7. **Community First** – CI, code style, contributor guide, permissive licence (MIT), welcoming governance.

---

## 3 High‑Level Architecture

```
+------------------+         +----------------------+         +------------------+
| Tokeniser (BPE)  | ----▶ | Embedding  (V×d_model)| ----▶ | Positional Encode |
+------------------+         +----------------------+         +------------------+
                                                                   │
                                                                   ▼
                                                   (×N)  +------------------+
                                                         | TransformerBlock |
                                                         +------------------+
                                                                   │
                                                                   ▼
                                                         +------------------+
                                                         |  LM Head (Softmax)|
                                                         +------------------+

```

- **N = 6** (default) transformer blocks identical to GPT‑2 small.
- **d_model = 256**, **n_head = 8**, **d_ff = 4×d_model**.

### 3.1 TransformerBlock (≈32 LoC)

```python
class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model, d_model)
        )
    def forward(self, x, mask):
        q = k = v = self.ln1(x)
        a, w = self.attn(q, k, v, attn_mask=mask, need_weights=True)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, w  # return weights for tracer

```

*Returning `w` exposes **raw attention maps** to the tracer without extra passes.*

---

## 4 Live Attention Tracer (`-trace`)

| Feature | Description |
| --- | --- |
| **CLI Flag** | `python mini_gpt.py --prompt "..." --trace` |
| **Display** | Streams q, k, v tensors & draws attention heatmap frame‑by‑frame via [rich](https://github.com/Textualize/rich). |
| **Modes** | • `ascii` (terminal, default)  • `html` (opens browser with JS heatmap) |
| **Overhead** | <10 % extra latency due to single extra copy of weights. |
| **Pedagogy** | Colours head‑wise, pauses between layers (`--step`) so instructors can discuss each hop. |

Implementation sketch (≈30 LoC):

```python
if args.trace:
    tracer = AttentionTracer(mode=args.mode)
...
for i, block in enumerate(blocks):
    x, w = block(x, mask)
    if args.trace:
        tracer.log(i, w)

```

---

## 5 Implementation Details

| Area | Decision | Rationale |
| --- | --- | --- |
| **Tokenizer** | Offline BPE (100 k merges) trained on open‑license Wiki‑40B slice; shipped as `tokenizer.json`. | Removes network/download friction; still English‑competent. |
| **Positional Encoding** | Classic GPT2 learned vector; optional RoPE (`--rope`) or ALiBi (`--alibi`). | Teach legacy vs modern approaches; low code overhead. |
| **Norm Order** | Pre‑LayerNorm (GPT‑J style) toggled by `--prenorm`. | Compare training stability live. |
| **Optimiser** | AdamW by default; SGD fallback for explanation of differences. |  |
| **Mixed Precision** | bfloat16 optional; CPU bf16 supported on modern Intel/AMD. |  |
| **Parallelism** | Torch `nn.MultiheadAttention` already multi‑threaded; set `OMP_NUM_THREADS`. |  |
| **Backend Swap** | Import guard: `if USE_TINYGRAD: from tinygrad import Tensor as T`. Provide helper script to convert weights. |  |

Code budget (≈185 lines):

```
 25  Imports & args          12  Positional enc.
 32  Block class             15  Model wrapper
 18  Tokenizer utils         28  Trace visualiser
 22  CLI / sampling loop     20  train / save / load
-----                         -----
200  lines (excl. comments)

```

---

## 6 Performance Targets

| Device | Config | Throughput (tok/s) | Latency (ms/tok) |
| --- | --- | --- | --- |
| **Intel i5‑8250U (2017)** | 6L‑256d | 22 | 45 |
| **Apple M2 (2023)** | 8L‑384d | 55 | 18 |
| **Raspberry Pi 5** | 4L‑128d | 9 | 111 |

Test corpus: WikiText‑2 val, greedy decode, sequence length 128.

---

## 7 Scalability & Extensibility

- **Hyper‑param flags**: `-n_layer`, `-n_head`, `-d_model`, `-context`.
- **Checkpoint Format**: single `state_dict.pt` with versioning key to auto‑upgrade shapes.
- **LoRA Fine‑Tuning**: 60 LoC add‑on script demonstrates low‑rank adapter injection.
- **Distributed**: Because model fits in RAM, distribute *batches*, not parameters; instruct via `torchrun` docs.

---

## 8 Testing & CI

| Layer | Test | Metric |
| --- | --- | --- |
| Tokeniser | Round‑trip encode/decode sample sentences. | Exact match |
| Block | Compare forward pass vs ref GPT‑2 weights slice. | MSE <1e‑5 |
| End‑to‑end | 1‑step greedy output for fixed seed. | Checksum |

GitHub Actions pipeline runs pytest + flake8 + black (<45 s total).

---

## 9 Documentation & Teaching Aids

- **Narrated Notebook** – ‘Build a GPT from scratch in 90 min’ with step‑wise reveals and exercises.
- **SVG Diagrams** auto‑exported from model (`-export-graph` flag) using [torchviz].
- **Cheat‑Sheet** PDF of tensor shapes, colour‑coded.
- **Live Demo** streamlit app (optional GPU) for quick experimentation.

---

## 10 Community & Governance

1. **MIT Licence**—maximise adoption.
2. **Contributing Guide** – commit message convention, separate docs PRs welcomed.
3. **Code of Conduct** – Contributor Covenant v2.1.
4. **Roadmap** – public Projects board (GitHub) tracking upcoming features (e.g., KV‑cache visualiser, MoE branch).
5. **Release cadence** – semantic versioning; monthly minor releases, patch as needed.

---

## 11 Security & Privacy

- No external calls during runtime (offline).
- Optional entropy limiter in sampler to prevent toxic completions during demos.
- Supply‑chain pinned hashes for pip dependencies via `requirements.lock`.

---

## 12 Future Roadmap

| Quarter | Planned Feature | PEDAGOGICAL VALUE |
| --- | --- | --- |
| 2025‑Q3 | KV‑Cache visualiser, sampling temperature slider in tracer. | Shows effect of KV cache on speed. |
| 2025‑Q4 | FlashAttention‑style CPU algebra flag. | Contrast vanilla vs fused kernels. |
| 2026‑Q1 | Multilingual tokenizer pack. | Wider adoption across curriculums. |

---

## 13 Glossary

| Term | In Mini‑GPTLate |
| --- | --- |
| **Context Length** | `--context` (default = 256) |
| **d_model** | Embedding/hidden width |
| **n_head** | Attention heads per block |
| **Trace Frame** | One layer’s attention matrix |

---

> “If you can’t explain it simply, you don’t understand it well enough.” — Albert Einstein
> 
> 
> Mini‑GPTLate aims to make the transformer simple enough to explain *live on the projector* while still being *useful* for genuine research.  Enjoy hacking!
>

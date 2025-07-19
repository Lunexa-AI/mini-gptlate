# Mini‑GPTLate Quick‑Start 🏃‍♂️

> **Goal:** get from `git clone` to *live attention heat‑maps* in under 5 minutes on any laptop.

---

## 1  Clone & install

```bash
# ➊ grab the repo
$ git clone https://github.com/your‑handle/mini‑gptlate.git
$ cd mini‑gptlate

# ➋ create a fresh env (optional but tidy)
$ python3 -m venv .venv
$ source .venv/bin/activate

# ➌ install deps (CPU‑only)
$ pip install -e .[dev]         # includes rich, pytest, flake8 …

# ➍ fetch the GPT‑2 tokenizer once (~450 kB)
$ curl -O https://huggingface.co/gpt2/raw/main/tokenizer.json \
       -o mini_gptlate/tokenizer.json
```

*Python ≥ 3.9, PyTorch ≥ 2.0 recommended.*

---

## 2  Generate your first tokens

```bash
# plain usage (greedy decode)
$ python -m mini_gptlate --prompt "The sky is" --max_new 20
```

Add `--trace` to see **live heat‑maps** after every step:

```bash
$ python -m mini_gptlate --prompt "Once upon a" --trace --max_new 30
```

You’ll get Unicode shaded tables like:

```
┏━━━━━━━━━━━━━┳━━┳━━┳━━ ...
┃ tok→tok     ┃On ┃ce ┃▁u ...
┣━━━━━━━━━━━━━╋━━╋━━╋━━ ...
┃ On          ┃█ ▃ ▂ ...
┃ ce          ┃▂ █ ▅ ...
...            ...
```

Each row shows **how much** the word (row) attends to previous words (columns).

---

## 3  Training on your own corpus

### 3·1  Local folder of `.txt`

```bash
$ python -m mini_gptlate.train \
    --data my_corpus/*.txt \
    --epochs 2 \
    --batch 16 \
    --cuda           # optional GPU
```

### 3·2  Hugging Face streaming (no download)

```bash
$ python -m mini_gptlate.train \
    --hf wikitext \
    --split wiki.train.raw \
    --epochs 1 --compile
```

*`--compile` toggles PyTorch 2 dynamic compilation for \~20‑30 % speed‑up.*

---

## 4  LoRA fine‑tuning in minutes

```bash
# assume you converted GPT‑2 small to a 6‑layer classroom model
$ python -m mini_gptlate.convert_hf_gpt2 --layers 6 --out gpt2_6L.pt

# LoRA rank‑4 adapters (~4 MB)
$ python -m mini_gptlate.lora_finetune \
    --data lyrics/*.txt \
    --ckpt gpt2_6L.pt \
    --steps 800 \
    --out lora_lyrics.pt
```

Load adapters at inference:

```python
base = torch.load("gpt2_6L.pt")
adapt = torch.load("lora_lyrics.pt")
model.load_state_dict(base, strict=False)
model.load_state_dict(adapt, strict=False)
```

---

## 5  Export a model graph (SVG)

```bash
$ python - <<'PY'
from mini_gptlate.model import GPTLate
from torchviz import make_dot
model = GPTLate()
svg = make_dot(model(torch.zeros(1,8,dtype=torch.long)[0])[0]).render("graph", format="svg")
print("Saved", svg)
PY
```

Embed the SVG in slides or docs.

---

## 6  Record attention frames → GIF

```bash
$ python -m mini_gptlate --prompt "Hello world" --trace --max_new 30 \
      | tee >(ansi2gif > runs/trace.gif)
```

> ✨ Pro‑tip: [`ansi2gif`](https://github.com/maaslalani/ansi2gif) converts terminal ANSI frames straight into gifs.

---

## 7  Run tests locally (optional safety net)

```bash
$ pytest -q               # smoke + perplexity sanity
```

---

## 8  Troubleshooting

| Symptom                             | Fix                                                                                           |
| ----------------------------------- | --------------------------------------------------------------------------------------------- |
| `FileNotFoundError: tokenizer.json` | Download GPT‑2 tokenizer or train your own (`tokenizers` lib).                                |
| Very slow generation                | Pass `--compile` (CPU) or `--cuda` (GPU).                                                     |
| Decoder shows `▁` characters        | That’s the BPE whitespace symbol—`decode()` strips it, but log‑prob printouts may include it. |
| `datasets` import error             | `pip install datasets` or avoid `--hf` flag.                                                  |

---

### Happy hacking!

If something feels confusing, open an issue or a discussion—mini‑GPTLate is *designed* to be dissected, extended, and taught live.

# Mini‑GPTLate Classroom Recipes 🍳

*Snack‑size copy‑pasta for demos, labs, and blog snippets.*

---

## 1  Hello‑world generation in three lines

```python
from mini_gptlate import cli
cli.main(["--prompt", "Once upon a", "--max_new", "25"])
```

Add attention:

```python
cli.main(["--prompt", "Quantum computers", "--trace", "--max_new", "30"])
```

---

## 2  Compare classic positions vs RoPE in real‑time

```python
from mini_gptlate.model import GPTLate, GPTConfig
from mini_gptlate.tokeniser import get_tokeniser
from mini_gptlate.tracer import AttentionTracer
import torch

tok = get_tokeniser()
ids = torch.tensor(tok.encode("Sin waves resemble rotations"))[None]

for label, rope_flag in [("Learned", False), ("RoPE", True)]:
    print(f"\n=== {label} positions ===")
    m = GPTLate(GPTConfig(rope=rope_flag))
    _, attns = m(ids, return_attn=True)
    AttentionTracer(ids[0].tolist(), tok.decode).log(0, attns[0])
```

---

## 3  Measure Torch 2 speed‑up

```python
import torch, time, contextlib
from mini_gptlate import model

plain   = model.GPTLate()
compiled = torch.compile(plain)

inp = torch.randint(0, plain.config.vocab_size, (4,128))

def bench(m):
    t0=time.time(); [m(inp) for _ in range(20)]; return time.time()-t0

print("eager   ", bench(plain))
print("compiled", bench(compiled))
```

---

## 4  Export SVG of the network

```python
from mini_gptlate.model import GPTLate
from torchviz import make_dot
model = GPTLate()
svg_path = make_dot(model(torch.zeros(1,8,dtype=torch.long)[0])[0]).render("gptlate", format="svg")
print("saved to", svg_path)
```

Embed the `gptlate.svg` in slides.

---

## 5  Save attention frames → GIF

```bash
python -m mini_gptlate.cli --prompt "AI is" --trace --max_new 50 \
  | ansi2gif --output attn.gif
```

*Requires `pip install ansi2gif`.*

---

## 6  Stream WikiText‑2 without download

```bash
python -m mini_gptlate.train --hf wikitext --split wiki.train.raw \
       --epochs 1 --batch 8 --compile
```

---

## 7  LoRA adapters in <5 MB

```bash
python -m mini_gptlate.lora_finetune --data poetry/*.txt \
       --ckpt gpt2_6L.pt --steps 500 --rank 4 --out lora_poetry.pt
```

Load at inference:

```python
import torch
from mini_gptlate.model import GPTLate, GPTConfig
base = torch.load("gpt2_6L.pt")
adapt= torch.load("lora_poetry.pt")
model = GPTLate(GPTConfig())
model.load_state_dict(base, strict=False)
model.load_state_dict(adapt, strict=False)
```

---

## 8  Run smoke‑tests before class

```bash
pytest -q
```

---

### Have fun hacking ✨

Feel free to PR your own recipes!

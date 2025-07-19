<div align="center">

# **Mini‑GPTLate**

*“A GPT‑2 you can read like a book, run on a potato, and teach in a single lecture.”*

![Stars](https://img.shields.io/github/stars/your‑handle/mini‑gptlate?style=flat‑square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat‑square)
![Python](https://img.shields.io/badge/python-3.9%2B-blue?style=flat‑square)

</div>

---

## ✨ What makes it special?

| Pillar               | Why it matters                                                                                                              |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **🧠 Readable**      | <200 effective LoC for the core model. Every tensor, shape, and attention map is visible & hackable.                        |
| **🚂 Teachable**     | Live ASCII heat‑maps, narrated notebook, and recipe snippets—perfect for boot‑camps or a Friday brown‑bag.                  |
| **🪶 Feather‑light** | Runs fully offline, CPU‑only, **<1 GB RAM** in default config; optional `torch.compile` flag for speed.                     |
| **🌍 Inclusive**     | One‑command script to build a **20 k‑vocab tokenizer** from *any* UTF‑8 corpus—great for Swahili classrooms or Hindi blogs. |
| **🔌 Plug‑n‑play**   | Convert GPT‑2 weights, fine‑tune LoRA adapters, or train from scratch on WikiText—all tools included.                       |

---

## 🚀 Quick start (60 s)

```bash
# 1. Clone & enter
$ git clone https://github.com/your‑handle/mini‑gptlate.git
$ cd mini‑gptlate

# 2. (Optional) create venv
$ python3 -m venv .venv && source .venv/bin/activate

# 3. Install deps (CPU)
$ pip install -e .[dev]

# 4. Get a tokenizer (~450 kB)
$ curl -L https://huggingface.co/gpt2/raw/main/tokenizer.json \
       -o mini_gptlate/tokenizer.json

# 5. Generate text with live attention
$ python -m mini_gptlate --prompt "In 2030, AI" --trace --max_new 40
```

> **Need another language?**
> `python tools/build_tokenizers.py --text /path/to/*.txt --out hindi_tok.json`

---

## 🏗️ Project layout

```
mini_gptlate/
├─ model.py            # 200‑line GPT‑2‑small core
├─ tokeniser.py        # offline BPE
├─ tracer.py           # ASCII attention heat‑maps
├─ cli.py              # text generation entry‑point
├─ train.py            # CPU training loop (+HF streaming)
├─ lora_finetune.py    # rank‑4 adapter demo
├─ convert_hf_gpt2.py  # bring GPT‑2 weights
├─ tools/
│   ├─ build_tokenizers.py   # multilingual micro‑BPE
│   └─ build_docs.py         # offline HTML docs
└─ docs/   quickstart.md · recipes.md · tutorial.ipynb
```

---

## 📚 Docs & teaching assets

* **`docs/quickstart.md`** – 5‑minute guide.
* **Notebook:** `docs/mini_gptlate_tutorial.ipynb` – *“Build a GPT from scratch in 90 min”*.
* **Recipe sheet:** `docs/recipes.md` – copy‑pasta demos.
* **Offline book:** `python tools/build_docs.py` → `docs_book/index.html` (works on a USB stick).

---

## 🛠️ Common workflows

| Goal                       | Command                                                                        |
| -------------------------- | ------------------------------------------------------------------------------ |
| **Slice GPT‑2 → 6 layers** | `python -m mini_gptlate.convert_hf_gpt2 --layers 6 --out gpt2_6L.pt`           |
| **Train on local `.txt`**  | `python -m mini_gptlate.train --data data/*.txt --epochs 2`                    |
| **Stream WikiText‑2**      | `python -m mini_gptlate.train --hf wikitext --split wiki.train.raw --epochs 1` |
| **LoRA fine‑tune**         | `python -m mini_gptlate.lora_finetune --data lyrics/*.txt --ckpt gpt2_6L.pt`   |
| **Compile for speed**      | add `--compile` flag (PyTorch 2)                                               |

---

## 🤝 Contributing (when CI is on)

1. Fork → create branch.
2. `pre‑commit run -a` (lint & tests).
3. PR with a screenshot of the tracer if UI changes.
4. Be kind → we follow the [Contributor Covenant](https://www.contributor-covenant.org/).

> **First‑timers welcome** – look for *good‑first‑issue* labels.

---

## ⚖️ License

MIT. Do what you love, just keep the headers.

---

*Happy hacking & teaching!* ✨


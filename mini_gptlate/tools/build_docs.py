"""Assemble an **offline HTML book** (docs_book/) using *jupyter‑book*.

It grabs the three key teaching artefacts:
    • docs/quickstart.md
    • docs/recipes.md
    • docs/mini_gptlate_tutorial.ipynb
and stitches them into a single static site that works without internet.

### Requirements
```bash
pip install jupyter-book  # one‑time, ~20 MB of deps
```

### Usage
```bash
python tools/build_docs.py          # outputs docs_book/_build/html/index.html
open docs_book/_build/html/index.html
```

Re‑run the script whenever the markdown or notebook changes.
"""
from __future__ import annotations

import subprocess
import shutil
from pathlib import Path
import sys
import textwrap

ROOT = Path(__file__).resolve().parent.parent  # repo root
SRC  = ROOT / "docs"
BOOK = ROOT / "docs_book"

FILES = [
    SRC / "quickstart.md",
    SRC / "recipes.md",
    SRC / "mini_gptlate_tutorial.ipynb",
]

CONFIG_YML = textwrap.dedent("""
    # Minimal Jupyter‑Book config
    title: Mini‑GPTLate Classroom Book
    author: Mini‑GPTLate Team
    language: en
    logo:  ''
    html_theme: sphinx_book_theme
    parse:
      myst_enable_extensions: [colon_fence]
""")

TOC_YML = textwrap.dedent("""
    format: jb-book
    root: quickstart
    chapters:
      - file: recipes
      - file: mini_gptlate_tutorial
""")


def ensure_prereqs():
    """Abort if jupyter‑book CLI isn’t available."""
    if shutil.which("jupyter-book") is None:
        sys.exit("[!] jupyter‑book not found. Install with:  pip install jupyter-book")


def stage_sources():
    dst_src = BOOK / "src"
    dst_src.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        shutil.copy2(f, dst_src / f.name)
    (dst_src / "_config.yml").write_text(CONFIG_YML)
    (dst_src / "_toc.yml").write_text(TOC_YML)


def build_book():
    dst_src = BOOK / "src"
    cmd = ["jupyter-book", "build", str(dst_src), "--path-output", str(BOOK)]
    print("[+] Running:", " ".join(cmd))
    subprocess.check_call(cmd)
    print("[✓] HTML ready at", BOOK / "_build" / "html" / "index.html")


def main():
    ensure_prereqs()
    stage_sources()
    build_book()


if __name__ == "__main__":
    main()

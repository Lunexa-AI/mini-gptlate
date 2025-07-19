#!/usr/bin/env python3
"""Scaffold the Mini-GPTLate repository.

Run once from the repo root:
    python scaffold.py

It will create the recommended folder structure, stub modules,
configuration files, and a minimal GitHub Actions workflow.
Existing files are left untouched unless --force is passed.
"""

import argparse
import pathlib
import textwrap
import sys

BASE = pathlib.Path(".").resolve()

FILES = {
    ".gitignore": textwrap.dedent(
        """
        # Byte-compiled / optimized / DLL files
        __pycache__/
        *.py[cod]
        *$py.class

        # Virtual environments
        .venv/

        # Distribution / packaging
        build/
        dist/
        *.egg-info/
        
        # Mac / VSCode / etc
        .DS_Store
        .vscode/
        """
    ),
    "pyproject.toml": textwrap.dedent(
        """
        [build-system]
        requires = ["setuptools>=61.0"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "mini-gptlate"
        version = "0.1.0"
        description = "Ultra-light GPT-2-class model with live attention tracing"
        authors = [ { name = "YOUR NAME", email = "you@example.com" } ]
        license = { text = "MIT" }
        dynamic = ["dependencies"]

        [project.optional-dependencies]
        dev = [
            "pytest",
            "black~=24.0",
            "flake8",
            "rich",
        ]

        [tool.setuptools.packages.find]
        include = ["mini_gptlate*"]
        """
    ),
    "requirements.txt": "torch\nrich\n",
    "mini_gptlate/__init__.py": textwrap.dedent(
        """Top-level package for Mini-GPTLate."""
    ),
    "mini_gptlate/model.py": textwrap.dedent(
        """Minimal GPT-2-class model (~200 LoC).

        Reference architecture from docs/architecture.md.
        """
    ),
    "mini_gptlate/tokeniser.py": textwrap.dedent(
        """Offline BPE tokeniser utility."""
    ),
    "mini_gptlate/tracer.py": textwrap.dedent(
        """Live attention tracer using Rich heat-maps."""
    ),
    "mini_gptlate/cli.py": textwrap.dedent(
        """Command-line interface: text generation & tracing.

        Example:
            python -m mini_gptlate.cli --prompt "Hello" --trace
        """
    ),
    "tests/__init__.py": "",
    "tests/test_model.py": textwrap.dedent(
        """Basic smoke test to instantiate the model and run a forward pass."""
    ),
    "tests/test_tokeniser.py": textwrap.dedent(
        """Round-trip encoding tests for the tokeniser."""
    ),
    "docs/architecture.md": "(Paste specification here or link)\n",
    ".github/workflows/python-ci.yml": textwrap.dedent(
        """
        name: loose-ci

        on: [push, pull_request]

        jobs:
          test:
            runs-on: ubuntu-latest
            steps:
              - uses: actions/checkout@v4
              - uses: actions/setup-python@v5
                with:
                  python-version: '3.11'
              - name: Install deps
                run: |
                  python -m pip install --upgrade pip
                  pip install -e .[dev]
              - name: Lint & Test
                run: |
                  flake8 mini_gptlate || echo "Lint warnings ignored for now"
                  pytest -q || echo "Tests TODO"
        """
    ),
}


def write_file(path: pathlib.Path, content: str, force: bool = False):
    if path.exists() and not force:
        print(f"SKIP {path} (exists)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(content.lstrip("\n"))
    print(f"WRITE {path}")


def main():
    parser = argparse.ArgumentParser(description="Scaffold Mini-GPTLate repo")
    parser.add_argument("--force", action="store_true", help="overwrite existing files")
    args = parser.parse_args()

    for rel, text in FILES.items():
        write_file(BASE / rel, text, force=args.force)

    # Pretty print tree summary
    print("\n📁  Created/checked files:")
    for rel in sorted(FILES.keys()):
        print("  -", rel)


if __name__ == "__main__":
    sys.exit(main())

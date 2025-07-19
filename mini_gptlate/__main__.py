"""Allow `python -m mini_gptlate` to behave like the CLI helper."""
from __future__ import annotations

import sys
from .cli import main as cli_main

if __name__ == "__main__":
    # Forward all arguments to the CLI.
    cli_main(sys.argv[1:])

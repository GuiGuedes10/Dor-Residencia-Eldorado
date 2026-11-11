"""
Top-level module execution entrypoint for REIA (`python3 -m reia`).

Delegates execution directly to the command-line interface in `reia.pipeline.cli`.
"""

import sys
from reia.pipeline.cli import main

if __name__ == "__main__":
    sys.exit(main())

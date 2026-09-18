#!/usr/bin/env python3
"""Print a fixture bake-off table (simulated, not GPU numbers).

    python examples/bakeoff.py
    python examples/bakeoff.py --output reports
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectrace.cli import main  # noqa: E402


if __name__ == "__main__":
    extra = sys.argv[1:]
    raise SystemExit(main(["bakeoff", "--traces", str(ROOT / "traces"), *extra]))

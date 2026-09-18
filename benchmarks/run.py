"""Benchmark runner used by the CLI and ``python -m benchmarks.run``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python benchmarks/run.py` from a clone without installing.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectrace.cli import main as spectrace_main  # noqa: E402
from spectrace.decoding import METHODS  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="spectrace fixture bake-off runner")
    parser.add_argument("--traces", default=str(ROOT / "traces"))
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--output", default="")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    cmd = [
        "bakeoff",
        "--traces",
        args.traces,
        "--methods",
        args.methods,
        "--seed",
        str(args.seed),
    ]
    if args.output:
        cmd.extend(["--output", args.output])
    return spectrace_main(cmd)


if __name__ == "__main__":
    raise SystemExit(main())

"""Command-line interface: ``spectrace run`` and ``spectrace bakeoff``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from spectrace import __version__
from spectrace.decoding import METHODS
from spectrace.metrics import load_price_table
from spectrace.providers import ProviderError, resolve_provider
from spectrace.replay import run_bakeoff
from spectrace.report import ascii_table, render_json, render_markdown, write_reports
from spectrace.traces import load_traces, traces_summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectrace",
        description="Bake-off speculative decoding / serving tricks on multi-step agent traces.",
    )
    parser.add_argument("--version", action="version", version=f"spectrace {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Replay one trace (or a directory) with a single method.")
    run.add_argument("--trace", required=True, help="JSON trace file or directory of traces.")
    run.add_argument(
        "--method",
        required=True,
        choices=METHODS,
        help="Serving method to simulate.",
    )
    _add_common(run)

    bake = sub.add_parser("bakeoff", help="Compare methods on a set of traces and print a table.")
    bake.add_argument("--traces", required=True, help="JSON trace file or directory.")
    bake.add_argument(
        "--methods",
        default=",".join(METHODS),
        help="Comma-separated methods (default: baseline,mock_speculative).",
    )
    _add_common(bake)

    listed = sub.add_parser("list-traces", help="Show fixture traces without running them.")
    listed.add_argument("--traces", default="traces", help="Trace file or directory.")
    return parser


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed for the mock decoder.")
    p.add_argument("--gamma", type=int, default=5, help="Draft window size for mock_speculative.")
    p.add_argument("--output", help="Directory to write report.json and report.md.")
    p.add_argument(
        "--format",
        default="table",
        choices=("table", "json", "md", "all"),
        help="Stdout format (default: table).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and validate traces; do not simulate serving.",
    )
    p.add_argument(
        "--provider",
        default="mock-local",
        help="mock-local (default) or openai-compat (requires SPECTRACE_BASE_URL).",
    )
    p.add_argument(
        "--price-table",
        help="JSON file of USD-per-million rates. Default: all zeros.",
    )
    p.add_argument("--model", default="mock-local", help="Price-table key (default mock-local = $0).")


def _load_prices(path: str | None):
    if not path:
        return load_price_table(None)
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return load_price_table(raw)


def _print_summary(summary, fmt: str) -> None:
    if fmt == "json":
        sys.stdout.write(render_json(summary))
        return
    if fmt == "md":
        sys.stdout.write(render_markdown(summary))
        return
    banner = "FIXTURE / SIMULATED — not GPU measurements. Default cost is $0."
    if fmt == "all":
        print(banner)
        print(ascii_table(summary.runs))
        print()
        sys.stdout.write(render_markdown(summary))
        return
    print(banner)
    print(ascii_table(summary.runs))
    cost = summary.cost_per_successful_task_usd
    cost_s = "n/a" if cost != cost else f"{cost:.4f}"
    print(f"cost per successful task (USD): {cost_s}   successful runs: {summary.n_success}/{len(summary.runs)}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "list-traces":
        traces = load_traces(args.traces)
        print(json.dumps(traces_summary(traces), indent=2))
        return 0

    if args.cmd == "run":
        traces = load_traces(args.trace)
        methods = [args.method]
        label = Path(args.trace).name
    else:
        traces = load_traces(args.traces)
        methods = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in methods if m not in METHODS]
        if unknown:
            parser.error(f"unknown method(s): {unknown}")
        label = Path(args.traces).name

    if args.dry_run:
        print(
            f"dry-run: {len(traces)} trace(s), method(s)={methods}, "
            f"provider={args.provider}, gamma={args.gamma}, seed={args.seed}"
        )
        for row in traces_summary(traces):
            print(
                f"  - {row['id']}  type={row['task_type']}  "
                f"messages={row['n_messages']}  tool_calls={row['n_tool_calls']}"
            )
        return 0

    try:
        provider = resolve_provider(args.provider)
    except ProviderError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    prices = _load_prices(args.price_table)
    summary = run_bakeoff(
        traces,
        methods,
        seed=args.seed,
        gamma=args.gamma,
        provider=provider,
        price_table=prices,
        model=args.model,
    )
    _print_summary(summary, args.format)
    if args.output:
        json_path, md_path = write_reports(summary, args.output, stem="report")
        print(f"wrote {json_path}")
        print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

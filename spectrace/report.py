"""JSON + Markdown bake-off reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spectrace.metrics import BakeoffSummary, RunMetrics

FIXTURE_BANNER = (
    "**FIXTURE / SIMULATED** — numbers come from the offline mock decoder, "
    "not from GPUs or paid APIs. Do not cite them as hardware results."
)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_cost(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return f"{value:.4f}"


def _fmt_success(ok: bool) -> str:
    return "yes" if ok else "no"


def markdown_table(runs: list[RunMetrics]) -> str:
    headers = [
        "trace",
        "method",
        "latency_ms",
        "accept_rate",
        "tokens_in",
        "tokens_out",
        "draft_acc/prop",
        "success",
        "cost_usd",
        "failure",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in runs:
        draft = "—"
        if r.draft_tokens_proposed:
            draft = f"{r.draft_tokens_accepted}/{r.draft_tokens_proposed}"
        row = [
            r.trace_id,
            r.method,
            f"{r.wall_latency_ms:.1f}",
            _fmt_rate(r.accept_rate),
            str(r.tokens_in),
            str(r.tokens_out),
            draft,
            _fmt_success(r.success),
            _fmt_cost(r.cost_estimate_usd),
            r.failure_mode or "—",
        ]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_markdown(summary: BakeoffSummary, *, title: str = "spectrace bake-off") -> str:
    cost = _fmt_cost(summary.cost_per_successful_task_usd)
    return "\n".join(
        [
            f"# {title}",
            "",
            FIXTURE_BANNER,
            "",
            markdown_table(summary.runs),
            "",
            f"- runs: **{len(summary.runs)}**",
            f"- successful task-runs: **{summary.n_success}**",
            f"- cost per successful task (USD): **{cost}**",
            "",
            "Default prices are $0. Pass `--price-table` to plug in a vendor table.",
            "",
        ]
    )


def render_json(summary: BakeoffSummary) -> str:
    payload: dict[str, Any] = summary.to_dict()
    payload["banner"] = FIXTURE_BANNER
    return json.dumps(payload, indent=2) + "\n"


def write_reports(summary: BakeoffSummary, output_dir: str | Path, *, stem: str = "report") -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(render_json(summary), encoding="utf-8")
    md_path.write_text(render_markdown(summary), encoding="utf-8")
    return json_path, md_path


def ascii_table(runs: list[RunMetrics]) -> str:
    """Fixed-width table for terminal output."""
    cols = [
        ("trace", 22, lambda r: r.trace_id),
        ("method", 18, lambda r: r.method),
        ("latency_ms", 12, lambda r: f"{r.wall_latency_ms:.1f}"),
        ("accept", 8, lambda r: _fmt_rate(r.accept_rate)),
        ("tok_in", 8, lambda r: str(r.tokens_in)),
        ("tok_out", 8, lambda r: str(r.tokens_out)),
        ("ok", 4, lambda r: "Y" if r.success else "N"),
        ("cost", 8, lambda r: _fmt_cost(r.cost_estimate_usd)),
    ]
    header = "  ".join(name.ljust(width) for name, width, _ in cols)
    rule = "  ".join("-" * width for _, width, _ in cols)
    rows = [header, rule]
    for r in runs:
        rows.append("  ".join(fn(r).ljust(width) for _, width, fn in cols))
    return "\n".join(rows)

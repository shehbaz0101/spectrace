#!/usr/bin/env python3
"""Redraw the dual-brain accept figure from a spectrace grade JSON.

Reads the ``rows`` list produced by ``spectrace grade`` (or the checked-in
LIVE report) and writes a ~1200px PNG. Never calls an API.

    python scripts/plot_dual_brain.py \
      --input docs/live-jev-grade-15traces-20261010.json \
      --output docs/dual-brain-accept.png

Needs matplotlib (optional; not a package dependency):

    pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


# Okabe–Ito, colorblind-safe.
TOKEN = "#E69F00"
JEV = "#009E73"
REJECT = "#D55E00"
TASK = "#0072B2"
INK = "#1C1917"
MUTED = "#57534E"
PAPER = "#FBF8F3"
PANEL = "#FFFFFF"
RULE = "#E7E0D6"

TRACE_LABELS = {
    "code_fix_01": "code",
    "data_analysis_01": "data",
    "destructive_shell_01": "shell",
    "destructive_sql_01": "sql",
    "hallucinated_tool_01": "halluc",
    "ignore_tool_error_01": "ignore",
    "loop_search_01": "loop",
    "multihop_retrieval_01": "hop",
    "off_goal_drift_01": "drift",
    "premature_answer_01": "early",
    "research_qa_01": "qa",
    "stall_churn_01": "stall",
    "travel_planner_01": "trip",
    "wrong_args_01": "args",
    "wrong_doc_01": "doc",
}


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def _as_rate(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_grade(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "rows" not in payload:
        raise SystemExit(f"{path}: expected a grade JSON object with a 'rows' list")
    rows = payload["rows"]
    if not isinstance(rows, list) or not rows:
        raise SystemExit(f"{path}: 'rows' is empty")
    return payload


def _trace_order(rows: list[dict[str, Any]]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for row in rows:
        seen.setdefault(str(row["trace"]), None)
    return list(seen)


def _style_axes(ax) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color(RULE)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.yaxis.label.set_color(MUTED)
    ax.grid(axis="y", color=RULE, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def _groups(rows: list[dict[str, Any]]) -> list[tuple[str, int, int]]:
    groups: list[tuple[str, int, int]] = []
    start = 0
    for i, row in enumerate(rows):
        nxt = rows[i + 1]["trace"] if i + 1 < len(rows) else None
        if nxt != row["trace"]:
            groups.append((str(row["trace"]), start, i))
            start = i + 1
    return groups


def render(payload: dict[str, Any], output: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        raise SystemExit(
            "matplotlib is required to draw the figure.\n"
            "  pip install matplotlib\n"
            "(It is optional — not a spectrace runtime or CI dependency.)"
        ) from None

    rows = payload["rows"]
    traces = _trace_order(rows)
    n_steps = len(rows)
    n_jev = sum(1 for r in rows if _as_bool(r.get("jev_accept")))
    n_reject = n_steps - n_jev
    n_task_traces = len({str(r["trace"]) for r in rows if _as_bool(r.get("task_success"))})
    n_traces = int(payload.get("n_traces") or len(traces))
    token_vals = [_as_rate(r.get("token_accept")) for r in rows]
    present_tokens = [v for v in token_vals if v is not None]
    token_min = min(present_tokens) if present_tokens else 0.0
    token_max = max(present_tokens) if present_tokens else 0.0

    live = bool(payload.get("live")) or str(payload.get("provider", "")).lower() == "jev"
    provider = str(payload.get("provider") or "unknown")
    run_at = str(payload.get("run_at") or "").strip()
    rule = str(payload.get("jev_accept_rule") or "tool_ok>=0.6 and progress>=0.5 and abort<0.5")
    jev_label = "LIVE TypeSafe Jev" if live else f"Jev ({provider})"

    groups = _groups(rows)
    per_trace_token: list[float] = []
    per_trace_jev: list[float] = []
    per_trace_task: list[float] = []
    for tid, a, b in groups:
        chunk = rows[a : b + 1]
        toks = [_as_rate(r.get("token_accept")) for r in chunk]
        toks = [v for v in toks if v is not None]
        per_trace_token.append(sum(toks) / len(toks) if toks else 0.0)
        per_trace_jev.append(sum(1 for r in chunk if _as_bool(r.get("jev_accept"))) / len(chunk))
        # task_success is a trace-level flag copied onto every step.
        per_trace_task.append(1.0 if any(_as_bool(r.get("task_success")) for r in chunk) else 0.0)

    n_g = len(groups)
    crowded = n_g > 8
    # 1200px wide. Extra height when many traces need rotated labels.
    fig_h = 8.60 if crowded else 7.80
    fig = plt.figure(figsize=(12.00, fig_h), dpi=100, facecolor=PAPER)
    gs = fig.add_gridspec(
        4,
        3,
        height_ratios=[0.72, 0.64, 2.45, 2.55],
        hspace=0.62 if crowded else 0.42,
        wspace=0.06,
        left=0.068,
        right=0.985,
        top=0.955,
        bottom=0.13 if crowded else 0.075,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.set_axis_off()
    title_ax.text(
        0.0,
        0.74,
        "Semantic acceptor  ≠  token acceptor",
        fontsize=18,
        fontweight="bold",
        color=INK,
        va="center",
        ha="left",
        transform=title_ax.transAxes,
    )
    title_ax.text(
        0.0,
        0.22,
        "System 1 (TypeSafe Jev) is a step grader.  System 2 (mock_speculative) only drafts tokens — not GPUs.",
        fontsize=9.4,
        color=MUTED,
        va="center",
        ha="left",
        transform=title_ax.transAxes,
    )
    # Compact color key, top-right — keeps the plot area free.
    key_x = 0.995
    if n_reject:
        key_rows = (
            (0.86, TOKEN, "s", "token_accept  mock"),
            (0.62, JEV, "o", "jev yes" + ("  LIVE" if live else "")),
            (0.38, REJECT, "v", "jev no"),
            (0.14, TASK, "s", "task success"),
        )
    else:
        key_rows = (
            (0.78, TOKEN, "s", "token_accept  mock"),
            (0.46, JEV, "o", "jev_accept" + ("  LIVE" if live else "")),
            (0.14, TASK, "s", "task success"),
        )
    for y, color, marker, text in key_rows:
        title_ax.scatter(
            [key_x - 0.205],
            [y],
            s=38,
            marker=marker,
            color=color,
            transform=title_ax.transAxes,
            clip_on=False,
        )
        title_ax.text(
            key_x - 0.185,
            y,
            text,
            fontsize=7.4,
            color=INK,
            va="center",
            ha="left",
            transform=title_ax.transAxes,
        )

    chips = [
        (f"{n_jev} / {n_steps}", "Jev-gated accept", jev_label, JEV),
        (f"{n_task_traces} / {n_traces}", "task success", "structural (whole trace)", TASK),
        (f"{token_min:.2f} – {token_max:.2f}", "token_accept range", "mock_speculative  ·  not GPUs", TOKEN),
    ]
    for i, (value, label, sub, color) in enumerate(chips):
        ax = fig.add_subplot(gs[1, i])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_axis_off()
        ax.add_patch(
            FancyBboxPatch(
                (0.02, 0.08),
                0.96,
                0.84,
                boxstyle="round,pad=0.012,rounding_size=0.06",
                linewidth=1.2,
                edgecolor=color,
                facecolor=PANEL,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(0.5, 0.70, value, fontsize=15.5, fontweight="bold", color=color, ha="center", va="center")
        ax.text(0.5, 0.40, label, fontsize=8.6, color=INK, ha="center", va="center")
        ax.text(0.5, 0.18, sub, fontsize=7.2, color=MUTED, ha="center", va="center")

    # --- Per-trace grouped bars (the glance) ---
    ax_t = fig.add_subplot(gs[2, :])
    _style_axes(ax_t)
    ax_t.set_title(
        "Per trace   ·   mean token_accept vs Jev-gated accept vs task success",
        loc="left",
        fontsize=9.2,
        color=INK,
        pad=6,
    )
    x = list(range(n_g))
    width = 0.26
    ax_t.bar([i - width for i in x], per_trace_token, width=width, color=TOKEN, edgecolor="#B06D00", linewidth=0.4, zorder=2)
    ax_t.bar(x, per_trace_jev, width=width, color=JEV, edgecolor="#065F46", linewidth=0.4, zorder=2)
    ax_t.bar([i + width for i in x], per_trace_task, width=width, color=TASK, edgecolor="#0C4A6E", linewidth=0.4, zorder=2)
    if not crowded:
        for i, (tok, jev, task) in enumerate(zip(per_trace_token, per_trace_jev, per_trace_task)):
            ax_t.text(i - width, tok + 0.035, f"{tok:.2f}", ha="center", va="bottom", fontsize=7.2, color="#9A6700")
            ax_t.text(i, jev + 0.035, f"{jev:.0%}", ha="center", va="bottom", fontsize=7.2, color="#065F46")
            ax_t.text(i + width, task + 0.035, f"{task:.0%}", ha="center", va="bottom", fontsize=7.2, color="#0C4A6E")
    ax_t.set_ylim(0, 1.22)
    ax_t.set_xlim(-0.55, n_g - 0.45)
    ax_t.set_ylabel("rate")
    ax_t.set_yticks([0.0, 0.5, 1.0])
    ax_t.set_yticklabels(["0", "0.5", "1.0"])
    ax_t.set_xticks(x)
    ax_t.set_xticklabels(
        [TRACE_LABELS.get(tid, tid) for tid, _a, _b in groups],
        fontsize=7.6 if crowded else 8.8,
        color=INK,
        rotation=40 if crowded else 0,
        ha="right" if crowded else "center",
    )

    # --- Per-step: token_accept varies; Jev rejects when the step is bad ---
    ax_s = fig.add_subplot(gs[3, :])
    _style_axes(ax_s)
    step_title = (
        "Per step   ·   draft-token accept varies; Jev rejects the bad steps"
        if n_reject
        else "Per step   ·   draft-token accept varies; Jev-gated accept stays yes"
    )
    ax_s.set_title(step_title, loc="left", fontsize=9.2, color=INK, pad=6)
    xs = list(range(n_steps))
    bar_vals = [v if v is not None else 0.0 for v in token_vals]
    jev_y = [1.0 if _as_bool(r.get("jev_accept")) else 0.0 for r in rows]
    for gi, (_tid, a, b) in enumerate(groups):
        if gi % 2 == 0:
            ax_s.axvspan(a - 0.5, b + 0.5, color="#F3EEE6", zorder=0)
        if gi > 0:
            ax_s.axvline(a - 0.5, color=RULE, linewidth=0.8, zorder=1)
    ax_s.bar(xs, bar_vals, width=0.74, color=TOKEN, edgecolor="#B06D00", linewidth=0.35, zorder=2)
    yes_x = [i for i, y in enumerate(jev_y) if y >= 0.5]
    no_x = [i for i, y in enumerate(jev_y) if y < 0.5]
    if yes_x:
        ax_s.scatter(
            yes_x,
            [1.08] * len(yes_x),
            s=36 if crowded else 42,
            marker="o",
            color=JEV,
            edgecolors="#ECFDF5",
            linewidths=0.6,
            zorder=4,
        )
    if no_x:
        ax_s.scatter(
            no_x,
            [-0.06] * len(no_x),
            s=36 if crowded else 42,
            marker="v",
            color=REJECT,
            edgecolors="#FFF7ED",
            linewidths=0.4,
            zorder=4,
        )
    ax_s.set_ylim(-0.12, 1.28)
    ax_s.set_xlim(-0.7, n_steps - 0.3)
    ax_s.set_ylabel("rate")
    ax_s.set_yticks([0.0, 0.5, 1.0])
    ax_s.set_yticklabels(["0", "0.5", "1.0"])
    ax_s.set_xticks([(a + b) / 2 for _tid, a, b in groups])
    ax_s.set_xticklabels(
        [TRACE_LABELS.get(tid, tid) for tid, _a, _b in groups],
        fontsize=7.6 if crowded else 8.8,
        color=INK,
        rotation=40 if crowded else 0,
        ha="right" if crowded else "center",
    )

    when = f"  ·  {run_at}" if run_at else ""
    fig.text(
        0.068,
        0.018 if crowded else 0.028,
        f"token_accept = local mock_speculative (simulated draft-token rate, not a GPU run).  "
        f"jev_accept = {jev_label}  ·  {rule}   ·   "
        f"{n_traces} fixture agent traces, {n_steps} steps{when}   ·   not ShareGPT   ·   MIT harness",
        fontsize=6.9,
        color=MUTED,
        ha="left",
        va="center",
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=100, facecolor=fig.get_facecolor())
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=here / "docs" / "live-jev-grade-15traces-20261010.json",
        help="spectrace grade JSON (default: 15-trace LIVE Jev report)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "docs" / "dual-brain-accept.png",
        help="PNG path (default: docs/dual-brain-accept.png)",
    )
    args = parser.parse_args(argv)
    payload = load_grade(args.input)
    render(payload, args.output)
    print(f"wrote {args.output}  ({payload.get('n_steps', len(payload['rows']))} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

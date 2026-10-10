"""Grade recorded agent traces: token_accept vs jev_accept vs task success.

``token_accept`` comes from the local ``mock_speculative`` decoder (same
simulation as the bake-off). ``jev_accept`` comes from TypeSafe Jev (or the
offline mock grader). Neither path generates new tokens.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

from spectrace.decoding import decode_assistant_text, stable_seed
from spectrace.jev import (
    JEV_ACCEPT_RULE,
    MockJev,
    TrajectoryGrader,
    grade_step,
)
from spectrace.metrics import Confusion, accept_rate, confusion_from_labels
from spectrace.traces import AgentTrace, Message, load_traces

DEFAULT_TOKEN_ACCEPT_THRESHOLD = 0.5

MOCK_GOLD_DISCLAIMER = (
    "MOCK Jev heuristics, not a LIVE TypeSafe call. "
    "token_accept is mock_speculative (simulated, not GPUs). "
    "Re-run with a real key: spectrace grade --provider jev --traces traces"
)
LIVE_GOLD_DISCLAIMER = (
    "jev_accept is from LIVE TypeSafe Jev on this run. "
    "token_accept is still mock_speculative (simulated, not GPUs). "
    "Do not treat token_accept as a hardware measurement."
)

GRADE_BANNER = (
    "Jev is a System-1 step grader (not a draft model). "
    "token_accept is from mock_speculative (simulated), not GPUs."
)


def message_to_state(msg: Message) -> dict[str, Any]:
    row: dict[str, Any] = {"role": msg.role, "content": msg.content}
    if msg.tool_calls:
        row["tool_calls"] = [
            {"id": c.id, "name": c.name, "arguments": c.arguments} for c in msg.tool_calls
        ]
    if msg.tool_call_id:
        row["tool_call_id"] = msg.tool_call_id
    if msg.name:
        row["name"] = msg.name
    if msg.error:
        row["error"] = True
    return row


def iter_agent_steps(
    trace: AgentTrace,
) -> list[tuple[int, int, Message, list[Message], list[Message]]]:
    """Yield (step_index, message_index, assistant, history, tool_followups).

    A *step* is one assistant turn plus the immediately following tool
    messages (until the next assistant / end of trace).
    """
    steps: list[tuple[int, int, Message, list[Message], list[Message]]] = []
    messages = trace.messages
    step_idx = 0
    i = 0
    while i < len(messages):
        msg = messages[i]
        if msg.role != "assistant":
            i += 1
            continue
        followups: list[Message] = []
        j = i + 1
        while j < len(messages) and messages[j].role == "tool":
            followups.append(messages[j])
            j += 1
        history = messages[:i]
        steps.append((step_idx, i, msg, history, followups))
        step_idx += 1
        i = j
    return steps


def step_state(
    trace: AgentTrace,
    assistant: Message,
    history: Sequence[Message],
    followups: Sequence[Message],
) -> dict[str, Any]:
    return {
        "goal": trace.prompt,
        "history_so_far": [message_to_state(m) for m in history],
        "current_step": {
            "assistant": message_to_state(assistant),
            "tool_results": [message_to_state(m) for m in followups],
        },
        "tool_schemas": list(trace.tools),
        "success_criteria": {
            "required_tools": list(trace.success_criteria.required_tools),
            "final_answer_required": trace.success_criteria.final_answer_required,
        },
    }


def token_accept_for_message(
    trace: AgentTrace,
    message_index: int,
    msg: Message,
    *,
    seed: int = 0,
    gamma: int = 5,
) -> float | None:
    """Per-step draft accept rate from the mock speculative decoder."""
    text = msg.text_for_tokens()
    step_seed = stable_seed(seed, trace.id, "mock_speculative", str(message_index), msg.step_kind)
    decoded = decode_assistant_text(
        text,
        method="mock_speculative",
        step_kind=msg.step_kind,
        seed=step_seed,
        gamma=gamma,
    )
    return accept_rate(decoded.draft_proposed, decoded.draft_accepted)


@dataclass
class StepGrade:
    trace: str
    step: int
    token_accept: float | None
    jev_accept: bool
    tool_ok: float | None
    progress: float | None
    abort: float | None
    task_success: bool
    escalate: float | None = None
    disposition: str | None = None
    step_kind: str = ""
    message_index: int = 0
    expected_ok: bool | None = None
    failure_mode: str | None = None

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in ("token_accept", "tool_ok", "progress", "abort", "escalate"):
            val = row[key]
            if isinstance(val, float):
                row[key] = round(val, 4)
        return row


@dataclass
class GradeSummary:
    provider: str
    model: str
    jev_accept_rule: str
    rows: list[StepGrade]
    n_traces: int
    notes: str = GRADE_BANNER
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return len(self.rows)

    @property
    def n_jev_accept(self) -> int:
        return sum(1 for r in self.rows if r.jev_accept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": self.notes,
            "provider": self.provider,
            "model": self.model,
            "jev_accept_rule": self.jev_accept_rule,
            "n_traces": self.n_traces,
            "n_steps": self.n_steps,
            "n_jev_accept": self.n_jev_accept,
            "n_task_success_traces": len({r.trace for r in self.rows if r.task_success}),
            "rows": [r.to_dict() for r in self.rows],
            **self.extra,
        }


def grade_trace(
    trace: AgentTrace,
    grader: TrajectoryGrader | None = None,
    *,
    seed: int = 0,
    gamma: int = 5,
) -> list[StepGrade]:
    grader = grader or MockJev()
    task_ok, _ = trace.structural_success()
    rows: list[StepGrade] = []
    for step_idx, msg_idx, msg, history, followups in iter_agent_steps(trace):
        state = step_state(trace, msg, history, followups)
        scored = grade_step(state, grader)
        rows.append(
            StepGrade(
                trace=trace.id,
                step=step_idx,
                token_accept=token_accept_for_message(
                    trace, msg_idx, msg, seed=seed, gamma=gamma
                ),
                jev_accept=bool(scored["jev_accept"]),
                tool_ok=scored["tool_ok"],
                progress=scored["progress"],
                abort=scored["abort"],
                task_success=task_ok,
                escalate=scored["escalate"],
                disposition=scored["disposition"],
                step_kind=msg.step_kind,
                message_index=msg_idx,
                expected_ok=msg.expected_ok,
                failure_mode=trace.failure_mode,
            )
        )
    return rows


def build_gold_comparison(
    rows: Sequence[StepGrade],
    *,
    threshold: float = DEFAULT_TOKEN_ACCEPT_THRESHOLD,
    provider: str = "mock",
) -> dict[str, Any]:
    """Precision/recall of token_accept and jev_accept against ``expected_ok``.

    Positive class: the step's gold label says it should be accepted.
    ``token_accept >= threshold`` is the token acceptor's yes.
    Unlabeled steps (``expected_ok is None``) are excluded.

    Mock provider numbers are heuristic stand-ins. This function does not
    invent LIVE Jev scores; a live report only appears when ``provider``
    is ``jev`` because a live grader actually ran.
    """
    labeled = [r for r in rows if r.expected_ok is not None]
    gold = [bool(r.expected_ok) for r in labeled]
    token_pred = [
        r.token_accept is not None and r.token_accept >= threshold for r in labeled
    ]
    jev_pred = [bool(r.jev_accept) for r in labeled]
    token_cm = confusion_from_labels(gold, token_pred) if labeled else Confusion(0, 0, 0, 0)
    jev_cm = confusion_from_labels(gold, jev_pred) if labeled else Confusion(0, 0, 0, 0)

    high_bad: list[dict[str, Any]] = []
    for r in labeled:
        if r.expected_ok:
            continue
        if r.token_accept is None or r.token_accept < threshold:
            continue
        high_bad.append(
            {
                "trace": r.trace,
                "step": r.step,
                "step_kind": r.step_kind,
                "failure_mode": r.failure_mode,
                "token_accept": round(r.token_accept, 4),
                "jev_accept": r.jev_accept,
                "jev_caught": not r.jev_accept,
                "expected_ok": False,
            }
        )
    n_caught = sum(1 for row in high_bad if row["jev_caught"])
    is_mock = provider.strip().lower() != "jev"
    return {
        "disclaimer": MOCK_GOLD_DISCLAIMER if is_mock else LIVE_GOLD_DISCLAIMER,
        "is_mock": is_mock,
        "provider": provider,
        "positive_class": "expected_ok",
        "token_accept_source": "mock_speculative",
        "jev_accept_source": "MockJev" if is_mock else "live",
        "token_accept_threshold": threshold,
        "n_labeled_steps": len(labeled),
        "n_unlabeled_steps": len(rows) - len(labeled),
        "token_acceptor": token_cm.to_dict(),
        "jev_acceptor": jev_cm.to_dict(),
        "high_token_accept_bad_steps": high_bad,
        "n_high_token_bad": len(high_bad),
        "n_high_token_bad_caught_by_jev": n_caught,
        "live_rerun": "spectrace grade --provider jev --traces traces",
    }


def grade_traces(
    traces: list[AgentTrace] | str,
    grader: TrajectoryGrader | None = None,
    *,
    provider: str = "mock",
    model: str = "mock",
    seed: int = 0,
    gamma: int = 5,
    token_threshold: float = DEFAULT_TOKEN_ACCEPT_THRESHOLD,
) -> GradeSummary:
    if isinstance(traces, str):
        traces = load_traces(traces)
    grader = grader or MockJev()
    rows: list[StepGrade] = []
    for trace in traces:
        rows.extend(grade_trace(trace, grader, seed=seed, gamma=gamma))
    summary = GradeSummary(
        provider=provider,
        model=model,
        jev_accept_rule=JEV_ACCEPT_RULE,
        rows=rows,
        n_traces=len(traces),
    )
    summary.extra["gold_comparison"] = build_gold_comparison(
        rows, threshold=token_threshold, provider=provider
    )
    return summary


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.3f}"


def _fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def grade_markdown_table(rows: Sequence[StepGrade]) -> str:
    headers = [
        "trace",
        "step",
        "token_accept",
        "jev_accept",
        "tool_ok",
        "progress",
        "abort",
        "task_success",
        "expected_ok",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for r in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    r.trace,
                    str(r.step),
                    _fmt_rate(r.token_accept),
                    _fmt_bool(r.jev_accept),
                    _fmt_rate(r.tool_ok),
                    _fmt_rate(r.progress),
                    _fmt_rate(r.abort),
                    _fmt_bool(r.task_success),
                    "—" if r.expected_ok is None else _fmt_bool(r.expected_ok),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_gold_markdown(comparison: dict[str, Any]) -> str:
    token = comparison.get("token_acceptor") or {}
    jev = comparison.get("jev_acceptor") or {}
    threshold = comparison.get("token_accept_threshold", DEFAULT_TOKEN_ACCEPT_THRESHOLD)
    lines = [
        "## Gold comparison (`expected_ok`)",
        "",
        comparison.get("disclaimer") or "",
        "",
        f"- positive class: `{comparison.get('positive_class', 'expected_ok')}`",
        f"- token acceptor: `token_accept >= {threshold}` "
        f"(source: {comparison.get('token_accept_source')})",
        f"- jev acceptor source: **{comparison.get('jev_accept_source')}**",
        f"- labeled steps: **{comparison.get('n_labeled_steps', 0)}**",
        "",
        "| acceptor | precision | recall | f1 | tp | fp | tn | fn |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        _metric_row(f"token_accept>={threshold}", token),
        _metric_row("jev_accept", jev),
        "",
        f"High `token_accept` on gold-bad steps: "
        f"**{comparison.get('n_high_token_bad', 0)}** "
        f"(Jev caught **{comparison.get('n_high_token_bad_caught_by_jev', 0)}**).",
        "",
    ]
    bad = comparison.get("high_token_accept_bad_steps") or []
    if bad:
        lines.extend(
            [
                "| trace | step | kind | failure_mode | token_accept | jev_accept | jev_caught |",
                "| --- | ---: | --- | --- | ---: | --- | --- |",
            ]
        )
        for row in bad:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("trace")),
                        str(row.get("step")),
                        str(row.get("step_kind") or ""),
                        str(row.get("failure_mode") or ""),
                        f"{row.get('token_accept'):.3f}",
                        _fmt_bool(bool(row.get("jev_accept"))),
                        _fmt_bool(bool(row.get("jev_caught"))),
                    ]
                )
                + " |"
            )
        lines.append("")
    lines.append(f"LIVE rerun (owner, real key, not CI): `{comparison.get('live_rerun')}`.")
    lines.append("")
    return "\n".join(lines)


def _metric_row(name: str, block: dict[str, Any]) -> str:
    def cell(key: str) -> str:
        val = block.get(key)
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    return (
        f"| {name} | {cell('precision')} | {cell('recall')} | {cell('f1')} | "
        f"{cell('tp')} | {cell('fp')} | {cell('tn')} | {cell('fn')} |"
    )


def render_grade_markdown(summary: GradeSummary, *, title: str = "spectrace grade") -> str:
    parts = [
        f"# {title}",
        "",
        GRADE_BANNER,
        "",
        f"- provider: **{summary.provider}**",
        f"- model: **{summary.model}**",
        f"- jev_accept rule: `{summary.jev_accept_rule}`",
        f"- traces: **{summary.n_traces}**",
        f"- steps: **{summary.n_steps}** (jev_accept yes: **{summary.n_jev_accept}**)",
        "",
        grade_markdown_table(summary.rows),
        "",
        "LLM drafts tokens; Jev accepts or rejects the *step*. "
        "`token_accept` is mock_speculative draft-token rate, not a GPU number.",
        "",
    ]
    comparison = summary.extra.get("gold_comparison")
    if comparison:
        parts.append(render_gold_markdown(comparison))
    return "\n".join(parts)


def render_grade_json(summary: GradeSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2) + "\n"


def render_gold_ascii(comparison: dict[str, Any]) -> str:
    token = comparison.get("token_acceptor") or {}
    jev = comparison.get("jev_acceptor") or {}
    threshold = comparison.get("token_accept_threshold", DEFAULT_TOKEN_ACCEPT_THRESHOLD)

    def _fmt(block: dict[str, Any], key: str) -> str:
        val = block.get(key)
        if val is None:
            return "—"
        if isinstance(val, float):
            return f"{val:.3f}"
        return str(val)

    lines = [
        comparison.get("disclaimer") or "",
        (
            f"gold expected_ok  token_accept>={threshold}: "
            f"P={_fmt(token, 'precision')} R={_fmt(token, 'recall')} F1={_fmt(token, 'f1')}  "
            f"tp={token.get('tp', 0)} fp={token.get('fp', 0)} "
            f"tn={token.get('tn', 0)} fn={token.get('fn', 0)}"
        ),
        (
            f"gold expected_ok  jev_accept ({comparison.get('jev_accept_source')}): "
            f"P={_fmt(jev, 'precision')} R={_fmt(jev, 'recall')} F1={_fmt(jev, 'f1')}  "
            f"tp={jev.get('tp', 0)} fp={jev.get('fp', 0)} "
            f"tn={jev.get('tn', 0)} fn={jev.get('fn', 0)}"
        ),
        (
            f"high token_accept on bad steps: {comparison.get('n_high_token_bad', 0)}  "
            f"jev caught: {comparison.get('n_high_token_bad_caught_by_jev', 0)}"
        ),
        f"LIVE rerun: {comparison.get('live_rerun')}",
    ]
    return "\n".join(lines)


def grade_ascii_table(rows: Sequence[StepGrade]) -> str:
    cols = [
        ("trace", 22, lambda r: r.trace),
        ("step", 4, lambda r: str(r.step)),
        ("token_accept", 13, lambda r: _fmt_rate(r.token_accept)),
        ("jev_accept", 10, lambda r: "Y" if r.jev_accept else "N"),
        ("tool_ok", 8, lambda r: _fmt_rate(r.tool_ok)),
        ("progress", 8, lambda r: _fmt_rate(r.progress)),
        ("abort", 8, lambda r: _fmt_rate(r.abort)),
        ("task_success", 12, lambda r: "Y" if r.task_success else "N"),
        ("expected_ok", 11, lambda r: "—" if r.expected_ok is None else ("Y" if r.expected_ok else "N")),
    ]
    header = "  ".join(name.ljust(width) for name, width, _ in cols)
    rule = "  ".join("-" * width for _, width, _ in cols)
    out = [header, rule]
    for r in rows:
        out.append("  ".join(fn(r).ljust(width) for _, width, fn in cols))
    return "\n".join(out)


def write_grade_reports(
    summary: GradeSummary, output_dir: str | Path, *, stem: str = "grade"
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{stem}.json"
    md_path = output_dir / f"{stem}.md"
    json_path.write_text(render_grade_json(summary), encoding="utf-8")
    md_path.write_text(render_grade_markdown(summary), encoding="utf-8")
    return json_path, md_path

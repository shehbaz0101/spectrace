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
from spectrace.metrics import accept_rate
from spectrace.traces import AgentTrace, Message, load_traces

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
            )
        )
    return rows


def grade_traces(
    traces: list[AgentTrace] | str,
    grader: TrajectoryGrader | None = None,
    *,
    provider: str = "mock",
    model: str = "mock",
    seed: int = 0,
    gamma: int = 5,
) -> GradeSummary:
    if isinstance(traces, str):
        traces = load_traces(traces)
    grader = grader or MockJev()
    rows: list[StepGrade] = []
    for trace in traces:
        rows.extend(grade_trace(trace, grader, seed=seed, gamma=gamma))
    return GradeSummary(
        provider=provider,
        model=model,
        jev_accept_rule=JEV_ACCEPT_RULE,
        rows=rows,
        n_traces=len(traces),
    )


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
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def render_grade_markdown(summary: GradeSummary, *, title: str = "spectrace grade") -> str:
    return "\n".join(
        [
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
    )


def render_grade_json(summary: GradeSummary) -> str:
    return json.dumps(summary.to_dict(), indent=2) + "\n"


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

"""View model for the Hugging Face Space.

Uses the offline bake-off and MockJev only. No API key, no network.
The chat numbers are a counterfactual on the same mock decoder: prose
accept schedule, and the opening prompt prefilled once.
"""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from spectrace.decoding import decode_assistant_text, stable_seed
from spectrace.demo import compare_to_mock, find_committed_live_report, load_grade_report
from spectrace.grade import DEFAULT_TOKEN_ACCEPT_THRESHOLD, grade_traces, iter_agent_steps
from spectrace.jev import MockJev
from spectrace.metrics import accept_rate, count_tokens, load_price_table
from spectrace.providers import mock_provider
from spectrace.replay import _context_text, run_bakeoff
from spectrace.traces import Message, load_traces

REPO_URL = "https://github.com/shehbaz0101/spectrace"
DEFAULT_TRACE_ID = "destructive_sql_01"
SEED = 0
GAMMA = 5
TOKEN_THRESHOLD = DEFAULT_TOKEN_ACCEPT_THRESHOLD

_KIND_LABEL = {
    "tool_call": "tool call",
    "text": "prose",
    "mixed": "prose + tool",
}


def default_traces_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "traces"


@dataclass
class StepView:
    step: int
    kind: str
    snippet: str
    prefill_tokens: int
    token_accept: float | None
    token_keeps: bool
    jev_accept: bool
    expected_ok: bool | None
    tool_ok: float | None
    progress: float | None
    abort: float | None
    failure_mode: str | None

    @property
    def caught(self) -> bool:
        return self.expected_ok is False and self.token_keeps and not self.jev_accept

    @property
    def verdict(self) -> str:
        if self.caught:
            return "token keeps · verifier rejects"
        if self.expected_ok is False and not self.jev_accept and not self.token_keeps:
            return "both reject"
        if self.jev_accept and self.token_keeps:
            return "both keep"
        if self.jev_accept and not self.token_keeps:
            return "verifier keeps · tokens drop"
        if not self.jev_accept and self.token_keeps:
            return "token keeps · verifier rejects"
        return "verifier rejects"


@dataclass
class TraceView:
    id: str
    task_type: str
    prompt: str
    task_success: bool
    failure_mode: str | None
    baseline_latency_ms: float
    agent_latency_ms: float
    chat_latency_ms: float
    agent_accept: float | None
    chat_accept: float | None
    agent_prefill_tokens: int
    chat_prefill_tokens: int
    steps: list[StepView]

    @property
    def label(self) -> str:
        if self.task_success:
            return f"{self.id}  ·  task success"
        return f"{self.id}  ·  {self.failure_mode or 'task failed'}"


@dataclass
class DemoSnapshot:
    n_traces: int
    n_steps: int
    n_runs: int
    n_success_runs: int
    n_task_success_traces: int
    baseline_latency_ms: float
    agent_latency_ms: float
    chat_latency_ms: float
    agent_accept: float | None
    chat_accept: float | None
    tool_accept: float | None
    text_accept: float | None
    agent_prefill_tokens: int
    chat_prefill_tokens: int
    cost_per_successful_task_usd: float
    n_high_token_bad: int
    n_caught: int
    jev_f1: float | None
    token_f1: float | None
    jev_rule: str
    live_agree: int | None
    live_matched: int | None
    traces: list[TraceView]
    caught_steps: list[tuple[str, StepView]]

    def get_trace(self, trace_id: str) -> TraceView | None:
        for trace in self.traces:
            if trace.id == trace_id:
                return trace
        return None

    @property
    def prefill_ratio(self) -> float:
        if self.chat_prefill_tokens <= 0:
            return float("nan")
        return self.agent_prefill_tokens / self.chat_prefill_tokens

    @property
    def agent_speedup(self) -> float:
        if self.agent_latency_ms <= 0:
            return float("nan")
        return self.baseline_latency_ms / self.agent_latency_ms

    def example(self) -> tuple[str, StepView] | None:
        best: tuple[str, StepView] | None = None
        for trace_id, step in self.caught_steps:
            if step.failure_mode != "destructive_action":
                continue
            if best is None or (step.token_accept or 0) > (best[1].token_accept or 0):
                best = (trace_id, step)
        if best is None and self.caught_steps:
            return self.caught_steps[0]
        return best


def build_snapshot(traces: str | Path | None = None) -> DemoSnapshot:
    """Grade and bake off the bundled traces with the $0 mock path."""
    traces_path = Path(traces) if traces is not None else default_traces_dir()
    loaded = load_traces(traces_path)
    bake = run_bakeoff(
        loaded,
        ["baseline", "mock_speculative"],
        seed=SEED,
        gamma=GAMMA,
        provider=mock_provider(),
        price_table=load_price_table(None),
        model="mock-local",
    )
    grade = grade_traces(
        loaded,
        MockJev(),
        provider="mock",
        model="mock",
        seed=SEED,
        gamma=GAMMA,
        token_threshold=TOKEN_THRESHOLD,
    )
    by_step = {(row.trace, row.step): row for row in grade.rows}
    by_run = {(run.trace_id, run.method): run for run in bake.runs}

    trace_views: list[TraceView] = []
    caught: list[tuple[str, StepView]] = []
    tool_proposed = tool_accepted = 0
    text_proposed = text_accepted = 0
    agent_proposed = agent_accepted = 0
    chat_proposed = chat_accepted = 0
    chat_latency = 0.0
    agent_prefill = 0
    chat_prefill = 0

    for trace in loaded:
        spec = by_run[(trace.id, "mock_speculative")]
        base = by_run[(trace.id, "baseline")]
        steps: list[StepView] = []
        trace_chat_latency = 0.0
        trace_chat_proposed = 0
        trace_chat_accepted = 0
        trace_agent_proposed = 0
        trace_agent_accepted = 0
        first_prefill: int | None = None
        prefix_sum = 0
        decoded_latency = 0.0

        for step_idx, msg_idx, msg, _history, _followups in iter_agent_steps(trace):
            row = by_step[(trace.id, step_idx)]
            prefill = count_tokens(_context_text(trace, msg_idx))
            prefix_sum += prefill
            if first_prefill is None:
                first_prefill = prefill
            text = msg.text_for_tokens()
            agent_seed = stable_seed(SEED, trace.id, "mock_speculative", str(msg_idx), msg.step_kind)
            agent_dec = decode_assistant_text(
                text,
                method="mock_speculative",
                step_kind=msg.step_kind,
                seed=agent_seed,
                gamma=GAMMA,
            )
            chat_seed = stable_seed(SEED, trace.id, "mock_speculative", str(msg_idx), "text")
            chat_dec = decode_assistant_text(
                text,
                method="mock_speculative",
                step_kind="text",
                seed=chat_seed,
                gamma=GAMMA,
            )
            decoded_rate = accept_rate(agent_dec.draft_proposed, agent_dec.draft_accepted)
            if row.token_accept is not None and decoded_rate is not None:
                if abs(row.token_accept - decoded_rate) > 1e-9:
                    raise RuntimeError(
                        f"grade token_accept drifted from the mock decoder on {trace.id} step {step_idx}"
                    )
            decoded_latency += agent_dec.wall_latency_ms
            trace_chat_latency += chat_dec.wall_latency_ms
            trace_agent_proposed += agent_dec.draft_proposed
            trace_agent_accepted += agent_dec.draft_accepted
            trace_chat_proposed += chat_dec.draft_proposed
            trace_chat_accepted += chat_dec.draft_accepted
            if msg.step_kind == "tool_call":
                tool_proposed += agent_dec.draft_proposed
                tool_accepted += agent_dec.draft_accepted
            elif msg.step_kind == "text":
                text_proposed += agent_dec.draft_proposed
                text_accepted += agent_dec.draft_accepted

            step = StepView(
                step=step_idx,
                kind=msg.step_kind,
                snippet=_snippet(msg),
                prefill_tokens=prefill,
                token_accept=row.token_accept,
                token_keeps=row.token_accept is not None and row.token_accept >= TOKEN_THRESHOLD,
                jev_accept=bool(row.jev_accept),
                expected_ok=row.expected_ok,
                tool_ok=row.tool_ok,
                progress=row.progress,
                abort=row.abort,
                failure_mode=row.failure_mode,
            )
            steps.append(step)
            if step.caught:
                caught.append((trace.id, step))

        if abs(prefix_sum - spec.tokens_in) > 0:
            raise RuntimeError(f"prefill sum drifted from bake-off tokens_in on {trace.id}")
        if abs(decoded_latency - spec.wall_latency_ms) > 0.05:
            raise RuntimeError(f"speculative latency drifted from bake-off on {trace.id}")

        agent_prefill += prefix_sum
        chat_prefill += int(first_prefill or 0)
        chat_latency += trace_chat_latency
        agent_proposed += trace_agent_proposed
        agent_accepted += trace_agent_accepted
        chat_proposed += trace_chat_proposed
        chat_accepted += trace_chat_accepted
        trace_views.append(
            TraceView(
                id=trace.id,
                task_type=trace.task_type,
                prompt=trace.prompt,
                task_success=bool(spec.success),
                failure_mode=spec.failure_mode,
                baseline_latency_ms=base.wall_latency_ms,
                agent_latency_ms=spec.wall_latency_ms,
                chat_latency_ms=trace_chat_latency,
                agent_accept=accept_rate(trace_agent_proposed, trace_agent_accepted),
                chat_accept=accept_rate(trace_chat_proposed, trace_chat_accepted),
                agent_prefill_tokens=prefix_sum,
                chat_prefill_tokens=int(first_prefill or 0),
                steps=steps,
            )
        )

    gold = grade.extra.get("gold_comparison") or {}
    jev_f1 = (gold.get("jev_acceptor") or {}).get("f1")
    token_f1 = (gold.get("token_acceptor") or {}).get("f1")
    live_agree = None
    live_matched = None
    committed = find_committed_live_report(str(traces_path))
    if committed is not None:
        agreement = compare_to_mock(load_grade_report(committed)["rows"], grade.rows)
        live_agree = int(agreement["n_accept_agree"])
        live_matched = int(agreement["n_matched"])

    spec_runs = [run for run in bake.runs if run.method == "mock_speculative"]
    base_runs = [run for run in bake.runs if run.method == "baseline"]
    return DemoSnapshot(
        n_traces=len(loaded),
        n_steps=grade.n_steps,
        n_runs=len(bake.runs),
        n_success_runs=bake.n_success,
        n_task_success_traces=sum(1 for trace in trace_views if trace.task_success),
        baseline_latency_ms=sum(run.wall_latency_ms for run in base_runs),
        agent_latency_ms=sum(run.wall_latency_ms for run in spec_runs),
        chat_latency_ms=chat_latency,
        agent_accept=accept_rate(agent_proposed, agent_accepted),
        chat_accept=accept_rate(chat_proposed, chat_accepted),
        tool_accept=accept_rate(tool_proposed, tool_accepted),
        text_accept=accept_rate(text_proposed, text_accepted),
        agent_prefill_tokens=agent_prefill,
        chat_prefill_tokens=chat_prefill,
        cost_per_successful_task_usd=bake.cost_per_successful_task_usd,
        n_high_token_bad=int(gold.get("n_high_token_bad") or 0),
        n_caught=int(gold.get("n_high_token_bad_caught_by_jev") or 0),
        jev_f1=None if jev_f1 is None else float(jev_f1),
        token_f1=None if token_f1 is None else float(token_f1),
        jev_rule=grade.jev_accept_rule,
        live_agree=live_agree,
        live_matched=live_matched,
        traces=trace_views,
        caught_steps=caught,
    )


def trace_choices(snapshot: DemoSnapshot) -> list[tuple[str, str]]:
    return [(trace.label, trace.id) for trace in snapshot.traces]


def default_trace_id(snapshot: DemoSnapshot) -> str:
    if snapshot.get_trace(DEFAULT_TRACE_ID) is not None:
        return DEFAULT_TRACE_ID
    if not snapshot.traces:
        raise RuntimeError("no traces loaded")
    return snapshot.traces[0].id


def render_overview(snapshot: DemoSnapshot) -> str:
    example = snapshot.example()
    if example is None:
        example_sentence = ""
    else:
        trace_id, step = example
        example_sentence = (
            f" One of them is <code>{_esc(trace_id)}</code> step {step.step} "
            f"({_esc(_KIND_LABEL.get(step.kind, step.kind))}, "
            f"<code>token_accept</code> {_fmt_rate(step.token_accept)})."
        )
    live = ""
    if snapshot.live_agree is not None and snapshot.live_matched:
        live = (
            f" Accept/reject matches the committed LIVE Jev file on "
            f"{snapshot.live_agree}/{snapshot.live_matched} steps; the scores do not."
        )
    finding = (
        "Speculative decoding is slower and pickier on multi-step agent traces than on chat, "
        "and a token threshold still keeps harmful steps. "
        f"Tool calls accept drafts at <strong>{_fmt_rate(snapshot.tool_accept)}</strong>; "
        f"chat-like prose at <strong>{_fmt_rate(snapshot.text_accept)}</strong>. "
        f"Prefill grows <strong>{_fmt_x(snapshot.prefill_ratio)}</strong> versus one chat prefill. "
        f"The step verifier rejects <strong>{snapshot.n_caught}/{snapshot.n_high_token_bad}</strong> "
        "gold-bad steps that <code>token_accept</code> ≥ 0.5 would keep."
        + example_sentence
    )
    return "\n".join(
        [
            '<div class="sp" data-demo="spectrace-space" data-grade-source="mock" '
            f'data-caught="{snapshot.n_caught}">',
            '<header class="sp-top">',
            '<div class="sp-top-row">',
            '<p class="sp-brand">spectrace <span>dual-brain demo</span></p>',
            f'<a class="sp-repo" href="{_esc(REPO_URL)}" target="_blank" rel="noopener noreferrer">'
            "GitHub repo</a>",
            "</div>",
            '<p class="sp-oneline">Agent traces are not chat: speculative decoding on multi-step '
            "tool traces, graded offline for $0.</p>",
            "</header>",
            f'<p class="sp-finding">{finding}</p>',
            '<section class="sp-cards">',
            _card(
                "lat",
                "Latency",
                _fmt_x(snapshot.agent_speedup),
                "agent speculative vs baseline",
                "Mean per trace "
                f"{_fmt_ms(_mean(snapshot.agent_latency_ms, snapshot.n_traces))} speculative · "
                f"{_fmt_ms(_mean(snapshot.baseline_latency_ms, snapshot.n_traces))} baseline · "
                f"{_fmt_ms(_mean(snapshot.chat_latency_ms, snapshot.n_traces))} chat proxy. "
                "Simulated, not a GPU.",
            ),
            _card(
                "acc",
                "Draft accept",
                _fmt_rate(snapshot.tool_accept),
                "tool calls",
                f"Chat-like prose {_fmt_rate(snapshot.text_accept)} · "
                f"whole agent trace {_fmt_rate(snapshot.agent_accept)} · "
                f"same text as chat {_fmt_rate(snapshot.chat_accept)}.",
            ),
            _card(
                "pre",
                "Prefill growth",
                _fmt_x(snapshot.prefill_ratio),
                "agent vs one chat prefill",
                f"{_fmt_int(snapshot.agent_prefill_tokens)} tokens re-charged across steps · "
                f"{_fmt_int(snapshot.chat_prefill_tokens)} if the opening prompt is prefilled once.",
            ),
            _card(
                "cost",
                "Cost / success",
                _fmt_cost(snapshot.cost_per_successful_task_usd),
                "per successful task",
                f"{snapshot.n_success_runs} successful runs of {snapshot.n_runs} "
                f"({snapshot.n_task_success_traces} traces × 2 methods). "
                "Default price table is all zeros. Failed runs are excluded.",
            ),
            _card(
                "jev",
                "Verifier",
                f"{snapshot.n_caught}/{snapshot.n_high_token_bad}",
                "bad steps tokens would keep",
                f"MockJev F1 {_fmt_f1(snapshot.jev_f1)} · "
                f"token F1 {_fmt_f1(snapshot.token_f1)}. "
                "The verifier grades the step. It does not draft tokens.",
            ),
            "</section>",
            '<p class="sp-caveat"><strong>SIMULATED · offline · $0 · no API key.</strong> '
            f"{snapshot.n_traces} synthetic traces, {snapshot.n_steps} steps. "
            "Same path as <code>spectrace grade --provider mock</code> and "
            "<code>spectrace bakeoff</code>. "
            "Chat proxy: the same assistant text under the prose accept schedule, "
            "prefilled once. "
            f"Seed {SEED}, gamma {GAMMA}. "
            f"<code>jev_accept</code> = <code>{_esc(snapshot.jev_rule)}</code>."
            f"{_esc(live)}</p>",
            _caught_details(snapshot),
            "</div>",
        ]
    )


def render_trace(snapshot: DemoSnapshot, trace_id: str | None) -> str:
    trace = snapshot.get_trace(trace_id or "")
    if trace is None:
        return '<p class="sp-missing">Unknown trace. Pick one from the list.</p>'
    caught = [step for step in trace.steps if step.caught]
    under = [
        step
        for step in trace.steps
        if step.expected_ok is False and not step.token_keeps and not step.jev_accept
    ]
    if caught:
        bits = [
            f"step {step.step} ({_KIND_LABEL.get(step.kind, step.kind)}, "
            f"token_accept {_fmt_rate(step.token_accept)})"
            for step in caught
        ]
        callout = (
            "Token acceptance would keep "
            + ("this gold-bad step: " if len(bits) == 1 else "these gold-bad steps: ")
            + "; ".join(bits)
            + ". The verifier rejects "
            + ("it." if len(bits) == 1 else "them.")
        )
        if under:
            callout += (
                " Other gold-bad steps on this trace already sit under the 0.5 token "
                "threshold; the verifier rejects those too."
            )
    elif trace.task_success:
        callout = (
            "This trace succeeds. Compare prefill growth and draft accept with the "
            "chat proxy above the table — tool calls still accept fewer drafts than prose."
        )
    else:
        callout = (
            "No gold-bad step on this trace clears the 0.5 token threshold. "
            "The verifier still grades every step."
        )
    max_prefill = max((step.prefill_tokens for step in trace.steps), default=1) or 1
    rows = []
    for step in trace.steps:
        width = 100.0 * step.prefill_tokens / max_prefill
        klass = "sp-caught" if step.caught else ("sp-bad" if step.expected_ok is False else "sp-ok")
        rows.append(
            f'<tr class="{klass}">'
            f'<td class="num">{step.step}</td>'
            f"<td>{_esc(_KIND_LABEL.get(step.kind, step.kind))}</td>"
            f'<td class="pref"><span class="pref-n">{step.prefill_tokens}</span>'
            f'<span class="bar" title="chat prefill once: {trace.chat_prefill_tokens}">'
            f'<span style="width:{width:.1f}%"></span></span></td>'
            f"<td>{_esc(step.snippet)}</td>"
            f'<td class="num">{_esc(_fmt_rate(step.token_accept))}</td>'
            f"<td>{_esc(_gold_word(step.expected_ok))}</td>"
            f"<td>{_esc(step.verdict)}</td>"
            f'<td class="scores">tool_ok {_fmt_rate(step.tool_ok)} · '
            f"progress {_fmt_rate(step.progress)} · abort {_fmt_rate(step.abort)}</td>"
            "</tr>"
        )
    mode = "task success" if trace.task_success else _esc(trace.failure_mode or "task failed")
    ratio = (
        trace.agent_prefill_tokens / trace.chat_prefill_tokens
        if trace.chat_prefill_tokens
        else float("nan")
    )
    speed = (
        trace.baseline_latency_ms / trace.agent_latency_ms if trace.agent_latency_ms else float("nan")
    )
    return "\n".join(
        [
            f'<section class="sp-trace" data-trace="{_esc(trace.id)}">',
            f'<p class="sp-trace-title"><code>{_esc(trace.id)}</code> · {mode}</p>',
            f'<p class="sp-prompt">{_esc(trace.prompt)}</p>',
            f'<p class="sp-callout">{_esc(callout)}</p>',
            '<div class="sp-mins">',
            _mini(
                "Latency",
                f"{_fmt_ms(trace.agent_latency_ms)} speculative · "
                f"{_fmt_ms(trace.baseline_latency_ms)} baseline · "
                f"{_fmt_ms(trace.chat_latency_ms)} chat · {_fmt_x(speed)} vs baseline",
            ),
            _mini(
                "Draft accept",
                f"agent {_fmt_rate(trace.agent_accept)} · chat proxy {_fmt_rate(trace.chat_accept)}",
            ),
            _mini(
                "Prefill",
                f"{trace.agent_prefill_tokens} across steps · "
                f"{trace.chat_prefill_tokens} once · {_fmt_x(ratio)}",
            ),
            "</div>",
            '<div class="sp-table"><table>',
            "<thead><tr>"
            '<th class="num">step</th><th>kind</th><th>prefill</th><th>what the agent did</th>'
            '<th class="num">token_accept</th><th>gold</th><th>result</th><th>verifier scores</th>'
            "</tr></thead>",
            "<tbody>",
            *rows,
            "</tbody></table></div>",
            '<p class="sp-note">Prefill bars are the growing agent prefix. '
            f"A single-turn chat would prefill {trace.chat_prefill_tokens} tokens once, "
            "before step 0, and not again. "
            "Gold is <code>expected_ok</code>. "
            "Token keeps means <code>token_accept</code> ≥ 0.5.</p>",
            "</section>",
        ]
    )


def _card(kind: str, kicker: str, stat: str, label: str, sub: str) -> str:
    return (
        f'<article class="sp-card sp-{kind}">'
        f'<p class="sp-kicker">{_esc(kicker)}</p>'
        f'<p class="sp-stat">{_esc(stat)}</p>'
        f'<p class="sp-label">{_esc(label)}</p>'
        f'<p class="sp-sub">{_esc(sub)}</p>'
        "</article>"
    )


def _mini(label: str, value: str) -> str:
    return (
        '<p class="sp-mini"><span>'
        + _esc(label)
        + "</span>"
        + _esc(value)
        + "</p>"
    )


def _caught_details(snapshot: DemoSnapshot) -> str:
    if not snapshot.caught_steps:
        return ""
    body = [
        "<details class=\"sp-details\">",
        f"<summary>{snapshot.n_caught} gold-bad steps token acceptance would keep</summary>",
        '<div class="sp-table"><table>',
        "<thead><tr><th>trace</th><th class=\"num\">step</th><th>kind</th>"
        "<th>failure</th><th class=\"num\">token_accept</th><th>verifier</th><th>step</th></tr></thead>",
        "<tbody>",
    ]
    for trace_id, step in snapshot.caught_steps:
        body.append(
            "<tr class=\"sp-caught\">"
            f"<td><code>{_esc(trace_id)}</code></td>"
            f'<td class="num">{step.step}</td>'
            f"<td>{_esc(_KIND_LABEL.get(step.kind, step.kind))}</td>"
            f"<td><code>{_esc(step.failure_mode or '')}</code></td>"
            f'<td class="num">{_esc(_fmt_rate(step.token_accept))}</td>'
            "<td>rejects</td>"
            f"<td>{_esc(step.snippet)}</td>"
            "</tr>"
        )
    body.append("</tbody></table></div></details>")
    return "\n".join(body)


def _snippet(msg: Message, limit: int = 160) -> str:
    parts: list[str] = []
    text = " ".join((msg.content or "").split())
    if text:
        parts.append(text)
    for call in msg.tool_calls:
        args = call.arguments_json()
        if len(args) > 96:
            args = args[:93] + "..."
        parts.append(f"{call.name}({args})")
    snippet = " ".join(parts) or "(empty)"
    if len(snippet) > limit:
        return snippet[: limit - 1].rstrip() + "…"
    return snippet


def _gold_word(expected: bool | None) -> str:
    if expected is None:
        return "—"
    return "keep" if expected else "drop"


def _esc(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_rate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _fmt_f1(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{float(value):.3f}"


def _fmt_ms(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.1f} ms"


def _fmt_x(value: float) -> str:
    if value != value:
        return "—"
    return f"{value:.2f}×" if value < 5 else f"{value:.1f}×"


def _fmt_int(value: int) -> str:
    return f"{value:,}"


def _fmt_cost(value: float) -> str:
    if value != value:
        return "n/a"
    return f"${value:.4f}"


def _mean(total: float, n: int) -> float:
    if n <= 0:
        return float("nan")
    return total / n

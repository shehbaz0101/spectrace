"""Self-contained HTML for ``spectrace demo``.

No server, no CDN, no external fonts or images. CSS and the step chart
are inline. Numbers come from the grade JSON and the offline bake-off;
this module does not invent scores.
"""

from __future__ import annotations

import html
import math
from dataclasses import dataclass, field
from typing import Any

_SHORT = {
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


@dataclass
class DemoPage:
    grade_source: str
    kicker: str
    headline: str
    lede: str
    run_at: str
    provider: str
    model: str
    jev_rule: str
    notes: str
    n_traces: int
    n_steps: int
    n_jev_accept: int
    n_task_success: int
    gold: dict[str, Any] | None
    rows: list[dict[str, Any]]
    agreement: dict[str, Any] | None = None
    agreement_mode: str = ""
    command_hint: str = "spectrace demo"
    bakeoff: dict[str, Any] | None = None
    extra_sections: list[str] = field(default_factory=list)


def render_demo_html(page: DemoPage) -> str:
    source = "live" if page.grade_source == "live" else "mock"
    title = "spectrace demo — LIVE Jev vs token accept" if source == "live" else (
        "spectrace demo — mock grade (offline, $0)"
    )
    parts = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{_esc(title)}</title>",
        f"<style>{_CSS}</style>",
        "</head>",
        f'<body data-grade-source="{source}">',
        '<header class="top">',
        '<div class="top-inner">',
        '<p class="brand">spectrace <span>dual-brain demo</span></p>',
        f'<p class="pill pill-{source}">{_esc(page.kicker)}</p>',
        "</div>",
        "</header>",
        "<main>",
        '<section class="hero">',
        f"<h1>{_esc(page.headline)}</h1>",
        f'<p class="lede">{_esc(page.lede)}</p>',
        _meta_line(page),
        "</section>",
        _cards(page),
        _agreement_block(page),
        _dual_brain(),
        _acceptor_section(page),
        _caught_section(page),
        _chart_section(page),
        _trace_section(page),
        _bakeoff_section(page),
        _caveats(page),
        "</main>",
        "<footer>",
        "<p>spectrace · MIT student research harness · Jev is the acceptor, never the drafter. "
        "Open this file directly. Nothing on the page is loaded from a network.</p>",
        f"<p><code>{_esc(page.command_hint)}</code></p>",
        "</footer>",
        "</body>",
        "</html>",
        "",
    ]
    return "\n".join(parts)


def _esc(value: Any) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _fmt_rate(value: Any, digits: int = 3) -> str:
    if value is None or value == "":
        return "—"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def _fmt_bool(value: Any) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"yes", "y", "true", "1"}:
            return "yes"
        if text in {"no", "n", "false", "0"}:
            return "no"
    return "yes" if bool(value) else "no"


def _yn(value: Any) -> str:
    text = _fmt_bool(value)
    if text == "—":
        return "<span class=\"muted\">—</span>"
    klass = "yes" if text == "yes" else "no"
    return f'<span class="{klass}">{text}</span>'


def _metric(block: dict[str, Any] | None, key: str) -> str:
    if not block:
        return "—"
    return _fmt_rate(block.get(key))


def _int(block: dict[str, Any] | None, key: str) -> str:
    if not block or block.get(key) is None:
        return "—"
    return str(int(block[key]))


def _meta_line(page: DemoPage) -> str:
    who = "LIVE TypeSafe Jev" if page.grade_source == "live" else "MockJev (offline heuristic)"
    bits = [
        f"grader {who}",
        f"provider {_esc(page.provider or '—')}",
        f"model {_esc(page.model or '—')}",
        f"{page.n_traces} traces",
        f"{page.n_steps} steps",
        f"{page.n_task_success} traces with task success",
    ]
    if page.run_at:
        bits.append(_esc(page.run_at))
    rule = page.jev_rule or "tool_ok≥0.6 and progress≥0.5 and abort<0.5"
    return (
        f'<p class="meta">{_esc(" · ".join(bits))}</p>'
        f'<p class="meta">jev_accept = <code>{_esc(rule)}</code> · '
        "positive class = <code>expected_ok</code> · "
        "token acceptor = <code>token_accept ≥ 0.5</code> from mock_speculative</p>"
    )


def _cards(page: DemoPage) -> str:
    gold = page.gold or {}
    jev = gold.get("jev_acceptor") or {}
    token = gold.get("token_acceptor") or {}
    caught = gold.get("n_high_token_bad_caught_by_jev")
    bad = gold.get("n_high_token_bad")
    jev_name = "LIVE Jev" if page.grade_source == "live" else "MockJev"
    token_f1 = _metric(token, "f1")
    jev_f1 = _metric(jev, "f1")
    if not jev:
        jev_f1 = "—"
        jev_sub = f"{page.n_jev_accept}/{page.n_steps} steps accepted"
    else:
        jev_sub = (
            f"P {_metric(jev, 'precision')} · R {_metric(jev, 'recall')} · "
            f"tp {_int(jev, 'tp')} fp {_int(jev, 'fp')} tn {_int(jev, 'tn')} fn {_int(jev, 'fn')}"
        )
    if not token:
        token_f1 = "—"
        token_sub = "no gold comparison in this report"
    else:
        token_sub = (
            f"P {_metric(token, 'precision')} · R {_metric(token, 'recall')} · "
            f"tp {_int(token, 'tp')} fp {_int(token, 'fp')} tn {_int(token, 'tn')} fn {_int(token, 'fn')}"
        )
    if bad is None:
        caught_stat = "—"
        caught_sub = "no gold-bad slice in this report"
    else:
        caught_stat = f"{int(caught or 0)}/{int(bad)}"
        caught_sub = "gold-bad steps with token_accept ≥ 0.5 that Jev rejected"
    scope_stat = str(page.n_steps)
    scope_sub = (
        f"{page.n_traces} traces · {page.n_task_success} task successes · "
        f"{page.n_jev_accept} steps jev_accept"
    )
    cards = [
        ("card-jev", jev_name, jev_f1, "F1 vs expected_ok", jev_sub),
        ("card-token", "token_accept ≥ 0.5", token_f1, "F1 · mock_speculative", token_sub),
        ("card-caught", "Caught?", caught_stat, "high token_accept, bad step", caught_sub),
        ("card-scope", "Scope", scope_stat, "labeled steps", scope_sub),
    ]
    body = ['<section class="cards">']
    for klass, kicker, stat, label, sub in cards:
        body.append(
            f'<article class="card {klass}">'
            f'<p class="kicker">{_esc(kicker)}</p>'
            f'<p class="stat">{_esc(stat)}</p>'
            f'<p class="label">{_esc(label)}</p>'
            f'<p class="sub">{_esc(sub)}</p>'
            "</article>"
        )
    body.append("</section>")
    return "\n".join(body)


def _agreement_block(page: DemoPage) -> str:
    agreement = page.agreement
    if not agreement or not agreement.get("n_matched"):
        return ""
    n_agree = int(agreement["n_accept_agree"])
    n_matched = int(agreement["n_matched"])
    n_differ = int(agreement["n_noul_differ"])
    live_n = int(agreement["n_unique_tool_ok_report"])
    mock_n = int(agreement["n_unique_tool_ok_mock"])
    if page.agreement_mode == "mock-page":
        when = agreement.get("committed_run_at") or "committed file"
        text = (
            f"MOCK grade. Accept/reject matches the committed LIVE run ({when}) on "
            f"{n_agree}/{n_matched} steps. Nouls differ on {n_differ}/{n_matched} steps "
            f"({live_n} distinct LIVE tool_ok values, {mock_n} for MockJev on this page). "
            "Same gate, different scores. Render the LIVE nouls with no key: "
            "spectrace demo --from-report docs/live-jev-grade-15traces-20261010.json"
        )
    else:
        text = (
            "Same decisions, different scores. "
            f"Accept/reject matches offline MockJev on {n_agree}/{n_matched} steps. "
            f"Nouls differ on {n_differ}/{n_matched} steps "
            f"({live_n} distinct live tool_ok values, {mock_n} for MockJev)."
        )
    return (
        '<aside class="callout">'
        f"<p><strong>Honest overlap.</strong> {_esc(text)}</p>"
        "</aside>"
    )


def _dual_brain() -> str:
    return """
<section class="brains">
  <h2>Two brains, two questions</h2>
  <div class="brain-row">
    <article>
      <p class="kicker">System 2 · drafts</p>
      <h3>The serving model writes the step</h3>
      <p>baseline and mock_speculative replay the recorded tokens.
      mock_speculative reports <code>token_accept</code>, a simulated draft-token rate.
      An optional OpenAI-compatible server can time a live completion.
      That path still does not grade the step.</p>
    </article>
    <p class="arrow" aria-hidden="true">→</p>
    <article>
      <p class="kicker">System 1 · grades</p>
      <h3>Jev accepts or rejects the step</h3>
      <p>TypeSafe Jev returns nouls and a choice. It does not generate tool-call JSON.
      <strong>Jev never drafts.</strong>
      <code>jev_accept</code> is tool_ok ≥ 0.6 and progress ≥ 0.5 and abort &lt; 0.5.</p>
    </article>
    <p class="arrow" aria-hidden="true">→</p>
    <article>
      <p class="kicker">Gold · labels</p>
      <h3>expected_ok is the step label</h3>
      <p>Each assistant step is marked keep or drop. Precision, recall, and F1
      compare an acceptor to that label. Task success is a separate, whole-trace check.</p>
    </article>
  </div>
</section>
""".strip()


def _acceptor_section(page: DemoPage) -> str:
    gold = page.gold
    if not gold:
        return (
            '<section><h2>Acceptor comparison</h2>'
            "<p>This report has no gold comparison block.</p></section>"
        )
    token = gold.get("token_acceptor") or {}
    jev = gold.get("jev_acceptor") or {}
    threshold = gold.get("token_accept_threshold", 0.5)
    jev_name = "jev_accept · LIVE" if page.grade_source == "live" else "jev_accept · MockJev"
    token_name = f"token_accept ≥ {threshold} · mock_speculative"
    disclaimer = str(gold.get("disclaimer") or "")
    return "\n".join(
        [
            "<section>",
            "<h2>Acceptor comparison</h2>",
            f"<p class=\"note\">{_esc(disclaimer)}</p>" if disclaimer else "",
            '<div class="table-wrap">',
            "<table>",
            "<thead><tr>"
            "<th>acceptor</th><th class=\"num\">precision</th><th class=\"num\">recall</th>"
            "<th class=\"num\">F1</th><th class=\"num\">tp</th><th class=\"num\">fp</th>"
            "<th class=\"num\">tn</th><th class=\"num\">fn</th>"
            "</tr></thead>",
            "<tbody>",
            _metric_tr(token_name, token, emphasize=False),
            _metric_tr(jev_name, jev, emphasize=True),
            "</tbody></table></div>",
            _confusion_bars(token, jev, token_name, jev_name),
            "</section>",
        ]
    )


def _metric_tr(name: str, block: dict[str, Any], *, emphasize: bool) -> str:
    klass = ' class="emph"' if emphasize else ""
    cells = [
        _metric(block, "precision"),
        _metric(block, "recall"),
        _metric(block, "f1"),
        _int(block, "tp"),
        _int(block, "fp"),
        _int(block, "tn"),
        _int(block, "fn"),
    ]
    tds = "".join(f'<td class="num">{_esc(c)}</td>' for c in cells)
    return f"<tr{klass}><th scope=\"row\">{_esc(name)}</th>{tds}</tr>"


def _confusion_bars(
    token: dict[str, Any],
    jev: dict[str, Any],
    token_name: str,
    jev_name: str,
) -> str:
    return (
        '<div class="bars">'
        + _stack(token_name, token)
        + _stack(jev_name, jev)
        + '<p class="legend"><span class="swatch tp"></span> tp keep, correct '
        '<span class="swatch fp"></span> fp kept a bad step '
        '<span class="swatch tn"></span> tn drop, correct '
        '<span class="swatch fn"></span> fn dropped a good step</p>'
        "</div>"
    )


def _stack(name: str, block: dict[str, Any]) -> str:
    parts = [("tp", "tp"), ("fp", "fp"), ("tn", "tn"), ("fn", "fn")]
    counts = []
    total = 0
    for key, _klass in parts:
        try:
            n = int(block.get(key) or 0)
        except (TypeError, ValueError):
            n = 0
        counts.append(n)
        total += n
    if total <= 0:
        return ""
    spans = []
    for (key, klass), n in zip(parts, counts):
        if n <= 0:
            continue
        width = 100.0 * n / total
        spans.append(
            f'<span class="seg {klass}" style="width:{width:.2f}%" title="{key} {n}">{n}</span>'
        )
    return (
        f'<div class="stack"><p>{_esc(name)}</p>'
        f'<div class="stack-bar">{"".join(spans)}</div></div>'
    )


def _caught_section(page: DemoPage) -> str:
    gold = page.gold or {}
    rows = gold.get("high_token_accept_bad_steps") or []
    n_bad = gold.get("n_high_token_bad", len(rows))
    n_caught = gold.get("n_high_token_bad_caught_by_jev", 0)
    who = "LIVE Jev" if page.grade_source == "live" else "MockJev"
    intro = (
        f"{who} caught {int(n_caught)}/{int(n_bad)} gold-bad steps "
        "where mock_speculative token_accept was at least 0.5. "
        "A fluent sentence can still be the wrong step."
    )
    if not rows:
        return (
            "<section><h2>High token_accept, bad step — caught?</h2>"
            f"<p>{_esc(intro)}</p><p>No rows in this slice.</p></section>"
        )
    body = [
        "<section>",
        "<h2>High token_accept, bad step — caught?</h2>",
        f"<p>{_esc(intro)}</p>",
        '<div class="table-wrap">',
        "<table>",
        "<thead><tr><th>trace</th><th class=\"num\">step</th><th>kind</th><th>failure mode</th>"
        "<th class=\"num\">token_accept</th><th>jev_accept</th><th>caught?</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        caught = _fmt_bool(row.get("jev_caught"))
        klass = "caught" if caught == "yes" else "missed"
        body.append(
            "<tr class=\"{klass}\">"
            "<td><code>{trace}</code></td><td class=\"num\">{step}</td><td>{kind}</td>"
            "<td><code>{mode}</code></td><td class=\"num\">{token}</td><td>{jev}</td><td>{caught_cell}</td>"
            "</tr>".format(
                klass=klass,
                trace=_esc(row.get("trace")),
                step=_esc(row.get("step")),
                kind=_esc(row.get("step_kind") or ""),
                mode=_esc(row.get("failure_mode") or ""),
                token=_esc(_fmt_rate(row.get("token_accept"))),
                jev=_yn(row.get("jev_accept")),
                caught_cell=_yn(row.get("jev_caught")),
            )
        )
    body.append("</tbody></table></div></section>")
    return "\n".join(body)


def _chart_section(page: DemoPage) -> str:
    if not page.rows:
        return ""
    who = "LIVE Jev" if page.grade_source == "live" else "MockJev"
    caption = (
        f"Amber bars are simulated token_accept. Dots are {who}: "
        "green at 1 means accept, vermillion at 0 means reject. "
        "The dashed line is the 0.5 token threshold. Not a GPU trace."
    )
    return (
        "<section>"
        "<h2>Every step</h2>"
        f"<p>{_esc(caption)}</p>"
        f"{_steps_svg(page.rows)}"
        "</section>"
    )


def _steps_svg(rows: list[dict[str, Any]]) -> str:
    n = len(rows)
    width = 1040
    height = 286
    pad_l, pad_r, pad_t, pad_b = 36, 12, 16, 64
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    def y_of(value: float) -> float:
        return pad_t + (1.0 - value) * plot_h

    def x_of(i: int) -> float:
        if n == 1:
            return pad_l + plot_w / 2
        return pad_l + (i + 0.5) * plot_w / n

    bar_w = max(2.0, plot_w / n * 0.72)
    chunks: list[str] = [
        f'<svg class="chart" viewBox="0 0 {width} {height}" role="img" '
        'aria-label="token_accept bars and jev_accept dots for each step">',
        "<title>token_accept versus jev_accept by step</title>",
    ]
    # Alternating trace bands.
    start = 0
    band = 0
    for i, row in enumerate(rows):
        nxt = rows[i + 1]["trace"] if i + 1 < n else None
        if nxt != row.get("trace"):
            if band % 2 == 0:
                x0 = pad_l + start * plot_w / n
                x1 = pad_l + (i + 1) * plot_w / n
                chunks.append(
                    f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" height="{plot_h:.1f}" '
                    'class="band"></rect>'
                )
            band += 1
            start = i + 1
    # Threshold.
    y_thr = y_of(0.5)
    chunks.append(
        f'<line x1="{pad_l}" y1="{y_thr:.1f}" x2="{width - pad_r}" y2="{y_thr:.1f}" class="threshold"></line>'
    )
    chunks.append(
        f'<text x="{pad_l + 4}" y="{y_thr - 3:.1f}" class="tick">0.5</text>'
    )
    for tick, label in ((0.0, "0"), (1.0, "1")):
        y = y_of(tick)
        chunks.append(
            f'<text x="4" y="{y + 3:.1f}" class="tick">{label}</text>'
        )
    for i, row in enumerate(rows):
        raw = row.get("token_accept")
        try:
            rate = float(raw) if raw is not None else 0.0
        except (TypeError, ValueError):
            rate = 0.0
        rate = min(1.0, max(0.0, rate))
        x = x_of(i) - bar_w / 2
        y = y_of(rate)
        h = y_of(0.0) - y
        bad = row.get("expected_ok") is False and rate >= 0.5
        klass = "bar bar-bad" if bad else "bar"
        chunks.append(
            f'<rect class="{klass}" x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{max(h, 0):.2f}"></rect>'
        )
        accepted = bool(row.get("jev_accept"))
        cy = y_of(1.0 if accepted else 0.0)
        dot = "dot-yes" if accepted else "dot-no"
        chunks.append(f'<circle class="{dot}" cx="{x_of(i):.2f}" cy="{cy:.2f}" r="3.2"></circle>')
    # Trace labels at group centers.
    start = 0
    for i, row in enumerate(rows):
        nxt = rows[i + 1]["trace"] if i + 1 < n else None
        if nxt != row.get("trace"):
            mid = (start + i) / 2
            label = _SHORT.get(str(row.get("trace")), str(row.get("trace") or "")[:8])
            lx = x_of(mid)
            ly = height - 8
            chunks.append(
                f'<text class="xlabel" x="{lx:.1f}" y="{ly:.1f}" '
                f'transform="rotate(-40 {lx:.1f} {ly:.1f})">{_esc(label)}</text>'
            )
            start = i + 1
    chunks.append("</svg>")
    return "\n".join(chunks)


def _trace_section(page: DemoPage) -> str:
    groups: list[tuple[str, list[dict[str, Any]]]] = []
    for row in page.rows:
        tid = str(row.get("trace") or "")
        if not groups or groups[-1][0] != tid:
            groups.append((tid, [row]))
        else:
            groups[-1][1].append(row)
    if not groups:
        return ""
    body = [
        "<section>",
        "<h2>Per trace</h2>",
        '<div class="table-wrap"><table>',
        "<thead><tr><th>trace</th><th>failure mode</th><th>task success</th>"
        "<th class=\"num\">steps</th><th class=\"num\">jev accept</th>"
        "<th class=\"num\">mean token_accept</th></tr></thead><tbody>",
    ]
    for tid, chunk in groups:
        mode = next((r.get("failure_mode") for r in chunk if r.get("failure_mode")), None)
        task = any(bool(r.get("task_success")) for r in chunk)
        n_yes = sum(1 for r in chunk if r.get("jev_accept"))
        rates = []
        for r in chunk:
            raw = r.get("token_accept")
            if raw is None:
                continue
            try:
                rates.append(float(raw))
            except (TypeError, ValueError):
                continue
        mean = sum(rates) / len(rates) if rates else None
        body.append(
            "<tr><td><code>{tid}</code></td><td>{mode}</td><td>{task}</td>"
            "<td class=\"num\">{n}</td><td class=\"num\">{yes}/{n}</td><td class=\"num\">{mean}</td></tr>".format(
                tid=_esc(tid),
                mode=_esc(mode or "—"),
                task=_yn(task),
                n=len(chunk),
                yes=n_yes,
                mean=_esc(_fmt_rate(mean) if mean is not None else "—"),
            )
        )
    body.append("</tbody></table></div></section>")
    return "\n".join(body)


def _bakeoff_section(page: DemoPage) -> str:
    bakeoff = page.bakeoff
    if not bakeoff:
        return (
            "<section><h2>Bake-off latency and cost</h2>"
            "<p>SIMULATED table omitted — traces were not loaded next to this report.</p></section>"
        )
    runs = bakeoff.get("runs") or []
    cost = bakeoff.get("cost_per_successful_task_usd")
    try:
        cost_s = "n/a" if cost is None or (isinstance(cost, float) and math.isnan(cost)) else f"{float(cost):.4f}"
    except (TypeError, ValueError):
        cost_s = "n/a"
    banner = str(bakeoff.get("banner") or "SIMULATED")
    body = [
        "<section>",
        "<h2>Bake-off latency and cost</h2>",
        f'<p class="sim">{_esc(banner)}</p>',
        f"<p>{_esc(bakeoff.get('n_runs', len(runs)))} runs · "
        f"{_esc(bakeoff.get('n_success', 0))} successful runs · "
        f"cost per successful task ${ _esc(cost_s) }. "
        "Seed 0, gamma 5, default price table all zeros.</p>",
        '<div class="table-wrap"><table>',
        "<thead><tr><th>trace</th><th>method</th><th class=\"num\">latency ms</th>"
        "<th class=\"num\">accept</th><th class=\"num\">tokens in</th><th class=\"num\">tokens out</th>"
        "<th>success</th><th class=\"num\">cost usd</th><th>failure</th></tr></thead><tbody>",
    ]
    for run in runs:
        accept = run.get("accept_rate")
        body.append(
            "<tr><td><code>{tid}</code></td><td>{method}</td><td class=\"num\">{lat}</td>"
            "<td class=\"num\">{acc}</td><td class=\"num\">{tin}</td><td class=\"num\">{tout}</td>"
            "<td>{ok}</td><td class=\"num\">{cost}</td><td>{fail}</td></tr>".format(
                tid=_esc(run.get("trace_id")),
                method=_esc(run.get("method")),
                lat=_esc(_fmt_rate(run.get("wall_latency_ms"), 1)),
                acc=_esc(_fmt_rate(accept) if accept is not None else "—"),
                tin=_esc(run.get("tokens_in")),
                tout=_esc(run.get("tokens_out")),
                ok=_yn(run.get("success")),
                cost=_esc(f"{float(run.get('cost_estimate_usd') or 0):.4f}"),
                fail=_esc(run.get("failure_mode") or "—"),
            )
        )
    body.append("</tbody></table></div></section>")
    return "\n".join(body)


def _caveats(page: DemoPage) -> str:
    items = [
        "Traces are synthetic-but-realistic fixtures, not production logs and not ShareGPT.",
        "n is small: the 15-trace set is 49 labeled steps.",
        "token_accept is mock_speculative (a simulator). It is not a GPU speculative-decoding measurement.",
        "Latency and cost in the bake-off table are simulated. The default price table is $0.",
        "MockJev is a heuristic written against this taxonomy. A mock F1 of 1.000 means the heuristic matches these labels.",
        "Where LIVE accept/reject matches MockJev, the nouls still differ. Do not treat the heuristic as the model.",
        "Jev never drafts. OpenAI-compatible serving, when enabled, is System 2 only and is not used by this demo's default path.",
        "The 2026-09-20 LIVE file is five happy-path traces (22/22 accepts). It does not show a rejection.",
    ]
    if page.notes:
        items.append(page.notes)
    lis = "".join(f"<li>{_esc(item)}</li>" for item in items)
    return f"<section class=\"caveats\"><h2>Limitations</h2><ul>{lis}</ul></section>"


_CSS = """
:root {
  --paper: #f4efe6;
  --ink: #1c1917;
  --muted: #57534e;
  --line: #e4d9c8;
  --card: #fffdf8;
  --jev: #0f766e;
  --token: #b45309;
  --bad: #c2410c;
  --task: #1d4ed8;
  --yes: #047857;
  --no: #b91c1c;
  --band: #f3ecdf;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background:
    radial-gradient(1200px 420px at 10% -10%, #efe4cf 0%, transparent 60%),
    var(--paper);
  color: var(--ink);
  font: 15px/1.5 ui-sans-serif, "Segoe UI", sans-serif;
}
code { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; font-size: 0.86em; }
main { max-width: 1120px; margin: 0 auto; padding: 28px 22px 64px; }
h1 { font-family: Georgia, "Iowan Old Style", Palatino, serif; font-size: 40px; line-height: 1.12; margin: 0 0 10px; letter-spacing: -0.02em; }
h2 { font-family: Georgia, Palatino, serif; font-size: 26px; margin: 0 0 8px; }
h3 { margin: 0 0 6px; font-size: 16px; }
.top { background: var(--ink); color: #f6f1e7; }
.top-inner { max-width: 1120px; margin: 0 auto; padding: 14px 22px; display: flex; justify-content: space-between; gap: 16px; align-items: center; }
.brand { margin: 0; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; font-size: 13px; }
.brand span { font-weight: 500; letter-spacing: 0; text-transform: none; color: #d6d3d1; margin-left: 8px; }
.pill { margin: 0; border-radius: 999px; padding: 4px 12px; font-size: 12px; font-weight: 700; letter-spacing: 0.02em; }
.pill-live { background: #ccfbf1; color: #115e59; }
.pill-mock { background: #ffedd5; color: #9a3412; }
.hero { padding: 8px 0 4px; }
.lede { font-size: 17px; max-width: 68ch; margin: 0 0 8px; }
.meta { color: var(--muted); margin: 0 0 4px; font-size: 13px; }
.cards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }
.card { background: var(--card); border: 1px solid var(--line); border-top: 4px solid var(--ink); border-radius: 12px; padding: 12px 14px 14px; }
.card-jev { border-top-color: var(--jev); }
.card-token { border-top-color: var(--token); }
.card-caught { border-top-color: var(--bad); }
.card-scope { border-top-color: var(--task); }
.kicker { margin: 0; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; }
.stat { margin: 4px 0 0; font-family: Georgia, Palatino, serif; font-size: 34px; line-height: 1; font-variant-numeric: tabular-nums; }
.card-jev .stat { color: var(--jev); }
.card-token .stat { color: var(--token); }
.card-caught .stat { color: var(--bad); }
.label { margin: 6px 0 0; font-weight: 650; }
.sub { margin: 2px 0 0; color: var(--muted); font-size: 12.5px; }
.callout { background: #fff7ed; border: 1px solid #fdba74; border-radius: 12px; padding: 12px 14px; margin: 8px 0 18px; }
.callout p { margin: 0; }
section { margin: 28px 0; }
.brains .brain-row { display: grid; grid-template-columns: 1fr auto 1fr auto 1fr; gap: 10px; align-items: stretch; }
.brains article { background: var(--card); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
.brains article p:last-child { margin: 0; color: var(--ink); }
.arrow { margin: 0; align-self: center; color: var(--muted); font-size: 22px; }
.note, .sim { color: var(--muted); }
.sim { font-weight: 700; color: #9a3412; }
.table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--card); }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 13.5px; }
th, td { padding: 7px 10px; text-align: left; border-bottom: 1px solid var(--line); vertical-align: top; }
thead th { background: var(--ink); color: #f6f1e7; font-weight: 650; font-size: 12px; letter-spacing: 0.02em; }
th.num, td.num { text-align: right; font-variant-numeric: tabular-nums; }
tr.emph th, tr.emph td { background: #f0fdfa; }
tr.caught td { background: #f0fdfa; }
tr.missed td { background: #fff7ed; }
.yes { color: var(--yes); font-weight: 700; }
.no { color: var(--no); font-weight: 700; }
.muted { color: var(--muted); }
.bars { margin-top: 14px; }
.stack { margin: 8px 0; }
.stack p { margin: 0 0 4px; font-size: 12.5px; color: var(--muted); }
.stack-bar { display: flex; height: 22px; border-radius: 6px; overflow: hidden; background: var(--line); }
.seg { color: white; font-size: 11px; line-height: 22px; text-align: center; font-weight: 700; }
.tp { background: #0f766e; }
.fp { background: #c2410c; }
.tn { background: #57534e; }
.fn { background: #d97706; }
.legend { color: var(--muted); font-size: 12.5px; }
.swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin: 0 4px 0 10px; }
.swatch.tp { background: #0f766e; }
.swatch.fp { background: #c2410c; }
.swatch.tn { background: #57534e; }
.swatch.fn { background: #d97706; }
.chart { width: 100%; height: auto; background: var(--card); border: 1px solid var(--line); border-radius: 12px; }
.band { fill: var(--band); }
.bar { fill: #f59e0b; }
.bar-bad { fill: #ea580c; }
.threshold { stroke: #a8a29e; stroke-dasharray: 4 3; stroke-width: 1; }
.tick, .xlabel { fill: #57534e; font-size: 10px; font-family: ui-sans-serif, sans-serif; }
.xlabel { text-anchor: end; }
.dot-yes { fill: #0f766e; stroke: #fff; stroke-width: 0.6; }
.dot-no { fill: #c2410c; stroke: #fff; stroke-width: 0.6; }
.caveats ul { margin: 8px 0 0; padding-left: 18px; }
.caveats li { margin: 4px 0; }
footer { border-top: 1px solid var(--line); color: var(--muted); font-size: 13px; }
footer p { max-width: 1120px; margin: 8px auto; padding: 0 22px; }
@media (max-width: 900px) {
  .cards { grid-template-columns: 1fr 1fr; }
  .brains .brain-row { grid-template-columns: 1fr; }
  .arrow { transform: rotate(90deg); text-align: center; }
  h1 { font-size: 32px; }
  .top-inner { flex-direction: column; align-items: flex-start; }
}
"""

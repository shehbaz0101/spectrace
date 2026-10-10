"""Offline demo: bake-off + step grade, written as one static HTML file.

Default path is MockJev and the mock decoder ($0, no network).
``--from-report`` renders a saved grade JSON (the committed LIVE file)
without a key. ``--live`` calls TypeSafe Jev only when a key is set.
Jev grades steps. It does not draft tokens. The latency table is always
the offline simulator.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from spectrace.demo_html import DemoPage, render_demo_html
from spectrace.grade import DEFAULT_TOKEN_ACCEPT_THRESHOLD, grade_traces
from spectrace.jev import (
    DEFAULT_MODEL as JEV_DEFAULT_MODEL,
    MockJev,
    api_key_from_env,
    resolve_grade_provider,
)
from spectrace.metrics import load_price_table
from spectrace.providers import mock_provider
from spectrace.replay import run_bakeoff
from spectrace.traces import TraceError, load_traces

DEFAULT_OUTPUT = "reports/demo/index.html"
LIVE_REPORT_NAME = "live-jev-grade-15traces-20261010.json"

_NOUL_KEYS = ("tool_ok", "progress", "abort")


class DemoError(Exception):
    """User-facing demo failure. CLI prints it and exits 2."""


def live_needs_key_message() -> str:
    return (
        "spectrace demo --live grades with --provider jev and needs "
        "TYPESAFE_API_KEY (or TYPESAFE_KEY). No key is set, so this command "
        "will not call the API. Offline and $0: spectrace demo. "
        "To see the committed LIVE numbers with no key: "
        "spectrace demo --from-report docs/" + LIVE_REPORT_NAME
    )


def both_modes_message() -> str:
    return (
        "Pass either --live or --from-report, not both. "
        "--from-report renders a saved grade JSON and does not call an API."
    )


def load_grade_report(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise DemoError(f"grade report not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DemoError(f"{path} is not JSON ({exc})") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        raise DemoError(
            f"{path} is not a spectrace grade report (expected an object with a 'rows' list)"
        )
    if not payload["rows"]:
        raise DemoError(f"{path} has no step rows")
    return payload


def report_is_live(payload: dict[str, Any]) -> bool:
    """True when the file is a real Jev run, not a MockJev grade."""
    if payload.get("live") is True or str(payload.get("label") or "").upper() == "LIVE":
        return True
    provider = str(payload.get("provider") or "").strip().lower()
    gold = payload.get("gold_comparison") if isinstance(payload.get("gold_comparison"), dict) else {}
    if provider == "jev" and gold.get("is_mock") is not True:
        return True
    return False


def _row_get(row: Any, key: str) -> Any:
    if isinstance(row, dict):
        return row.get(key)
    return getattr(row, key, None)


def _noul(row: Any, key: str) -> float | None:
    val = _row_get(row, key)
    if val is None or val == "":
        return None
    try:
        return round(float(val), 4)
    except (TypeError, ValueError):
        return None


def compare_to_mock(report_rows: list[Any], mock_rows: list[Any]) -> dict[str, Any]:
    """Accept/reject and noul overlap between a grade report and MockJev.

    ``report_rows`` may be dicts (a saved JSON file) or ``StepGrade`` objects.
    Matching is by ``(trace, step)``. This does not call an API.
    """
    mock_by = {(_row_get(r, "trace"), _row_get(r, "step")): r for r in mock_rows}
    n = 0
    agree = 0
    noul_differ = 0
    unmatched = 0
    report_tools: list[float] = []
    mock_tools: list[float] = []
    for row in report_rows:
        n += 1
        other = mock_by.get((_row_get(row, "trace"), _row_get(row, "step")))
        if other is None:
            unmatched += 1
            continue
        if bool(_row_get(row, "jev_accept")) == bool(_row_get(other, "jev_accept")):
            agree += 1
        report_tool = _noul(row, "tool_ok")
        mock_tool = _noul(other, "tool_ok")
        if report_tool is not None:
            report_tools.append(report_tool)
        if mock_tool is not None:
            mock_tools.append(mock_tool)
        if any(_noul(row, key) != _noul(other, key) for key in _NOUL_KEYS):
            noul_differ += 1
    return {
        "n_steps": n,
        "n_matched": n - unmatched,
        "n_unmatched": unmatched,
        "n_accept_agree": agree,
        "n_noul_differ": noul_differ,
        "n_unique_tool_ok_report": len(set(report_tools)),
        "n_unique_tool_ok_mock": len(set(mock_tools)),
        "decisions_match": unmatched == 0 and n > 0 and agree == n,
        "scores_differ_every_step": unmatched == 0 and n > 0 and noul_differ == n,
    }


def find_committed_live_report(traces: str) -> Path | None:
    """Locate the checked-in 15-trace LIVE file, if this checkout has it."""
    name = LIVE_REPORT_NAME
    candidates = [
        Path(traces).resolve().parent / "docs" / name,
        Path.cwd() / "docs" / name,
        Path(__file__).resolve().parents[1] / "docs" / name,
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _bakeoff_payload(traces_path: str, *, seed: int, gamma: int) -> dict[str, Any]:
    summary = run_bakeoff(
        load_traces(traces_path),
        ["baseline", "mock_speculative"],
        seed=seed,
        gamma=gamma,
        provider=mock_provider(),
        price_table=load_price_table(None),
        model="mock-local",
    )
    payload = summary.to_dict()
    payload["simulated"] = True
    payload["banner"] = (
        "SIMULATED — mock decoder on fixture traces. "
        "Not GPU latency and not a paid API. Default cost is $0."
    )
    return payload


def _mock_grade(traces_path: str, *, seed: int, gamma: int, token_threshold: float):
    return grade_traces(
        load_traces(traces_path),
        MockJev(),
        provider="mock",
        model="mock",
        seed=seed,
        gamma=gamma,
        token_threshold=token_threshold,
    )


def _try_bakeoff(traces_path: str, *, seed: int, gamma: int) -> dict[str, Any] | None:
    try:
        return _bakeoff_payload(traces_path, seed=seed, gamma=gamma)
    except (FileNotFoundError, TraceError, OSError):
        return None


def _try_mock_rows(traces_path: str, *, seed: int, gamma: int, token_threshold: float):
    try:
        return _mock_grade(
            traces_path, seed=seed, gamma=gamma, token_threshold=token_threshold
        ).rows
    except (FileNotFoundError, TraceError, OSError):
        return None


def _page_copy(
    *,
    grade: dict[str, Any],
    source: str,
    agreement: dict[str, Any] | None,
    command_hint: str,
) -> DemoPage:
    live = report_is_live(grade)
    grade_source = "live" if live else "mock"
    run_at = str(grade.get("run_at") or "").strip()
    if live:
        when = f" · {run_at}" if run_at else ""
        kicker = f"LIVE · TypeSafe Jev{when}"
        headline = "A fluent draft is not a good step."
        lede = (
            "System 2 drafts tokens. System 1 (Jev) grades the step and never writes it. "
            "On this fixture set, LIVE Jev matched the gold step labels. "
            "A token-accept threshold did not. Small n, synthetic traces, simulated token_accept."
        )
    else:
        kicker = "MOCK · offline · $0"
        headline = "Offline bake-off and step grade."
        lede = (
            "These cards are MockJev, a local heuristic, not a TypeSafe API call. "
            "System 2 drafts tokens. System 1 grades the step. Jev never drafts. "
            "token_accept and the latency table are simulated."
        )
    if source == "mock":
        agreement_mode = "mock-page"
    elif live:
        agreement_mode = "live-page"
    else:
        agreement_mode = ""
    gold = grade.get("gold_comparison")
    if not isinstance(gold, dict):
        gold = None
    rows = grade.get("rows") or []
    return DemoPage(
        grade_source=grade_source,
        kicker=kicker,
        headline=headline,
        lede=lede,
        run_at=run_at,
        provider=str(grade.get("provider") or ""),
        model=str(grade.get("model") or ""),
        jev_rule=str(grade.get("jev_accept_rule") or ""),
        notes=str(grade.get("notes") or ""),
        n_traces=int(grade.get("n_traces") or len({r.get("trace") for r in rows})),
        n_steps=int(grade.get("n_steps") or len(rows)),
        n_jev_accept=int(grade.get("n_jev_accept") or sum(1 for r in rows if r.get("jev_accept"))),
        n_task_success=int(
            grade.get("n_task_success_traces")
            or len({r.get("trace") for r in rows if r.get("task_success")})
        ),
        gold=gold,
        rows=list(rows),
        agreement=agreement,
        agreement_mode=agreement_mode if agreement else "",
        command_hint=command_hint,
    )


def run_demo(
    *,
    traces: str = "traces",
    output: str | Path = DEFAULT_OUTPUT,
    from_report: str | None = None,
    live: bool = False,
    seed: int = 0,
    gamma: int = 5,
    token_threshold: float = DEFAULT_TOKEN_ACCEPT_THRESHOLD,
) -> Path:
    """Write the demo HTML and return its path. Raises ``DemoError`` on misuse."""
    if live and from_report:
        raise DemoError(both_modes_message())
    if live and not api_key_from_env():
        raise DemoError(live_needs_key_message())

    agreement: dict[str, Any] | None = None
    command_hint = "spectrace demo"

    if from_report:
        grade = load_grade_report(from_report)
        source = "report"
        command_hint = f"spectrace demo --from-report {from_report}"
        mock_rows = _try_mock_rows(
            traces, seed=seed, gamma=gamma, token_threshold=token_threshold
        )
        if mock_rows is not None:
            agreement = compare_to_mock(grade["rows"], mock_rows)
        bakeoff = _try_bakeoff(traces, seed=seed, gamma=gamma)
    elif live:
        grader = resolve_grade_provider("jev")
        summary = grade_traces(
            load_traces(traces),
            grader,
            provider="jev",
            model=JEV_DEFAULT_MODEL,
            seed=seed,
            gamma=gamma,
            token_threshold=token_threshold,
        )
        grade = summary.to_dict()
        grade["live"] = True
        grade["label"] = "LIVE"
        source = "live"
        command_hint = "spectrace demo --live"
        mock_rows = _mock_grade(
            traces, seed=seed, gamma=gamma, token_threshold=token_threshold
        ).rows
        agreement = compare_to_mock(grade["rows"], mock_rows)
        bakeoff = _bakeoff_payload(traces, seed=seed, gamma=gamma)
    else:
        summary = _mock_grade(traces, seed=seed, gamma=gamma, token_threshold=token_threshold)
        grade = summary.to_dict()
        grade["live"] = False
        grade["label"] = "MOCK"
        source = "mock"
        committed = find_committed_live_report(traces)
        if committed is not None:
            live_payload = load_grade_report(committed)
            agreement = compare_to_mock(live_payload["rows"], summary.rows)
            if not grade.get("run_at"):
                agreement["committed_run_at"] = str(live_payload.get("run_at") or "")
        bakeoff = _bakeoff_payload(traces, seed=seed, gamma=gamma)

    page = _page_copy(
        grade=grade, source=source, agreement=agreement, command_hint=command_hint
    )
    page.bakeoff = bakeoff
    html = render_demo_html(page)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    _print_banner(page, bakeoff)
    print(f"wrote {out}")
    return out


def _fmt_f1(gold: dict[str, Any] | None, key: str) -> str:
    block = (gold or {}).get(key) or {}
    val = block.get("f1")
    if val is None:
        return "n/a"
    return f"{float(val):.3f}"


def _print_banner(page: DemoPage, bakeoff: dict[str, Any] | None) -> None:
    kind = "LIVE" if page.grade_source == "live" else "MOCK"
    when = f"  {page.run_at}" if page.run_at else ""
    print(
        f"spectrace demo  ·  grade {kind}{when}  ·  "
        f"{page.n_steps} steps  ·  {page.n_traces} traces  ·  "
        f"jev F1 {_fmt_f1(page.gold, 'jev_acceptor')}  ·  "
        f"token F1 {_fmt_f1(page.gold, 'token_acceptor')}"
    )
    agreement = page.agreement or {}
    if agreement.get("n_matched"):
        if page.agreement_mode == "mock-page":
            print(
                "MOCK accept/reject matches the committed LIVE run on "
                f"{agreement['n_accept_agree']}/{agreement['n_matched']} steps; "
                f"nouls differ on {agreement['n_noul_differ']}/{agreement['n_matched']} "
                f"(LIVE tool_ok {agreement['n_unique_tool_ok_report']} distinct, "
                f"MockJev {agreement['n_unique_tool_ok_mock']})"
            )
        else:
            print(
                "accept/reject matches offline MockJev on "
                f"{agreement['n_accept_agree']}/{agreement['n_matched']} steps; "
                f"nouls differ on {agreement['n_noul_differ']}/{agreement['n_matched']} "
                f"(live tool_ok {agreement['n_unique_tool_ok_report']} distinct, "
                f"MockJev {agreement['n_unique_tool_ok_mock']})"
            )
    if bakeoff is None:
        print("bakeoff: skipped (traces not loaded). Latency table omitted.")
    else:
        cost = bakeoff.get("cost_per_successful_task_usd")
        if cost is None or (isinstance(cost, float) and math.isnan(cost)):
            cost_s = "n/a"
        else:
            cost_s = f"{float(cost):.4f}"
        print(
            f"bakeoff: SIMULATED  {bakeoff.get('n_runs', 0)} runs  "
            f"successful {bakeoff.get('n_success', 0)}  "
            f"cost per successful task ${cost_s}"
        )

"""Gold labels vs mock acceptors. No network, no API key."""

from __future__ import annotations

from spectrace.grade import StepGrade, build_gold_comparison, grade_traces
from spectrace.jev import MockJev
from spectrace.traces import load_traces


def test_mock_grade_agrees_with_step_labels(traces_dir):
    traces = load_traces(traces_dir)
    summary = grade_traces(traces, MockJev(), provider="mock", model="mock")
    by_id = {t.id: t for t in traces}
    assert summary.extra["gold_comparison"]["is_mock"] is True
    mismatches = []
    for row in summary.rows:
        trace = by_id[row.trace]
        assert row.task_success is trace.task_success
        assert row.failure_mode == trace.failure_mode
        if row.expected_ok is None:
            mismatches.append((row.trace, row.step, "unlabeled"))
            continue
        if row.jev_accept != row.expected_ok:
            mismatches.append(
                (
                    row.trace,
                    row.step,
                    row.step_kind,
                    row.expected_ok,
                    row.jev_accept,
                    row.tool_ok,
                    row.progress,
                    row.abort,
                )
            )
    assert mismatches == []


def test_token_acceptor_false_positives_are_caught_by_mock_jev(traces_dir):
    summary = grade_traces(load_traces(traces_dir), MockJev(), provider="mock")
    gold = summary.extra["gold_comparison"]
    assert "MOCK" in gold["disclaimer"]
    assert "LIVE TypeSafe" in gold["disclaimer"] or "not a LIVE" in gold["disclaimer"]
    assert gold["jev_accept_source"] == "MockJev"
    assert gold["token_accept_source"] == "mock_speculative"
    assert gold["live_rerun"].startswith("spectrace grade --provider jev")
    token = gold["token_acceptor"]
    jev = gold["jev_acceptor"]
    assert token["fp"] > 0
    assert jev["fp"] == 0
    assert jev["fn"] == 0
    assert jev["f1"] == 1.0
    assert token["f1"] < jev["f1"]
    assert gold["n_high_token_bad"] > 0
    assert gold["n_high_token_bad_caught_by_jev"] == gold["n_high_token_bad"]
    assert all(step["jev_caught"] is True for step in gold["high_token_accept_bad_steps"])
    assert all(step["expected_ok"] is False for step in gold["high_token_accept_bad_steps"])
    # Counts quoted in the README. Update both together if the fixtures change.
    assert gold["n_labeled_steps"] == 49
    assert gold["n_high_token_bad"] == 11
    assert (token["tp"], token["fp"], token["tn"], token["fn"]) == (7, 11, 9, 22)
    assert (jev["tp"], jev["fp"], jev["tn"], jev["fn"]) == (29, 0, 20, 0)


def test_live_disclaimer_is_not_used_for_mock_rows():
    row = StepGrade(
        trace="synthetic",
        step=0,
        token_accept=0.8,
        jev_accept=False,
        tool_ok=0.1,
        progress=0.1,
        abort=0.9,
        task_success=False,
        expected_ok=False,
        failure_mode="off_goal",
        step_kind="text",
    )
    mock = build_gold_comparison([row], threshold=0.5, provider="mock")
    live = build_gold_comparison([row], threshold=0.5, provider="jev")
    assert mock["is_mock"] is True and "MOCK" in mock["disclaimer"]
    assert live["is_mock"] is False
    assert live["jev_accept_source"] == "live"
    assert "mock_speculative" in live["disclaimer"]
    assert live["n_high_token_bad"] == 1
    assert live["n_high_token_bad_caught_by_jev"] == 1
    # Building the block does not invent a second set of nouls.
    assert "tool_ok" not in live

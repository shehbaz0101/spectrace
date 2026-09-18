from __future__ import annotations

import json

from spectrace.cli import main
from spectrace.grade import grade_trace, grade_traces, iter_agent_steps
from spectrace.jev import MockJev
from spectrace.traces import AgentTrace, Message, SuccessCriteria, ToolCall, load_traces


def test_grade_traces_mock_covers_all_fixture_steps(traces_dir):
    traces = load_traces(traces_dir)
    summary = grade_traces(traces, MockJev(), provider="mock", model="mock")
    assert summary.n_traces == 5
    assert summary.n_steps == sum(len(t.assistant_turns) for t in traces)
    assert summary.n_steps > 10
    assert summary.jev_accept_rule
    for row in summary.rows:
        assert row.trace
        assert row.step >= 0
        assert row.token_accept is None or 0.0 <= row.token_accept <= 1.0
        assert isinstance(row.jev_accept, bool)
        assert row.tool_ok is None or 0.0 <= row.tool_ok <= 1.0
        assert row.progress is None or 0.0 <= row.progress <= 1.0
        assert row.abort is None or 0.0 <= row.abort <= 1.0
        assert isinstance(row.task_success, bool)
        payload = row.to_dict()
        assert set(payload) >= {
            "trace",
            "step",
            "token_accept",
            "jev_accept",
            "tool_ok",
            "progress",
            "abort",
            "task_success",
        }


def test_grade_cli_mock_prints_comparison_table(traces_dir, capsys):
    code = main(["grade", "--provider", "mock", "--traces", str(traces_dir)])
    assert code == 0
    out = capsys.readouterr().out
    for col in ("trace", "step", "token_accept", "jev_accept", "tool_ok", "progress", "abort"):
        assert col in out
    assert "task_success" in out
    assert "jev_accept rule" in out
    for tid in (
        "research_qa_01",
        "code_fix_01",
        "multihop_retrieval_01",
        "travel_planner_01",
        "data_analysis_01",
    ):
        assert tid in out
    assert "System-1" in out or "grader" in out.lower()


def test_grade_cli_writes_json_and_markdown(traces_dir, tmp_path, capsys):
    outdir = tmp_path / "grade-out"
    code = main(
        [
            "grade",
            "--provider",
            "mock",
            "--traces",
            str(traces_dir / "code_fix.json"),
            "--output",
            str(outdir),
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads((outdir / "grade.json").read_text())
    assert payload["provider"] == "mock"
    assert "tool_ok>=0.6" in payload["jev_accept_rule"].replace(" ", "")
    assert payload["rows"]
    row = payload["rows"][0]
    assert row["trace"] == "code_fix_01"
    for key in ("step", "token_accept", "jev_accept", "tool_ok", "progress", "abort", "task_success"):
        assert key in row
    md = (outdir / "grade.md").read_text()
    assert "| trace | step | token_accept | jev_accept |" in md
    stdout = capsys.readouterr().out
    assert "code_fix_01" in stdout
    assert "wrote" in stdout


def test_grade_cli_jev_without_key_errors(traces_dir, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    code = main(["grade", "--provider", "jev", "--traces", str(traces_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in err
    assert "--provider mock" in err or "mock" in err


def test_grade_trace_token_accept_from_mock_speculative(traces_dir):
    trace = next(t for t in load_traces(traces_dir) if t.id == "travel_planner_01")
    rows = grade_trace(trace, MockJev(), seed=0, gamma=5)
    assert rows
    tool_rows = [r for r in rows if r.step_kind == "tool_call"]
    assert tool_rows
    assert all(r.token_accept is not None for r in tool_rows)
    # Same seed/gamma as bake-off: tool JSON should not all be 1.0.
    assert min(r.token_accept for r in tool_rows) < 1.0


def _looping_trace() -> AgentTrace:
    call = ToolCall(id="c1", name="web_search", arguments={"query": "pop"})
    return AgentTrace(
        id="loop_01",
        task_type="research_qa",
        prompt="Find the population",
        messages=[
            Message(role="user", content="Find the population"),
            Message(role="assistant", tool_calls=[call]),
            Message(role="tool", tool_call_id="c1", name="web_search", content="8M"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c2", name="web_search", arguments={"query": "pop"})],
            ),
            Message(role="tool", tool_call_id="c2", name="web_search", content="8M"),
        ],
        success_criteria=SuccessCriteria(required_tools=("web_search",), final_answer_required=True),
        expected_success=True,
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {"type": "object", "required": ["query"]},
                },
            }
        ],
    )


def test_grade_trace_flags_repeat_and_missing_answer():
    rows = grade_trace(_looping_trace(), MockJev())
    assert len(rows) == 2
    assert rows[0].jev_accept is True
    assert rows[1].jev_accept is False
    assert rows[1].abort >= 0.5
    assert rows[0].task_success is False  # no final answer
    assert rows[1].task_success is False


def test_iter_agent_steps_groups_tool_followups(traces_dir):
    trace = next(t for t in load_traces(traces_dir) if t.id == "code_fix_01")
    steps = iter_agent_steps(trace)
    assert steps
    # First assistant is a tool call with a tool follow-up.
    _idx, _msg_i, assistant, history, followups = steps[0]
    assert assistant.role == "assistant"
    assert assistant.tool_calls
    assert followups and followups[0].role == "tool"
    assert all(m.role != "assistant" or m is assistant for m in history)

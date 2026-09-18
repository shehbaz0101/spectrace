from __future__ import annotations

from spectrace.traces import load_traces


def test_loads_five_synthetic_traces(traces_dir):
    traces = load_traces(traces_dir)
    ids = {t.id for t in traces}
    assert ids == {
        "research_qa_01",
        "code_fix_01",
        "multihop_retrieval_01",
        "travel_planner_01",
        "data_analysis_01",
    }


def test_each_trace_is_multi_step_with_tools(traces_dir):
    for trace in load_traces(traces_dir):
        assert len(trace.messages) >= 6
        assert trace.assistant_turns
        assert trace.tool_names_called()
        assert any(m.role == "tool" for m in trace.messages)
        ok, mode = trace.structural_success()
        assert ok, (trace.id, mode)


def test_code_fix_records_recovered_tool_error(traces_dir):
    trace = next(t for t in load_traces(traces_dir) if t.id == "code_fix_01")
    errors = [m for m in trace.messages if m.error]
    assert errors and errors[0].role == "tool"
    ok, mode = trace.structural_success()
    assert ok and mode is None

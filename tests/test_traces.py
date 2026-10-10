from __future__ import annotations

from spectrace.traces import load_traces

HAPPY_IDS = {
    "research_qa_01",
    "code_fix_01",
    "multihop_retrieval_01",
    "travel_planner_01",
    "data_analysis_01",
}

# id -> structural failure mode the fixture is built to exhibit
FAILURE_IDS = {
    "loop_search_01": "repeated_tool_call",
    "hallucinated_tool_01": "unknown_tool",
    "wrong_args_01": "invalid_arguments",
    "ignore_tool_error_01": "unrecovered_tool_error",
    "off_goal_drift_01": "off_goal",
    "premature_answer_01": "premature_final_answer",
    "destructive_shell_01": "destructive_action",
    "destructive_sql_01": "destructive_action",
    "wrong_doc_01": "wrong_source",
    "stall_churn_01": "stalled",
}


def test_loads_labeled_trace_catalog(traces_dir):
    traces = load_traces(traces_dir)
    ids = {t.id for t in traces}
    assert HAPPY_IDS <= ids
    assert set(FAILURE_IDS) <= ids
    assert len(traces) == len(HAPPY_IDS) + len(FAILURE_IDS)
    assert 12 <= len(traces) <= 20


def test_each_trace_is_multi_step_and_labeled(traces_dir):
    for trace in load_traces(traces_dir):
        assert len(trace.messages) >= 4
        assert trace.assistant_turns
        assert "synthetic" in trace.provenance
        assert trace.task_success is not None
        labels = [m.expected_ok for m in trace.assistant_turns]
        assert all(isinstance(v, bool) for v in labels)
        ok, mode = trace.structural_success()
        assert ok is trace.task_success, (trace.id, ok, trace.task_success, mode)
        if trace.id in HAPPY_IDS:
            assert ok and mode is None
            assert trace.failure_mode is None
            assert all(labels)
            assert trace.tool_names_called()
        if trace.id in FAILURE_IDS:
            assert ok is False
            assert mode == FAILURE_IDS[trace.id] == trace.failure_mode
            assert mode != "expected_failure"
            assert any(v is False for v in labels)


def test_code_fix_records_recovered_tool_error(traces_dir):
    trace = next(t for t in load_traces(traces_dir) if t.id == "code_fix_01")
    errors = [m for m in trace.messages if m.error]
    assert errors and errors[0].role == "tool"
    ok, mode = trace.structural_success()
    assert ok and mode is None

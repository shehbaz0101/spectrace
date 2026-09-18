from __future__ import annotations

from spectrace.metrics import PriceRate
from spectrace.replay import run_trace
from spectrace.traces import AgentTrace, Message, SuccessCriteria, ToolCall


def _failing_trace() -> AgentTrace:
    return AgentTrace(
        id="incomplete_01",
        task_type="research_qa",
        prompt="Find the population",
        messages=[
            Message(role="user", content="Find the population"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "pop"})],
            ),
            Message(role="tool", tool_call_id="c1", name="web_search", content="..."),
        ],
        success_criteria=SuccessCriteria(required_tools=("web_search", "web_fetch"), final_answer_required=True),
        expected_success=True,
    )


def test_missing_tool_and_answer_failure_mode():
    metrics = run_trace(_failing_trace(), "baseline")
    assert metrics.success is False
    assert metrics.failure_mode == "missing_required_tool"


def test_nonzero_cost_with_custom_rate():
    from spectrace.traces import load_trace
    from pathlib import Path

    trace = load_trace(Path(__file__).resolve().parents[1] / "traces" / "code_fix.json")
    rate = PriceRate(input_per_million=10.0, output_per_million=20.0, draft_per_million=1.0)
    metrics = run_trace(trace, "mock_speculative", price_rate=rate)
    assert metrics.cost_estimate_usd > 0

from __future__ import annotations

import random

from spectrace.decoding import (
    P_ACCEPT_TEXT,
    P_ACCEPT_TOOL,
    decode_assistant_text,
    stable_seed,
)
from spectrace.metrics import accept_rate, estimate_cost, load_price_table, lookup_rate
from spectrace.replay import run_bakeoff, run_trace
from spectrace.traces import load_traces


def test_accept_rate_none_when_no_drafts():
    assert accept_rate(0, 0) is None


def test_default_cost_is_zero():
    assert estimate_cost(tokens_in=10_000, tokens_out=2_000, draft_tokens_proposed=8_000) == 0.0


def test_price_table_charges_draft_tokens():
    table = load_price_table(
        {
            "example-target": {
                "input_per_million": 1.0,
                "output_per_million": 2.0,
                "draft_per_million": 4.0,
            }
        }
    )
    rate = lookup_rate(table, "example-target")
    cost = estimate_cost(
        tokens_in=1_000_000,
        tokens_out=1_000_000,
        draft_tokens_proposed=1_000_000,
        rate=rate,
    )
    assert cost == 7.0


def test_tool_call_drafts_accepted_less_than_prose():
    prose = "The median order value in North America is higher than EMEA and APAC. " * 8
    tool = '{"name":"check_constraints","arguments":{"plan":{"stay":"Alfama"},"constraints":{"no_car":true}}}' * 4
    assert P_ACCEPT_TOOL[0] < P_ACCEPT_TEXT[0]
    text_rates = []
    tool_rates = []
    for seed in range(20):
        text_step = decode_assistant_text(prose, method="mock_speculative", step_kind="text", seed=seed)
        tool_step = decode_assistant_text(tool, method="mock_speculative", step_kind="tool_call", seed=seed)
        text_rates.append(text_step.draft_accepted / text_step.draft_proposed)
        tool_rates.append(tool_step.draft_accepted / tool_step.draft_proposed)
    assert sum(tool_rates) / len(tool_rates) < sum(text_rates) / len(text_rates)


def test_speculative_is_deterministic():
    text = '{"query":"Tokyo population 2026"}'
    a = decode_assistant_text(text, method="mock_speculative", step_kind="tool_call", seed=42)
    b = decode_assistant_text(text, method="mock_speculative", step_kind="tool_call", seed=42)
    assert a == b
    c = decode_assistant_text(text, method="mock_speculative", step_kind="tool_call", seed=43)
    assert c != a


def test_stable_seed_ignores_hash_randomization():
    assert stable_seed(0, "code_fix_01", "baseline") == stable_seed(0, "code_fix_01", "baseline")


def test_baseline_has_no_accept_rate(traces_dir):
    trace = load_traces(traces_dir)[0]
    metrics = run_trace(trace, "baseline")
    assert metrics.accept_rate is None
    assert metrics.draft_tokens_proposed == 0
    assert metrics.success is True
    assert metrics.cost_estimate_usd == 0.0


def test_mock_speculative_records_accept_rate(traces_dir):
    trace = load_traces(traces_dir)[0]
    metrics = run_trace(trace, "mock_speculative", seed=0)
    assert metrics.draft_tokens_proposed > 0
    assert 0.0 <= metrics.accept_rate <= 1.0
    assert metrics.wall_latency_ms > 0


def test_speculative_faster_than_baseline_on_fixtures(traces_dir):
    for trace in load_traces(traces_dir):
        base = run_trace(trace, "baseline")
        spec = run_trace(trace, "mock_speculative")
        assert spec.wall_latency_ms < base.wall_latency_ms, trace.id
        assert spec.tokens_in == base.tokens_in
        assert spec.tokens_out == base.tokens_out


def test_bakeoff_aggregates_cost_per_success(traces_dir):
    summary = run_bakeoff(str(traces_dir), ["baseline", "mock_speculative"])
    assert summary.n_success == len(summary.runs) == 10
    assert summary.cost_per_successful_task_usd == 0.0
    methods = {r.method for r in summary.runs}
    assert methods == {"baseline", "mock_speculative"}


def test_random_module_used_only_with_local_rng():
    """Guard: decoder must not call random.random() on the module RNG."""
    random.seed(123)
    ahead = [random.random() for _ in range(5)]
    decode_assistant_text("hello world " * 20, method="mock_speculative", step_kind="text", seed=1)
    random.seed(123)
    assert [random.random() for _ in range(5)] == ahead

"""Replay an agent trace through a serving method and collect metrics.

The default path does not generate new tokens. It walks the recorded
assistant turns, charges growing prefix context as tokens_in (the agent
tax vs. single-turn chat), and asks the chosen decoder to simulate
how those tokens would have been served.
"""

from __future__ import annotations

from spectrace.decoding import METHODS, decode_assistant_text, stable_seed
from spectrace.metrics import (
    PriceRate,
    RunMetrics,
    accept_rate,
    aggregate,
    count_tokens,
    estimate_cost,
    lookup_rate,
)
from spectrace.providers import ProviderSpec, mock_provider
from spectrace.traces import AgentTrace, load_traces


def _context_text(trace: AgentTrace, upto: int) -> str:
    parts: list[str] = []
    for msg in trace.messages[:upto]:
        parts.append(f"{msg.role}: {msg.text_for_tokens()}")
    return "\n".join(parts)


def run_trace(
    trace: AgentTrace,
    method: str,
    *,
    seed: int = 0,
    gamma: int = 5,
    provider: ProviderSpec | None = None,
    price_rate: PriceRate | None = None,
) -> RunMetrics:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method!r}")
    provider = provider or mock_provider()
    price_rate = price_rate or PriceRate()

    tokens_in = 0
    tokens_out = 0
    draft_proposed = 0
    draft_accepted = 0
    rounds = 0
    latency = 0.0
    n_gen = 0

    for idx, msg in enumerate(trace.messages):
        if msg.role != "assistant":
            continue
        n_gen += 1
        text = msg.text_for_tokens()
        tokens_in += count_tokens(_context_text(trace, idx))
        step_seed = stable_seed(seed, trace.id, method, str(idx), msg.step_kind)
        step = decode_assistant_text(
            text,
            method=method,
            step_kind=msg.step_kind,
            seed=step_seed,
            gamma=gamma,
        )
        tokens_out += step.tokens_out
        draft_proposed += step.draft_proposed
        draft_accepted += step.draft_accepted
        rounds += step.rounds
        latency += step.wall_latency_ms

    success, failure_mode = trace.structural_success()
    cost = estimate_cost(
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        draft_tokens_proposed=draft_proposed,
        rate=price_rate,
    )
    return RunMetrics(
        trace_id=trace.id,
        method=method,
        wall_latency_ms=latency,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        draft_tokens_proposed=draft_proposed,
        draft_tokens_accepted=draft_accepted,
        accept_rate=accept_rate(draft_proposed, draft_accepted),
        success=success,
        cost_estimate_usd=cost,
        failure_mode=failure_mode,
        n_generation_steps=n_gen,
        n_tool_calls=len(trace.tool_names_called()),
        n_verify_rounds=rounds if method == "mock_speculative" else 0,
        provider=provider.name,
        notes="simulated replay of fixture tokens; not a live model call",
    )


def run_bakeoff(
    traces: list[AgentTrace] | str,
    methods: list[str] | None = None,
    *,
    seed: int = 0,
    gamma: int = 5,
    provider: ProviderSpec | None = None,
    price_table: dict[str, PriceRate] | None = None,
    model: str = "mock-local",
):
    if isinstance(traces, str):
        traces = load_traces(traces)
    methods = methods or list(METHODS)
    rate = lookup_rate(price_table or {}, model)
    runs = []
    for trace in traces:
        for method in methods:
            runs.append(
                run_trace(
                    trace,
                    method,
                    seed=seed,
                    gamma=gamma,
                    provider=provider,
                    price_rate=rate,
                )
            )
    return aggregate(runs)

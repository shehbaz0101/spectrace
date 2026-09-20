"""Replay an agent trace through a serving method and collect metrics.

The default path does not generate new tokens. It walks the recorded
assistant turns, charges growing prefix context as tokens_in (the agent
tax vs. single-turn chat), and asks the chosen decoder to simulate
how those tokens would have been served.

When ``--provider openai-compat`` is selected *and* ``SPECTRACE_BASE_URL``
is set, each recorded assistant prefix is also sent to an OpenAI-compatible
``chat/completions`` endpoint. Live wall time (and usage tokens when the
server reports them) overlay the replayed fixture metrics. Task success
stays structural on the recorded trajectory. Jev is not called here.
"""

from __future__ import annotations

from typing import Sequence

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
from spectrace.openai_compat import ChatCompletionResult, OpenAICompatClient
from spectrace.providers import ProviderSpec, is_live_provider, mock_provider
from spectrace.traces import AgentTrace, load_traces

LIVE_NOTES = (
    "live OpenAI-compat draft/serve on recorded prefixes; "
    "task success is the fixture trajectory; "
    "accept_rate is still the local mock decoder (Jev is grade-only)"
)
MOCK_NOTES = "simulated replay of fixture tokens; not a live model call"


def _context_text(trace: AgentTrace, upto: int) -> str:
    parts: list[str] = []
    for msg in trace.messages[:upto]:
        parts.append(f"{msg.role}: {msg.text_for_tokens()}")
    return "\n".join(parts)


def _assistant_indexes(trace: AgentTrace) -> list[int]:
    return [idx for idx, msg in enumerate(trace.messages) if msg.role == "assistant"]


def collect_live_steps(
    trace: AgentTrace,
    client: OpenAICompatClient,
    *,
    max_tokens: int | None = None,
) -> list[ChatCompletionResult]:
    """One live draft/serve call per recorded assistant turn (prefix only)."""
    return [
        client.serve_recorded_step(trace, idx, max_tokens=max_tokens)
        for idx in _assistant_indexes(trace)
    ]


def _overlay_live(
    *,
    recorded_tokens_in: int,
    recorded_tokens_out: int,
    recorded_latency: float,
    live_results: Sequence[ChatCompletionResult],
) -> tuple[int, int, float, dict]:
    live_lat = sum(r.latency_ms for r in live_results)
    live_in_parts = [r.prompt_tokens for r in live_results]
    live_out_parts = [r.completion_tokens for r in live_results]
    have_in = bool(live_in_parts) and all(x is not None for x in live_in_parts)
    have_out = bool(live_out_parts) and all(x is not None for x in live_out_parts)
    tokens_in = int(sum(live_in_parts)) if have_in else recorded_tokens_in  # type: ignore[arg-type]
    tokens_out = int(sum(live_out_parts)) if have_out else recorded_tokens_out  # type: ignore[arg-type]
    extra = {
        "live": True,
        "live_latency_ms": round(live_lat, 2),
        "live_prompt_tokens": int(sum(live_in_parts)) if have_in else None,  # type: ignore[arg-type]
        "live_completion_tokens": int(sum(live_out_parts)) if have_out else None,  # type: ignore[arg-type]
        "fixture_tokens_in": recorded_tokens_in,
        "fixture_tokens_out": recorded_tokens_out,
        "fixture_latency_ms": round(recorded_latency, 2),
        "live_steps": len(live_results),
    }
    return tokens_in, tokens_out, live_lat, extra


def run_trace(
    trace: AgentTrace,
    method: str,
    *,
    seed: int = 0,
    gamma: int = 5,
    provider: ProviderSpec | None = None,
    price_rate: PriceRate | None = None,
    client: OpenAICompatClient | None = None,
    live_steps: Sequence[ChatCompletionResult] | None = None,
    max_tokens: int | None = None,
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

    extra: dict = {}
    notes = MOCK_NOTES
    if is_live_provider(provider):
        if live_steps is None:
            live_client = client or OpenAICompatClient.from_spec(provider)
            live_steps = collect_live_steps(trace, live_client, max_tokens=max_tokens)
        if len(live_steps) != n_gen:
            raise ValueError(
                f"live_steps length {len(live_steps)} != assistant turns {n_gen}"
            )
        tokens_in, tokens_out, latency, extra = _overlay_live(
            recorded_tokens_in=tokens_in,
            recorded_tokens_out=tokens_out,
            recorded_latency=latency,
            live_results=live_steps,
        )
        notes = LIVE_NOTES

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
        notes=notes,
        extra=extra,
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
    client: OpenAICompatClient | None = None,
    max_tokens: int | None = None,
):
    if isinstance(traces, str):
        traces = load_traces(traces)
    methods = methods or list(METHODS)
    rate = lookup_rate(price_table or {}, model)
    provider = provider or mock_provider()
    live_client = client
    if is_live_provider(provider) and live_client is None:
        live_client = OpenAICompatClient.from_spec(provider)

    # One live pass per trace, reused across methods so bakeoff does not
    # double-call a paid or local server.
    live_cache: dict[str, list[ChatCompletionResult]] = {}
    if live_client is not None and is_live_provider(provider):
        for trace in traces:
            live_cache[trace.id] = collect_live_steps(
                trace, live_client, max_tokens=max_tokens
            )

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
                    client=live_client,
                    live_steps=live_cache.get(trace.id),
                    max_tokens=max_tokens,
                )
            )
    summary = aggregate(runs)
    if is_live_provider(provider):
        summary.notes = (
            "LIVE OpenAI-compat draft/serve — wall latency (and usage tokens "
            "when reported) from chat/completions. Task success is the recorded "
            "fixture. accept_rate is still the mock decoder. Jev is not used here."
        )
    return summary

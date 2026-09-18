"""Token accounting, wall-latency simulation, and cost estimates.

Default prices are $0 so demos and pytest need no vendor account.
Draft-token cost is modeled separately: speculative decoding does extra
draft-model work even when the billed target tokens stay the same.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Sequence


def count_tokens(text: str) -> int:
    """Cheap, deterministic stand-in for a tokenizer (~4 chars / token)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def chunk_tokens(text: str, width: int = 4) -> list[str]:
    if not text:
        return []
    return [text[i : i + width] for i in range(0, len(text), width)]


@dataclass(frozen=True)
class PriceRate:
    """USD per million tokens."""

    input_per_million: float = 0.0
    output_per_million: float = 0.0
    draft_per_million: float = 0.0


DEFAULT_PRICE_TABLE: dict[str, PriceRate] = {
    "mock-local": PriceRate(),
}


def usd_from_tokens(n_tokens: int, usd_per_million: float) -> float:
    return (n_tokens / 1_000_000.0) * usd_per_million


def estimate_cost(
    *,
    tokens_in: int,
    tokens_out: int,
    draft_tokens_proposed: int = 0,
    rate: PriceRate | None = None,
) -> float:
    rate = rate or PriceRate()
    return (
        usd_from_tokens(tokens_in, rate.input_per_million)
        + usd_from_tokens(tokens_out, rate.output_per_million)
        + usd_from_tokens(draft_tokens_proposed, rate.draft_per_million)
    )


@dataclass
class RunMetrics:
    trace_id: str
    method: str
    wall_latency_ms: float
    tokens_in: int
    tokens_out: int
    draft_tokens_proposed: int
    draft_tokens_accepted: int
    accept_rate: float | None
    success: bool
    cost_estimate_usd: float
    failure_mode: str | None
    n_generation_steps: int = 0
    n_tool_calls: int = 0
    n_verify_rounds: int = 0
    provider: str = "mock-local"
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        if self.accept_rate is not None:
            row["accept_rate"] = round(self.accept_rate, 4)
        row["wall_latency_ms"] = round(self.wall_latency_ms, 2)
        row["cost_estimate_usd"] = round(self.cost_estimate_usd, 6)
        return row


@dataclass
class BakeoffSummary:
    runs: list[RunMetrics]
    n_success: int
    cost_per_successful_task_usd: float
    notes: str = "FIXTURE / SIMULATED — not measured on GPUs."

    def to_dict(self) -> dict[str, Any]:
        return {
            "notes": self.notes,
            "n_runs": len(self.runs),
            "n_success": self.n_success,
            "cost_per_successful_task_usd": round(self.cost_per_successful_task_usd, 6),
            "runs": [r.to_dict() for r in self.runs],
        }


def accept_rate(proposed: int, accepted: int) -> float | None:
    if proposed <= 0:
        return None
    return accepted / proposed


def aggregate(runs: Sequence[RunMetrics]) -> BakeoffSummary:
    successes = [r for r in runs if r.success]
    n_success = len(successes)
    if n_success == 0:
        cost = float("nan")
    else:
        # Cost *per successful task* uses only successful runs so a failing
        # method cannot look cheaper by aborting early.
        cost = sum(r.cost_estimate_usd for r in successes) / n_success
    return BakeoffSummary(runs=list(runs), n_success=n_success, cost_per_successful_task_usd=cost)


def load_price_table(raw: dict[str, Any] | None) -> dict[str, PriceRate]:
    if not raw:
        return dict(DEFAULT_PRICE_TABLE)
    table: dict[str, PriceRate] = dict(DEFAULT_PRICE_TABLE)
    for name, vals in raw.items():
        if not isinstance(vals, dict):
            continue
        table[name] = PriceRate(
            input_per_million=float(vals.get("input_per_million", 0.0)),
            output_per_million=float(vals.get("output_per_million", 0.0)),
            draft_per_million=float(vals.get("draft_per_million", 0.0)),
        )
    return table


def lookup_rate(table: dict[str, PriceRate], model: str) -> PriceRate:
    return table.get(model) or table.get("mock-local") or PriceRate()

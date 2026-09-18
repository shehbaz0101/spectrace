"""Serving methods: sequential baseline vs mock speculative decoding.

The mock decoder never touches a GPU. It replays recorded assistant text
and *simulates* Leviathan-style draft/verify (gamma tokens per round,
longest accepted prefix, plus one guaranteed target token).

Acceptance probabilities are lower on tool-call JSON than on free text.
That is the point of this harness: ShareGPT chat is a poor proxy for
agent traces, where structured tool arguments are common and brittle.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from spectrace.metrics import chunk_tokens, count_tokens

METHODS = ("baseline", "mock_speculative")

# Simulated millisecond costs — *not* hardware measurements.
TARGET_MS_PER_TOKEN = 6.0
DRAFT_MS_PER_TOKEN = 1.2
VERIFY_MS_PER_ROUND = 6.0  # one target forward pass over the draft window
STEP_OVERHEAD_MS = 4.0  # orchestration / tool-message bookkeeping

# Per-position P(accept this draft token | previous accepted). Length = gamma.
# Tool-call JSON drifts faster (schema tokens, quotes, argument keys).
P_ACCEPT_TEXT = (0.92, 0.88, 0.84, 0.80, 0.76)
P_ACCEPT_TOOL = (0.78, 0.62, 0.48, 0.36, 0.28)
P_ACCEPT_MIXED = (0.85, 0.74, 0.62, 0.52, 0.42)

DEFAULT_GAMMA = 5


def stable_seed(base: int, *parts: str) -> int:
    material = f"{base}|" + "|".join(parts)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _p_schedule(step_kind: str, gamma: int) -> tuple[float, ...]:
    src = {
        "tool_call": P_ACCEPT_TOOL,
        "mixed": P_ACCEPT_MIXED,
    }.get(step_kind, P_ACCEPT_TEXT)
    if gamma <= len(src):
        return src[:gamma]
    extra = tuple(src[-1] for _ in range(gamma - len(src)))
    return src + extra


@dataclass
class DecodeStep:
    step_kind: str
    tokens_out: int
    draft_proposed: int
    draft_accepted: int
    rounds: int
    wall_latency_ms: float


@dataclass
class DecodeTrace:
    method: str
    steps: list[DecodeStep]
    tokens_in: int
    tokens_out: int
    draft_proposed: int
    draft_accepted: int
    rounds: int
    wall_latency_ms: float


def _simulate_speculative_text(
    text: str,
    *,
    rng: random.Random,
    gamma: int,
    step_kind: str,
) -> DecodeStep:
    tokens = chunk_tokens(text)
    n = len(tokens)
    if n == 0:
        return DecodeStep(step_kind, 0, 0, 0, 0, STEP_OVERHEAD_MS)

    p = _p_schedule(step_kind, gamma)
    remaining = n
    proposed = 0
    accepted = 0
    rounds = 0
    latency = STEP_OVERHEAD_MS

    while remaining > 0:
        gamma_i = min(gamma, remaining)
        proposed += gamma_i
        prefix = 0
        for j in range(gamma_i):
            if rng.random() < p[j]:
                prefix += 1
            else:
                break
        accepted += prefix
        # Target always contributes the next token (bonus or correction)
        # unless the window already consumed the remaining sequence.
        produced = min(prefix + 1, remaining)
        remaining -= produced
        rounds += 1
        latency += gamma_i * DRAFT_MS_PER_TOKEN + VERIFY_MS_PER_ROUND

    return DecodeStep(
        step_kind=step_kind,
        tokens_out=n,
        draft_proposed=proposed,
        draft_accepted=accepted,
        rounds=rounds,
        wall_latency_ms=latency,
    )


def _simulate_baseline_text(text: str, step_kind: str) -> DecodeStep:
    n = count_tokens(text)
    latency = STEP_OVERHEAD_MS + n * TARGET_MS_PER_TOKEN
    return DecodeStep(
        step_kind=step_kind,
        tokens_out=n,
        draft_proposed=0,
        draft_accepted=0,
        rounds=n,
        wall_latency_ms=latency,
    )


def decode_assistant_text(
    text: str,
    *,
    method: str,
    step_kind: str,
    seed: int,
    gamma: int = DEFAULT_GAMMA,
) -> DecodeStep:
    if method == "baseline":
        return _simulate_baseline_text(text, step_kind)
    if method == "mock_speculative":
        rng = random.Random(seed)
        return _simulate_speculative_text(text, rng=rng, gamma=gamma, step_kind=step_kind)
    raise ValueError(f"unknown method: {method!r} (expected one of {METHODS})")

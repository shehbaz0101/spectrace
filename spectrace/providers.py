"""Optional model providers.

Default is a local mock that replays the recorded trace — no network, no keys.
An OpenAI-compatible path is sketched for later (`SPECTRACE_BASE_URL`) but is
never used unless explicitly selected. Tests stay on the mock.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class ProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderSpec:
    name: str
    model: str
    base_url: str | None = None
    requires_key: bool = False


def mock_provider() -> ProviderSpec:
    return ProviderSpec(name="mock-local", model="mock-local")


def from_env() -> ProviderSpec:
    """Live OpenAI-compatible endpoint. Unused unless SPECTRACE_BASE_URL is set."""
    base = os.environ.get("SPECTRACE_BASE_URL", "").strip()
    model = os.environ.get("SPECTRACE_MODEL", "").strip() or "openai-compat"
    if not base:
        return mock_provider()
    return ProviderSpec(
        name="openai-compat",
        model=model,
        base_url=base.rstrip("/"),
        requires_key=True,
    )


def resolve_provider(name: str | None) -> ProviderSpec:
    if not name or name in {"mock", "mock-local", "default"}:
        return mock_provider()
    if name in {"openai", "openai-compat", "live"}:
        spec = from_env()
        if spec.name != "openai-compat":
            raise ProviderError(
                "live provider requested but SPECTRACE_BASE_URL is unset; "
                "the default path is mock-local (offline, $0)."
            )
        return spec
    raise ProviderError(f"unknown provider: {name}")

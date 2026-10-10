"""Optional model providers.

Default is a local mock that replays the recorded trace — no network, no keys.
The OpenAI-compatible path (``SPECTRACE_BASE_URL``) is the System-2
draft/serve side. It is never used unless explicitly selected via
``--provider openai-compat``. TypeSafe Jev is not a provider here; Jev
only grades (``spectrace grade``). Tests stay on the mock.
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
    api_key: str | None = None


def mock_provider() -> ProviderSpec:
    return ProviderSpec(name="mock-local", model="mock-local")


def api_key_from_env() -> str:
    return os.environ.get("SPECTRACE_API_KEY", "").strip()


def missing_base_url_message() -> str:
    return (
        "live provider requested but SPECTRACE_BASE_URL is unset; "
        "the default path is mock-local (offline, $0). "
        "OpenAI-compat drafts/serves; Jev (`spectrace grade`) only accepts."
    )


def from_env() -> ProviderSpec:
    """Live OpenAI-compatible endpoint. Unused unless SPECTRACE_BASE_URL is set."""
    base = os.environ.get("SPECTRACE_BASE_URL", "").strip()
    model = os.environ.get("SPECTRACE_MODEL", "").strip() or "openai-compat"
    if not base:
        return mock_provider()
    key = api_key_from_env()
    return ProviderSpec(
        name="openai-compat",
        model=model,
        base_url=base.rstrip("/"),
        requires_key=False,
        api_key=key or None,
    )


def resolve_provider(name: str | None) -> ProviderSpec:
    if not name or name in {"mock", "mock-local", "default"}:
        return mock_provider()
    if name in {"openai", "openai-compat", "live"}:
        spec = from_env()
        if spec.name != "openai-compat":
            raise ProviderError(missing_base_url_message())
        return spec
    raise ProviderError(f"unknown provider: {name}")


def is_live_provider(spec: ProviderSpec | None) -> bool:
    return bool(spec and spec.name == "openai-compat" and spec.base_url)

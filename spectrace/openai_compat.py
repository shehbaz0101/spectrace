"""OpenAI-compatible chat/completions client (System-2 draft / serve only).

Used solely when the caller explicitly selects ``--provider openai-compat``
*and* ``SPECTRACE_BASE_URL`` is set. Default bake-off / grade paths never
import this for I/O.

This module generates assistant tokens. TypeSafe Jev does not — Jev stays
the System-1 acceptor (``spectrace grade``).
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable

from spectrace import __version__
from spectrace.metrics import count_tokens
from spectrace.providers import (
    ProviderSpec,
    api_key_from_env,
    missing_base_url_message,
)
from spectrace.traces import AgentTrace, Message

DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_MAX_TOKENS = 256
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


class OpenAICompatError(RuntimeError):
    """Base error for the live OpenAI-compatible draft/serve path."""


class OpenAICompatConfigError(OpenAICompatError):
    """Missing ``SPECTRACE_BASE_URL`` or other local configuration problem."""


class OpenAICompatAPIError(OpenAICompatError):
    """HTTP / parse failure talking to an OpenAI-compatible server."""


@dataclass
class ChatCompletionResult:
    content: str
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str | None = None
    n_attempts: int = 1


def completions_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise OpenAICompatConfigError(missing_base_url_message())
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def message_to_openai(msg: Message) -> dict[str, Any]:
    """Convert a recorded spectrace message to an OpenAI chat message."""
    if msg.role == "tool":
        row: dict[str, Any] = {
            "role": "tool",
            "tool_call_id": msg.tool_call_id or msg.name or "",
            "content": msg.content or "",
        }
        if msg.name:
            row["name"] = msg.name
        return row
    row = {"role": msg.role, "content": msg.content or ""}
    if msg.role == "assistant" and msg.tool_calls:
        row["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.arguments_json(),
                },
            }
            for call in msg.tool_calls
        ]
        if not (msg.content or "").strip():
            row["content"] = None
    return row


def prefix_messages(trace: AgentTrace, upto: int) -> list[dict[str, Any]]:
    return [message_to_openai(m) for m in trace.messages[:upto]]


def _opt_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_chat_completion(payload: Any) -> ChatCompletionResult:
    """Extract text / tool calls / usage from a chat.completions JSON body."""
    if not isinstance(payload, dict):
        raise OpenAICompatAPIError("OpenAI-compat response is not a JSON object")
    choices = payload.get("choices") or []
    if not isinstance(choices, list) or not choices:
        raise OpenAICompatAPIError("OpenAI-compat response missing choices[0]")
    first = choices[0] if isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    if content is None:
        content = ""
    elif not isinstance(content, str):
        content = json.dumps(content)
    raw_calls = message.get("tool_calls") or []
    tool_calls = [c for c in raw_calls if isinstance(c, dict)] if isinstance(raw_calls, list) else []
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    prompt = _opt_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _opt_int(usage.get("completion_tokens", usage.get("output_tokens")))
    finish = first.get("finish_reason")
    return ChatCompletionResult(
        content=content,
        tool_calls=tool_calls,
        prompt_tokens=prompt,
        completion_tokens=completion,
        model=str(payload.get("model") or ""),
        finish_reason=str(finish) if finish is not None else None,
    )


def _backoff_s(attempt: int) -> float:
    return 0.4 * (2**attempt)


def _auto_max_tokens(recorded_text: str, override: int | None) -> int:
    if override is not None and override > 0:
        return int(override)
    n = count_tokens(recorded_text)
    return max(64, min(DEFAULT_MAX_TOKENS, max(n * 2, 64)))


class OpenAICompatClient:
    """POST ``{base}/chat/completions`` with timeout + retry.

    Instantiating does not perform I/O. Default bake-off never constructs this
    unless ``--provider openai-compat`` is set.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self.base_url = (base_url or "").strip().rstrip("/")
        if not self.base_url:
            raise OpenAICompatConfigError(missing_base_url_message())
        self.model = model or "openai-compat"
        self.api_key = (api_key if api_key is not None else api_key_from_env()).strip()
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        self.endpoint = completions_url(self.base_url)
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep or time.sleep
        self._monotonic = monotonic or time.perf_counter

    @classmethod
    def from_spec(cls, spec: ProviderSpec, **kwargs: Any) -> "OpenAICompatClient":
        if spec.name != "openai-compat" or not spec.base_url:
            raise OpenAICompatConfigError(missing_base_url_message())
        return cls(
            base_url=spec.base_url,
            model=spec.model,
            api_key=spec.api_key,
            **kwargs,
        )

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
    ) -> ChatCompletionResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": f"spectrace/{__version__}",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers=headers,
        )
        last_error: Exception | None = None
        started = self._monotonic()
        for attempt in range(self.max_attempts):
            try:
                with self._urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                try:
                    parsed = json.loads(text) if text else {}
                except json.JSONDecodeError as exc:
                    raise OpenAICompatAPIError(f"OpenAI-compat returned non-JSON body: {exc}") from exc
                result = parse_chat_completion(parsed)
                result.latency_ms = (self._monotonic() - started) * 1000.0
                result.n_attempts = attempt + 1
                if not result.model:
                    result.model = self.model
                return result
            except OpenAICompatAPIError:
                raise
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    err_body = ""
                last_error = OpenAICompatAPIError(
                    f"OpenAI-compat HTTP {exc.code}: {err_body or exc.reason}"
                )
                if exc.code in RETRY_STATUS and attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = OpenAICompatAPIError(f"OpenAI-compat network error: {exc.reason}")
                if attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
            except TimeoutError as exc:
                last_error = OpenAICompatAPIError(
                    f"OpenAI-compat timed out after {self.timeout}s"
                )
                if attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
        raise last_error or OpenAICompatAPIError("OpenAI-compat request failed")

    def serve_recorded_step(
        self,
        trace: AgentTrace,
        message_index: int,
        *,
        max_tokens: int | None = None,
    ) -> ChatCompletionResult:
        """Draft/serve the next assistant turn from the recorded prefix.

        Sends ``messages[:message_index]`` (everything *before* the recorded
        assistant step). Does not replace the fixture trajectory or ask Jev
        to generate text.
        """
        if message_index < 0 or message_index >= len(trace.messages):
            raise OpenAICompatError(f"message_index {message_index} out of range")
        recorded = trace.messages[message_index]
        return self.chat_completions(
            prefix_messages(trace, message_index),
            tools=list(trace.tools) or None,
            max_tokens=_auto_max_tokens(recorded.text_for_tokens(), max_tokens),
        )

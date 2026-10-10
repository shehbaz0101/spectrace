from __future__ import annotations

import io
import json
import os
import urllib.error
from email.message import Message

import pytest

from spectrace.cli import main
from spectrace.openai_compat import (
    ChatCompletionResult,
    OpenAICompatAPIError,
    OpenAICompatClient,
    OpenAICompatConfigError,
    completions_url,
    message_to_openai,
    parse_chat_completion,
)
from spectrace.providers import ProviderError, ProviderSpec, from_env, resolve_provider
from spectrace.replay import run_bakeoff, run_trace
from spectrace.traces import Message as TraceMessage
from spectrace.traces import ToolCall, load_traces


class _FakeHTTP:
    def __init__(self, payload: dict, status: int = 200):
        self._payload = json.dumps(payload).encode("utf-8")
        self.status = status

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _ok_payload(**overrides):
    body = {
        "id": "chatcmpl-test",
        "model": "local-model",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "pong"},
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
    }
    body.update(overrides)
    return body


def test_from_env_without_url_is_mock(monkeypatch):
    monkeypatch.delenv("SPECTRACE_BASE_URL", raising=False)
    monkeypatch.delenv("SPECTRACE_API_KEY", raising=False)
    spec = from_env()
    assert spec.name == "mock-local"
    assert spec.base_url is None


def test_from_env_with_url(monkeypatch):
    monkeypatch.setenv("SPECTRACE_BASE_URL", "http://127.0.0.1:11434/v1/")
    monkeypatch.setenv("SPECTRACE_MODEL", "llama")
    monkeypatch.setenv("SPECTRACE_API_KEY", "sk-test-not-real")
    spec = from_env()
    assert spec.name == "openai-compat"
    assert spec.base_url == "http://127.0.0.1:11434/v1"
    assert spec.model == "llama"
    assert spec.api_key == "sk-test-not-real"
    assert spec.requires_key is False


def test_resolve_live_requires_base_url(monkeypatch):
    monkeypatch.delenv("SPECTRACE_BASE_URL", raising=False)
    with pytest.raises(ProviderError, match="SPECTRACE_BASE_URL"):
        resolve_provider("openai-compat")
    with pytest.raises(ProviderError, match="grade"):
        resolve_provider("live")
    assert resolve_provider("mock-local").name == "mock-local"
    assert resolve_provider(None).name == "mock-local"


def test_client_refuses_empty_base_url():
    with pytest.raises(OpenAICompatConfigError, match="SPECTRACE_BASE_URL"):
        OpenAICompatClient(base_url="", model="x")


def test_completions_url_normalizes():
    assert completions_url("http://localhost:8000/v1") == "http://localhost:8000/v1/chat/completions"
    assert (
        completions_url("http://localhost:8000/v1/chat/completions")
        == "http://localhost:8000/v1/chat/completions"
    )


def test_message_to_openai_tool_call():
    msg = TraceMessage(
        role="assistant",
        content="",
        tool_calls=[ToolCall(id="c1", name="web_search", arguments={"query": "pop"})],
    )
    row = message_to_openai(msg)
    assert row["role"] == "assistant"
    assert row["content"] is None
    assert row["tool_calls"][0]["function"]["name"] == "web_search"
    assert json.loads(row["tool_calls"][0]["function"]["arguments"]) == {"query": "pop"}

    tool = TraceMessage(role="tool", content="8M", tool_call_id="c1", name="web_search")
    trow = message_to_openai(tool)
    assert trow == {"role": "tool", "tool_call_id": "c1", "content": "8M", "name": "web_search"}


def test_parse_chat_completion_usage_aliases():
    parsed = parse_chat_completion(
        {
            "choices": [{"message": {"content": "hi", "tool_calls": [{"id": "t1"}]}}],
            "usage": {"input_tokens": 3, "output_tokens": 2},
        }
    )
    assert parsed.content == "hi"
    assert parsed.prompt_tokens == 3
    assert parsed.completion_tokens == 2
    assert parsed.tool_calls == [{"id": "t1"}]


def test_client_posts_and_parses(monkeypatch):
    monkeypatch.setenv("SPECTRACE_API_KEY", "test-key-not-real")
    seen: dict[str, object] = {}
    clock = {"t": 1.0}

    def fake_now():
        clock["t"] += 0.05
        return clock["t"]

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["method"] = request.get_method()
        seen["auth"] = request.headers.get("Authorization") or request.get_header("Authorization")
        seen["body"] = json.loads(request.data.decode("utf-8"))
        return _FakeHTTP(_ok_payload())

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="local-model",
        urlopen=fake_urlopen,
        sleep=lambda _s: None,
        monotonic=fake_now,
    )
    result = client.chat_completions(
        [{"role": "user", "content": "ping"}],
        tools=[{"type": "function", "function": {"name": "web_search"}}],
        max_tokens=32,
    )
    assert seen["url"] == "http://127.0.0.1:9/v1/chat/completions"
    assert seen["method"] == "POST"
    assert seen["timeout"] == 30.0
    auth = str(seen["auth"])
    assert auth.startswith("Bearer")
    assert auth.endswith("test-key-not-real")
    body = seen["body"]
    assert body["model"] == "local-model"
    assert body["stream"] is False
    assert body["messages"][0]["content"] == "ping"
    assert body["tools"][0]["function"]["name"] == "web_search"
    assert body["tool_choice"] == "auto"
    assert result.content == "pong"
    assert result.prompt_tokens == 12
    assert result.completion_tokens == 4
    assert result.latency_ms == pytest.approx(50.0, abs=1.0)


def test_client_omits_authorization_without_key(monkeypatch):
    monkeypatch.delenv("SPECTRACE_API_KEY", raising=False)
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        seen["headers"] = dict(request.header_items())
        return _FakeHTTP(_ok_payload())

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        api_key="",
        urlopen=fake_urlopen,
        sleep=lambda _s: None,
    )
    client.chat_completions([{"role": "user", "content": "x"}])
    joined = " ".join(f"{k}:{v}" for k, v in seen["headers"].items())
    assert "Authorization" not in joined
    assert "Bearer" not in joined


def test_client_retries_then_succeeds():
    calls = {"n": 0}
    sleeps: list[float] = []

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise urllib.error.HTTPError(
                request.full_url,
                429,
                "rate limited",
                Message(),
                io.BytesIO(b'{"error":"slow down"}'),
            )
        return _FakeHTTP(_ok_payload())

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        urlopen=fake_urlopen,
        sleep=sleeps.append,
        max_attempts=3,
    )
    result = client.chat_completions([{"role": "user", "content": "x"}])
    assert calls["n"] == 3
    assert sleeps == [0.4, 0.8]
    assert result.content == "pong"
    assert result.n_attempts == 3


def test_client_http_401_does_not_retry():
    calls = {"n": 0}

    def fake_urlopen(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            401,
            "unauthorized",
            Message(),
            io.BytesIO(b'{"error":"bad key"}'),
        )

    client = OpenAICompatClient(
        base_url="http://127.0.0.1:9/v1",
        model="m",
        api_key="bad",
        urlopen=fake_urlopen,
        sleep=lambda _s: None,
    )
    with pytest.raises(OpenAICompatAPIError, match="401"):
        client.chat_completions([{"role": "user", "content": "x"}])
    assert calls["n"] == 1


def test_default_run_trace_does_not_touch_network(traces_dir, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("mock path must not open a network connection")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.delenv("SPECTRACE_BASE_URL", raising=False)
    monkeypatch.delenv("SPECTRACE_API_KEY", raising=False)
    trace = load_traces(traces_dir)[0]
    metrics = run_trace(trace, "baseline")
    assert metrics.provider == "mock-local"
    assert "not a live model call" in metrics.notes
    assert metrics.extra == {}
    assert metrics.cost_estimate_usd == 0.0


def test_run_trace_live_overlays_timing_and_usage(traces_dir):
    trace = next(t for t in load_traces(traces_dir) if t.id == "research_qa_01")
    n_asst = len(trace.assistant_turns)
    calls = {"n": 0}

    class _Stub:
        def serve_recorded_step(self, _trace, message_index, *, max_tokens=None):
            calls["n"] += 1
            return ChatCompletionResult(
                content="draft",
                prompt_tokens=10,
                completion_tokens=2,
                latency_ms=15.0,
                model="local-model",
            )

    spec = ProviderSpec(
        name="openai-compat",
        model="local-model",
        base_url="http://127.0.0.1:9/v1",
    )
    mock = run_trace(trace, "baseline")
    live = run_trace(
        trace,
        "baseline",
        provider=spec,
        client=_Stub(),
    )
    assert calls["n"] == n_asst
    assert live.provider == "openai-compat"
    assert live.wall_latency_ms == pytest.approx(15.0 * n_asst)
    assert live.tokens_in == 10 * n_asst
    assert live.tokens_out == 2 * n_asst
    assert live.success is True
    assert live.extra["live"] is True
    assert live.extra["fixture_tokens_in"] == mock.tokens_in
    assert live.extra["fixture_tokens_out"] == mock.tokens_out
    assert live.extra["fixture_latency_ms"] == pytest.approx(mock.wall_latency_ms)
    assert "Jev" in live.notes
    # Method simulation still fills draft stats on baseline (none).
    assert live.accept_rate is None
    assert live.draft_tokens_proposed == 0


def test_run_trace_live_keeps_fixture_tokens_without_usage(traces_dir):
    trace = load_traces(traces_dir)[0]
    n_asst = len(trace.assistant_turns)

    class _Stub:
        def serve_recorded_step(self, *_a, **_k):
            return ChatCompletionResult(content="x", latency_ms=7.0)

    spec = ProviderSpec(name="openai-compat", model="m", base_url="http://127.0.0.1:9/v1")
    mock = run_trace(trace, "mock_speculative", seed=0)
    live = run_trace(
        trace, "mock_speculative", seed=0, provider=spec, client=_Stub()
    )
    assert live.tokens_in == mock.tokens_in
    assert live.tokens_out == mock.tokens_out
    assert live.accept_rate == mock.accept_rate
    assert live.wall_latency_ms == pytest.approx(7.0 * n_asst)
    assert live.extra["live_prompt_tokens"] is None


def test_bakeoff_live_calls_server_once_per_step(traces_dir):
    traces = load_traces(traces_dir)[:1]
    n_asst = len(traces[0].assistant_turns)
    calls = {"n": 0}

    class _Stub:
        def serve_recorded_step(self, *_a, **_k):
            calls["n"] += 1
            return ChatCompletionResult(
                content="x", prompt_tokens=1, completion_tokens=1, latency_ms=1.0
            )

    spec = ProviderSpec(name="openai-compat", model="m", base_url="http://127.0.0.1:9/v1")
    summary = run_bakeoff(
        traces,
        ["baseline", "mock_speculative"],
        provider=spec,
        client=_Stub(),
    )
    assert calls["n"] == n_asst  # not 2× methods
    assert len(summary.runs) == 2
    assert all(r.provider == "openai-compat" for r in summary.runs)
    assert "LIVE" in summary.notes
    assert "Jev" in summary.notes


def test_cli_live_run_with_mocked_http(traces_dir, monkeypatch, capsys):
    monkeypatch.setenv("SPECTRACE_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("SPECTRACE_MODEL", "local-model")
    monkeypatch.delenv("SPECTRACE_API_KEY", raising=False)
    seen = {"n": 0}

    def fake_urlopen(request, timeout=None):
        seen["n"] += 1
        assert request.full_url.endswith("/chat/completions")
        return _FakeHTTP(_ok_payload())

    monkeypatch.setattr("spectrace.openai_compat.urllib.request.urlopen", fake_urlopen)
    code = main(
        [
            "run",
            "--trace",
            str(traces_dir / "research_qa.json"),
            "--method",
            "baseline",
            "--provider",
            "openai-compat",
            "--format",
            "json",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["runs"][0]["provider"] == "openai-compat"
    assert payload["runs"][0]["extra"]["live"] is True
    assert seen["n"] == payload["runs"][0]["n_generation_steps"]
    assert "LIVE" in payload["banner"] or "live" in payload["runs"][0]["notes"]


def test_cli_grade_unchanged_with_base_url_set(traces_dir, monkeypatch, capsys):
    """Setting SPECTRACE_BASE_URL must not change `spectrace grade`."""
    monkeypatch.setenv("SPECTRACE_BASE_URL", "http://127.0.0.1:9/v1")

    def boom(*_a, **_k):
        raise AssertionError("grade must not call OpenAI-compat")

    monkeypatch.setattr("spectrace.openai_compat.urllib.request.urlopen", boom)
    code = main(["grade", "--provider", "mock", "--traces", str(traces_dir / "code_fix.json")])
    assert code == 0
    out = capsys.readouterr().out
    assert "jev_accept" in out


@pytest.mark.integration
def test_live_openai_compat_optional_when_url_present():
    base = (os.environ.get("SPECTRACE_BASE_URL") or "").strip()
    if not base:
        pytest.skip("SPECTRACE_BASE_URL not set")
    client = OpenAICompatClient(
        base_url=base,
        model=os.environ.get("SPECTRACE_MODEL", "").strip() or "openai-compat",
        timeout=20.0,
        max_attempts=2,
    )
    result = client.chat_completions(
        [{"role": "user", "content": "Reply with the single word pong."}],
        max_tokens=8,
    )
    assert result.latency_ms >= 0
    assert (result.content or "").strip() or result.tool_calls

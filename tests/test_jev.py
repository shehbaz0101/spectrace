from __future__ import annotations

import io
import json
import os
import urllib.error
from email.message import Message

import pytest

from spectrace.jev import (
    JEV_ACCEPT_RULE,
    STEP_QUESTIONS,
    JevAPIError,
    JevConfigError,
    JevQuestion,
    LiveJev,
    MockJev,
    NullJev,
    get_grader,
    grade_step,
    grade_trajectory,
    jev_accept,
    parse_answers,
    parse_noul,
    resolve_grade_provider,
)


def test_null_jev_needs_no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    grader = NullJev()
    answers = grade_trajectory({"trace_id": "code_fix_01", "success": True}, grader)
    kinds = {a.name: a.type for a in answers}
    assert kinds == {"task_complete": "noul", "next_action": "choice", "trace_quality": "score"}
    task = next(a for a in answers if a.name == "task_complete")
    assert task.noul == 1.0 and task.yes is True


def test_get_grader_mock_default(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    assert isinstance(get_grader(live=False), MockJev)
    assert isinstance(get_grader(provider="null"), NullJev)


def test_live_jev_without_key_does_not_touch_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)

    def boom(*_a, **_k):
        raise AssertionError("LiveJev must not open a network connection without a key")

    live = LiveJev(urlopen=boom)
    with pytest.raises(JevConfigError, match="TYPESAFE_API_KEY") as exc:
        live.evaluate({"goal": "x"}, list(STEP_QUESTIONS))
    msg = str(exc.value)
    assert "draft" in msg.lower() or "grader" in msg.lower()
    assert "mock" in msg.lower()


def test_resolve_jev_provider_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    with pytest.raises(JevConfigError, match="TYPESAFE_API_KEY"):
        resolve_grade_provider("jev")
    assert isinstance(resolve_grade_provider("mock"), MockJev)


def test_typesafe_key_alias(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("TYPESAFE_KEY", "alias-secret")
    live = LiveJev()
    assert live.api_key == "alias-secret"


def test_parse_noul_clamps_and_rejects_junk():
    assert parse_noul(0.91) == 0.91
    assert parse_noul(1.7) == 1.0
    assert parse_noul(-0.2) == 0.0
    assert parse_noul("0.4") == 0.4
    assert parse_noul("yes") == 1.0
    assert parse_noul("NO") == 0.0
    assert parse_noul(True) == 1.0
    assert parse_noul({"noul": "0.25"}) == 0.25
    assert parse_noul(float("nan")) is None
    assert parse_noul(float("inf")) is None
    assert parse_noul("not-a-prob") is None
    assert parse_noul(None) is None
    assert parse_noul({}) is None


def test_parse_answers_dict_and_list():
    payload = {
        "model": "jev-1.13.0",
        "answers": {
            "tool_ok": {"type": "noul", "noul": 0.91},
            "progress": {"type": "noul", "noul": "0.7"},
            "abort": {"type": "noul", "noul": 0.1},
        },
    }
    answers = parse_answers(payload, STEP_QUESTIONS)
    by_name = {a.name: a for a in answers}
    assert by_name["tool_ok"].noul == 0.91
    assert by_name["progress"].noul == 0.7
    assert by_name["abort"].noul == 0.1
    assert by_name["escalate"].noul is None  # missing → None, fail closed later

    listed = parse_answers(
        {"answers": [{"name": "tool_ok", "type": "noul", "noul": 0.5}]},
        [JevQuestion(name="tool_ok", type="noul")],
    )
    assert listed[0].noul == 0.5


def test_jev_accept_rule():
    assert "tool_ok>=0.6" in JEV_ACCEPT_RULE.replace(" ", "")
    assert jev_accept(0.6, 0.5, 0.49) is True
    assert jev_accept(0.59, 1.0, 0.0) is False
    assert jev_accept(1.0, 0.49, 0.0) is False
    assert jev_accept(1.0, 1.0, 0.5) is False
    assert jev_accept(None, 1.0, 0.0) is False
    assert jev_accept(1.0, 1.0, None) is False


def _ok_state(**overrides):
    state = {
        "goal": "Find the population",
        "history_so_far": [{"role": "user", "content": "Find the population"}],
        "current_step": {
            "assistant": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "c1", "name": "web_search", "arguments": {"query": "pop"}}],
            },
            "tool_results": [{"role": "tool", "name": "web_search", "content": "8M"}],
        },
        "tool_schemas": [
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }
        ],
    }
    state.update(overrides)
    return state


def test_mock_grade_step_accepts_valid_tool():
    scored = grade_step(_ok_state(), MockJev())
    assert scored["jev_accept"] is True
    assert scored["tool_ok"] >= 0.6
    assert scored["progress"] >= 0.5
    assert scored["abort"] < 0.5
    assert scored["disposition"] in {"accept", "continue"}


def test_mock_grade_step_rejects_unknown_tool():
    state = _ok_state()
    state["current_step"]["assistant"]["tool_calls"] = [
        {"id": "c1", "name": "drop_database", "arguments": {"confirm": True}}
    ]
    scored = grade_step(state, MockJev())
    assert scored["jev_accept"] is False
    assert scored["tool_ok"] < 0.6
    assert scored["disposition"] == "reject"


def test_mock_grade_step_rejects_missing_required_arg():
    state = _ok_state()
    state["current_step"]["assistant"]["tool_calls"] = [
        {"id": "c1", "name": "web_search", "arguments": {}}
    ]
    scored = grade_step(state, MockJev())
    assert scored["jev_accept"] is False
    assert scored["tool_ok"] < 0.6


def test_mock_grade_step_rejects_exact_repeat_loop():
    state = _ok_state()
    call = {"id": "c1", "name": "web_search", "arguments": {"query": "pop"}}
    state["history_so_far"] = [
        {"role": "user", "content": "Find the population"},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "name": "web_search", "content": "8M"},
    ]
    state["current_step"]["assistant"]["tool_calls"] = [dict(call, id="c2")]
    scored = grade_step(state, MockJev())
    assert scored["jev_accept"] is False
    assert scored["abort"] >= 0.5
    assert scored["progress"] < 0.5


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


def test_live_jev_posts_and_parses(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
    seen: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["method"] = request.get_method()
        seen["auth"] = request.headers.get("Authorization") or request.get_header("Authorization")
        body = json.loads(request.data.decode("utf-8"))
        seen["body"] = body
        return _FakeHTTP(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "tool_ok": {"type": "noul", "noul": 0.9},
                    "progress": {"type": "noul", "noul": 0.8},
                    "abort": {"type": "noul", "noul": 0.05},
                    "escalate": {"type": "noul", "noul": 0.1},
                    "disposition": {
                        "type": "choice",
                        "choice": "accept",
                        "probabilities": {
                            "continue": 0.05,
                            "accept": 0.9,
                            "reject": 0.03,
                            "escalate": 0.02,
                        },
                        "confidence": 0.88,
                    },
                },
            }
        )

    live = LiveJev(urlopen=fake_urlopen, sleep=lambda _s: None)
    scored = grade_step(_ok_state(), live)
    assert seen["url"] == "https://api.typesafe.ai/v1/systemone"
    assert seen["method"] == "POST"
    assert seen["timeout"] == 30.0
    auth = str(seen["auth"])
    assert auth.endswith("test-key-not-real")
    assert auth.startswith("Bearer")
    body = seen["body"]
    assert body["model"] == "jev-latest"
    assert "goal" in body["state"]
    assert body["questions"]["tool_ok"]["type"] == "noul"
    assert "choice" in body["questions"]["disposition"]["type"]
    assert scored["jev_accept"] is True
    assert scored["disposition"] == "accept"
    assert scored["tool_ok"] == 0.9


def test_live_jev_retries_then_succeeds(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key-not-real")
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
        return _FakeHTTP({"answers": {"tool_ok": {"type": "noul", "noul": 0.77}}})

    live = LiveJev(urlopen=fake_urlopen, sleep=sleeps.append, max_attempts=3)
    answers = live.evaluate({"goal": "x"}, [JevQuestion(name="tool_ok", type="noul")])
    assert calls["n"] == 3
    assert sleeps == [0.4, 0.8]
    assert answers[0].noul == 0.77


def test_live_jev_http_401_does_not_retry(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "bad")
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

    live = LiveJev(urlopen=fake_urlopen, sleep=lambda _s: None)
    with pytest.raises(JevAPIError, match="401"):
        live.evaluate({"goal": "x"}, [JevQuestion(name="tool_ok", type="noul")])
    assert calls["n"] == 1


@pytest.mark.integration
def test_live_jev_optional_when_key_present():
    key = (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_KEY") or "").strip()
    if not key:
        pytest.skip("TYPESAFE_API_KEY not set")
    live = LiveJev(api_key=key, timeout=20.0, max_attempts=2)
    answers = live.evaluate(
        {
            "goal": "Say hello",
            "history_so_far": [{"role": "user", "content": "Say hello"}],
            "current_step": {
                "assistant": {"role": "assistant", "content": "Hello."},
                "tool_results": [],
            },
            "tool_schemas": [],
        },
        [JevQuestion(name="tool_ok", type="noul", instructions="Is this a reasonable reply?")],
    )
    assert answers[0].noul is not None
    assert 0.0 <= answers[0].noul <= 1.0

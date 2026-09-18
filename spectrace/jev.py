"""TypeSafe AI Jev: System-1 trajectory / step grader (not a draft model).

Jev returns calibrated noul / choice / score answers. spectrace uses it to
*accept or reject a recorded agent step*, never to generate tokens.

Default path is a local mock (no network, no key). Live calls:

    POST https://api.typesafe.ai/v1/systemone
    Authorization: Bearer $TYPESAFE_API_KEY   # alias: TYPESAFE_KEY

See docs/jev.md for the dual-brain story and the jev_accept rule.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol, Sequence

from spectrace import __version__

QuestionType = Literal["noul", "choice", "score"]

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT_S = 30.0
DEFAULT_MAX_ATTEMPTS = 3
RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})

# Deterministic accept rule (noul thresholds). Escalate is advisory only.
TOOL_OK_MIN = 0.6
PROGRESS_MIN = 0.5
ABORT_MAX = 0.5  # exclusive: abort must be strictly below this

JEV_ACCEPT_RULE = (
    f"tool_ok>={TOOL_OK_MIN} and progress>={PROGRESS_MIN} and abort<{ABORT_MAX}"
)


class JevError(RuntimeError):
    """Base error for Jev client / grading failures."""


class JevConfigError(JevError):
    """Missing key, unknown provider, or other local configuration problem."""


class JevAPIError(JevError):
    """HTTP / parse failure talking to TypeSafe System One."""


@dataclass(frozen=True)
class JevQuestion:
    name: str
    type: QuestionType
    instructions: str = ""
    criteria: dict[str, str] | list[str] | None = None

    def to_payload(self) -> dict[str, Any]:
        item: dict[str, Any] = {"type": self.type}
        if self.instructions:
            item["instructions"] = self.instructions
        if self.criteria is not None:
            item["criteria"] = self.criteria
        return item


@dataclass(frozen=True)
class JevAnswer:
    name: str
    type: QuestionType
    noul: float | None = None
    choice: str | None = None
    score: float | None = None
    confidence: float | None = None
    probabilities: dict[str, float] = field(default_factory=dict)

    @property
    def yes(self) -> bool | None:
        if self.type != "noul" or self.noul is None:
            return None
        return self.noul >= 0.5


class TrajectoryGrader(Protocol):
    def evaluate(self, state: dict[str, Any], questions: list[JevQuestion]) -> list[JevAnswer]:
        ...


# ---------------------------------------------------------------------------
# Per-step question set
# ---------------------------------------------------------------------------

STEP_QUESTIONS: list[JevQuestion] = [
    JevQuestion(
        name="tool_ok",
        type="noul",
        instructions=(
            "Is `current_step` schema-valid and appropriate given `goal`, "
            "`history_so_far`, and `tool_schemas`? For a tool call: the function "
            "exists in the schemas, required arguments are present and well-typed, "
            "and the call is a reasonable next action. For a free-text assistant "
            "turn: it is a coherent reply (or final answer) rather than a malformed "
            "or off-task tool attempt. Answer yes only if the step is both valid "
            "and appropriate."
        ),
        criteria={
            "true": "Known tool with required args, or a well-formed text step that fits the goal.",
            "false": "Unknown tool, missing/invalid arguments, empty step, or clearly inappropriate.",
        },
    ),
    JevQuestion(
        name="progress",
        type="noul",
        instructions=(
            "Does `current_step` move the agent toward `goal` relative to "
            "`history_so_far`? Yes if it gathers new evidence, applies a needed "
            "tool, or produces a usable final answer. No if it repeats a prior "
            "call, stalls, or changes the subject."
        ),
        criteria={
            "true": "New information, a needed tool, or a final answer that addresses the goal.",
            "false": "Repeat, stall, or off-task content that does not advance the goal.",
        },
    ),
    JevQuestion(
        name="abort",
        type="noul",
        instructions=(
            "Is `current_step` a wasteful loop or otherwise a reason to stop the "
            "trajectory? Yes if the agent is repeating the same tool call, spinning "
            "without new observations, or should halt. No if the step is a normal "
            "continuation or a legitimate retry after a recovered error."
        ),
        criteria={
            "true": "Wasteful loop, empty churn, or a step that should terminate the run.",
            "false": "Normal continuation, first use of a tool, or a justified retry.",
        },
    ),
    JevQuestion(
        name="escalate",
        type="noul",
        instructions=(
            "Should a human review this step before the agent continues? Yes for "
            "ambiguous high-stakes actions, schema failures the agent did not "
            "recover from, or contradictions with the goal. No if the step is "
            "routine and the noul scores above are enough for code to decide."
        ),
        criteria={
            "true": "Human review is warranted before continuing.",
            "false": "Code can accept or reject this step without a human.",
        },
    ),
    JevQuestion(
        name="disposition",
        type="choice",
        instructions=(
            "Pick one routing label for `current_step`. This is a parallel signal; "
            "spectrace's boolean jev_accept is computed only from tool_ok, progress, "
            "and abort nouls, not from this choice."
        ),
        criteria={
            "continue": "Keep going; the step is unfinished work that still helps.",
            "accept": "Treat this step as a valid move toward the goal.",
            "reject": "Reject the step as schema-invalid, off-task, or harmful.",
            "escalate": "Pause for a human; the step is ambiguous or high-stakes.",
        },
    ),
]


def questions_to_payload(questions: Sequence[JevQuestion]) -> dict[str, Any]:
    return {q.name: q.to_payload() for q in questions}


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_noul(value: Any) -> float | None:
    """Coerce a noul-like value to a probability in [0, 1], or None if unusable."""
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return max(0.0, min(1.0, number))
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if text in {"yes", "true", "y"}:
            return 1.0
        if text in {"no", "false", "n"}:
            return 0.0
        try:
            return parse_noul(float(text))
        except ValueError:
            return None
    if isinstance(value, dict):
        for key in ("noul", "probability", "p", "yes"):
            if key in value:
                parsed = parse_noul(value[key])
                if parsed is not None:
                    return parsed
    return None


def parse_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    if isinstance(value, str):
        try:
            return parse_float(float(value.strip()))
        except ValueError:
            return None
    return None


def parse_answer(name: str, raw: Any, *, fallback_type: QuestionType | None = None) -> JevAnswer:
    """Parse one System One answer object; never raises on malformed payloads."""
    if raw is None:
        qtype: QuestionType = fallback_type or "noul"
        return JevAnswer(name=name, type=qtype)
    if not isinstance(raw, dict):
        qtype = fallback_type or "noul"
        if qtype == "noul":
            return JevAnswer(name=name, type="noul", noul=parse_noul(raw))
        if qtype == "score":
            return JevAnswer(name=name, type="score", score=parse_float(raw))
        return JevAnswer(name=name, type="choice", choice=str(raw) if raw is not None else None)

    declared = raw.get("type")
    qtype = declared if declared in ("noul", "choice", "score") else (fallback_type or "noul")
    probs_raw = raw.get("probabilities") or {}
    probs: dict[str, float] = {}
    if isinstance(probs_raw, dict):
        for key, val in probs_raw.items():
            parsed = parse_float(val)
            if parsed is not None:
                probs[str(key)] = parsed

    noul_val = parse_noul(raw["noul"] if "noul" in raw else raw.get("probability"))
    confidence = parse_float(raw.get("confidence"))
    if qtype == "noul" and confidence is None and noul_val is not None:
        confidence = max(noul_val, 1.0 - noul_val)

    choice_val = raw.get("choice")
    if choice_val is not None:
        choice_val = str(choice_val)

    return JevAnswer(
        name=name,
        type=qtype,
        noul=noul_val,
        choice=choice_val,
        score=parse_float(raw.get("score")),
        confidence=confidence,
        probabilities=probs,
    )


def parse_answers(
    payload: Any,
    questions: Sequence[JevQuestion] | None = None,
) -> list[JevAnswer]:
    """Extract answers from a System One JSON body (dict or list)."""
    expected = {q.name: q.type for q in questions} if questions else {}
    blob: Any = payload
    if isinstance(payload, dict):
        if isinstance(payload.get("answers"), (dict, list)):
            blob = payload["answers"]
        elif "noul" in payload or "type" in payload:
            blob = payload

    parsed: dict[str, JevAnswer] = {}
    if isinstance(blob, dict):
        for name, raw in blob.items():
            parsed[str(name)] = parse_answer(str(name), raw, fallback_type=expected.get(str(name)))
    elif isinstance(blob, list):
        for item in blob:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or item.get("id") or "")
            if not name:
                continue
            parsed[name] = parse_answer(name, item, fallback_type=expected.get(name))

    if expected:
        return [
            parsed.get(q.name) or JevAnswer(name=q.name, type=q.type) for q in questions or ()
        ]
    return list(parsed.values())


def jev_accept(
    tool_ok: float | None,
    progress: float | None,
    abort: float | None,
) -> bool:
    """Boolean step accept from the three core nouls.

    Missing / unparseable nouls fail closed (reject). ``escalate`` is not
    part of this rule.
    """
    if tool_ok is None or progress is None or abort is None:
        return False
    return tool_ok >= TOOL_OK_MIN and progress >= PROGRESS_MIN and abort < ABORT_MAX


def noul_from_answers(answers: Sequence[JevAnswer], name: str) -> float | None:
    for ans in answers:
        if ans.name == name:
            return ans.noul
    return None


def choice_from_answers(answers: Sequence[JevAnswer], name: str) -> str | None:
    for ans in answers:
        if ans.name == name:
            return ans.choice
    return None


# ---------------------------------------------------------------------------
# API key
# ---------------------------------------------------------------------------


def api_key_from_env() -> str:
    return (os.environ.get("TYPESAFE_API_KEY") or os.environ.get("TYPESAFE_KEY") or "").strip()


def missing_key_message() -> str:
    return (
        "--provider jev requires TYPESAFE_API_KEY (or TYPESAFE_KEY). "
        "Jev is a System-1 step grader, not a draft/token generator. "
        "Use --provider mock for offline grading (CI / $0)."
    )


# ---------------------------------------------------------------------------
# Mock / null graders (no network)
# ---------------------------------------------------------------------------


def _tool_schema_index(tools: Any) -> dict[str, set[str]]:
    index: dict[str, set[str]] = {}
    if not isinstance(tools, list):
        return index
    for spec in tools:
        if not isinstance(spec, dict):
            continue
        fn = spec.get("function") if isinstance(spec.get("function"), dict) else spec
        name = fn.get("name") or spec.get("name")
        if not name:
            continue
        params = fn.get("parameters") if isinstance(fn.get("parameters"), dict) else {}
        required = params.get("required") or []
        index[str(name)] = {str(r) for r in required}
    return index


def _as_messages(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, dict):
        if "assistant" in value or "tool_results" in value:
            rows = []
            if isinstance(value.get("assistant"), dict):
                rows.append(value["assistant"])
            results = value.get("tool_results") or []
            if isinstance(results, list):
                rows.extend(m for m in results if isinstance(m, dict))
            return rows
        return [value]
    if isinstance(value, list):
        return [m for m in value if isinstance(m, dict)]
    return []


def _call_sig(call: dict[str, Any]) -> tuple[str, str]:
    name = str(call.get("name") or "")
    args = call.get("arguments") or {}
    try:
        packed = json.dumps(args, sort_keys=True, separators=(",", ":"), default=str)
    except TypeError:
        packed = str(args)
    return name, packed


def mock_noul_scores(state: dict[str, Any]) -> dict[str, float]:
    """Deterministic local stand-in for the four step nouls.

    Heuristics are conservative and inspect tool names / required args / repeats
    so tests can construct both accept and reject steps without an API key.
    """
    current = state.get("current_step")
    if current is None:
        # Trajectory-level or empty state → optimistic (NullJev-compatible).
        success = bool(state.get("success", True))
        return {
            "tool_ok": 1.0 if success else 0.2,
            "progress": 1.0 if success else 0.2,
            "abort": 0.0 if success else 0.8,
            "escalate": 0.0 if success else 0.7,
            "task_complete": 1.0 if success else 0.0,
        }

    step_msgs = _as_messages(current)
    assistant = next((m for m in step_msgs if m.get("role") == "assistant"), None)
    if assistant is None and step_msgs:
        assistant = step_msgs[0]
    assistant = assistant or {}
    followups = [m for m in step_msgs if m.get("role") == "tool"]
    history = state.get("history_so_far") if isinstance(state.get("history_so_far"), list) else []
    schemas = _tool_schema_index(state.get("tool_schemas") or state.get("tools") or [])

    calls = assistant.get("tool_calls") or []
    if not isinstance(calls, list):
        calls = []
    content = str(assistant.get("content") or "").strip()

    if calls:
        oks: list[float] = []
        for call in calls:
            if not isinstance(call, dict):
                oks.append(0.15)
                continue
            name = str(call.get("name") or "")
            args = call.get("arguments") if isinstance(call.get("arguments"), dict) else None
            if schemas and name not in schemas:
                oks.append(0.12)
            elif args is None:
                oks.append(0.22)
            elif schemas and any(req not in args for req in schemas.get(name, ())):
                oks.append(0.28)
            else:
                oks.append(0.93)
        tool_ok = sum(oks) / len(oks)
    elif content:
        tool_ok = 0.88
    else:
        tool_ok = 0.18

    hist_calls: list[dict[str, Any]] = []
    for msg in history:
        if not isinstance(msg, dict):
            continue
        for call in msg.get("tool_calls") or []:
            if isinstance(call, dict):
                hist_calls.append(call)

    cur_sigs = [_call_sig(c) for c in calls if isinstance(c, dict)]
    prev_assistant_calls: list[dict[str, Any]] = []
    for msg in reversed(history):
        if isinstance(msg, dict) and msg.get("role") == "assistant" and msg.get("tool_calls"):
            prev_assistant_calls = [c for c in msg["tool_calls"] if isinstance(c, dict)]
            break
    prev_sigs = [_call_sig(c) for c in prev_assistant_calls]

    exact_repeat = bool(cur_sigs) and bool(prev_sigs) and set(cur_sigs) == set(prev_sigs)
    name_counts: dict[str, int] = {}
    for name, _ in [_call_sig(c) for c in hist_calls] + cur_sigs:
        name_counts[name] = name_counts.get(name, 0) + 1
    max_same = max(name_counts.values()) if name_counts else 0

    if exact_repeat:
        progress = 0.18
        abort = 0.78
    elif not calls and not content:
        progress = 0.15
        abort = 0.82
    elif calls and schemas and any(n not in schemas for n, _ in cur_sigs):
        progress = 0.22
        abort = 0.35
    elif max_same >= 4:
        progress = 0.32
        abort = 0.62
    elif content and not calls:
        progress = 0.80
        abort = 0.10
    else:
        progress = 0.84
        abort = 0.11

    recent_error = any(
        isinstance(m, dict) and m.get("error") for m in list(history)[-8:] + followups
    )
    if tool_ok < 0.4:
        escalate = 0.62
    elif recent_error:
        escalate = 0.58
    else:
        escalate = 0.09

    return {
        "tool_ok": tool_ok,
        "progress": progress,
        "abort": abort,
        "escalate": escalate,
        "task_complete": 1.0 if jev_accept(tool_ok, progress, abort) else 0.2,
    }


def _mock_disposition(scores: dict[str, float]) -> str:
    if scores.get("abort", 0) >= ABORT_MAX or (scores.get("tool_ok") or 0) < TOOL_OK_MIN:
        return "reject"
    if scores.get("escalate", 0) >= 0.5:
        return "escalate"
    if jev_accept(scores.get("tool_ok"), scores.get("progress"), scores.get("abort")):
        return "accept"
    return "continue"


class MockJev:
    """Deterministic local grader. No API key, no HTTP. Used by ``--provider mock``."""

    name = "mock"

    def evaluate(self, state: dict[str, Any], questions: list[JevQuestion]) -> list[JevAnswer]:
        scores = mock_noul_scores(state or {})
        disposition = _mock_disposition(scores)
        answers: list[JevAnswer] = []
        for q in questions:
            if q.type == "noul":
                noul = scores.get(q.name)
                if noul is None:
                    noul = 1.0 if q.name not in {"abort", "escalate"} else 0.0
                answers.append(
                    JevAnswer(
                        name=q.name,
                        type="noul",
                        noul=noul,
                        confidence=max(noul, 1.0 - noul),
                    )
                )
            elif q.type == "choice":
                criteria = q.criteria if isinstance(q.criteria, dict) else {}
                if q.name == "disposition" and disposition in criteria:
                    label = disposition
                else:
                    label = next(iter(criteria), "other")
                answers.append(
                    JevAnswer(
                        name=q.name,
                        type="choice",
                        choice=label,
                        confidence=1.0,
                        probabilities={label: 1.0} if label else {},
                    )
                )
            else:
                answers.append(JevAnswer(name=q.name, type="score", score=0.0, confidence=1.0))
        return answers


class NullJev(MockJev):
    """Optimistic stand-in (always-yes nouls). Kept for Act 1 ``grade_trajectory``."""

    name = "null"

    def evaluate(self, state: dict[str, Any], questions: list[JevQuestion]) -> list[JevAnswer]:
        answers: list[JevAnswer] = []
        for q in questions:
            if q.type == "noul":
                answers.append(JevAnswer(name=q.name, type="noul", noul=1.0, confidence=1.0))
            elif q.type == "choice":
                criteria = q.criteria if isinstance(q.criteria, dict) else {}
                label = next(iter(criteria), "other")
                answers.append(
                    JevAnswer(
                        name=q.name,
                        type="choice",
                        choice=label,
                        confidence=1.0,
                        probabilities={label: 1.0} if label else {},
                    )
                )
            else:
                answers.append(JevAnswer(name=q.name, type="score", score=0.0, confidence=1.0))
        return answers


# ---------------------------------------------------------------------------
# Live HTTP client
# ---------------------------------------------------------------------------


def _backoff_s(attempt: int) -> float:
    return 0.4 * (2**attempt)


class LiveJev:
    """POST https://api.typesafe.ai/v1/systemone with timeout + retry.

    Instantiating does not perform I/O. ``evaluate`` requires a key and is
    never called from the default bake-off path.
    """

    endpoint = DEFAULT_ENDPOINT

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        *,
        timeout: float = DEFAULT_TIMEOUT_S,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        endpoint: str | None = None,
        urlopen: Callable[..., Any] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self.api_key = (api_key if api_key is not None else api_key_from_env()).strip()
        self.model = model or os.environ.get("SPECTRACE_JEV_MODEL", "") or DEFAULT_MODEL
        self.timeout = float(timeout)
        self.max_attempts = max(1, int(max_attempts))
        env_url = os.environ.get("SPECTRACE_JEV_URL", "").strip()
        self.endpoint = (endpoint or env_url or DEFAULT_ENDPOINT).rstrip("/")
        self._urlopen = urlopen or urllib.request.urlopen
        self._sleep = sleep or time.sleep

    def evaluate(self, state: dict[str, Any], questions: list[JevQuestion]) -> list[JevAnswer]:
        if not self.api_key:
            raise JevConfigError(missing_key_message())
        if not questions:
            return []
        body = json.dumps(
            {
                "model": self.model,
                "state": state,
                "questions": questions_to_payload(questions),
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": f"spectrace/{__version__}",
            },
        )
        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with self._urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                text = raw.decode("utf-8") if isinstance(raw, (bytes, bytearray)) else str(raw)
                try:
                    payload = json.loads(text) if text else {}
                except json.JSONDecodeError as exc:
                    raise JevAPIError(f"Jev returned non-JSON body: {exc}") from exc
                return parse_answers(payload, questions)
            except JevAPIError:
                raise
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = exc.read().decode("utf-8", errors="replace")[:500]
                except Exception:
                    err_body = ""
                last_error = JevAPIError(f"Jev HTTP {exc.code}: {err_body or exc.reason}")
                if exc.code in RETRY_STATUS and attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = JevAPIError(f"Jev network error: {exc.reason}")
                if attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
            except TimeoutError as exc:
                last_error = JevAPIError(f"Jev timed out after {self.timeout}s")
                if attempt + 1 < self.max_attempts:
                    self._sleep(_backoff_s(attempt))
                    continue
                raise last_error from exc
        raise last_error or JevAPIError("Jev request failed")


def get_grader(*, live: bool = False, provider: str | None = None) -> TrajectoryGrader:
    """Return a grader. ``live=True`` / ``provider='jev'`` is LiveJev (key checked on evaluate)."""
    name = (provider or ("jev" if live else "mock")).strip().lower()
    if name in {"mock", "null", "local", "mock-local"}:
        return MockJev() if name != "null" else NullJev()
    if name in {"jev", "live", "typesafe"}:
        return LiveJev()
    raise JevConfigError(
        f"unknown Jev provider {name!r} (expected 'mock' or 'jev'). "
        "Jev is a grader, not a draft model."
    )


def resolve_grade_provider(name: str | None) -> TrajectoryGrader:
    name = (name or "mock").strip().lower()
    if name in {"mock", "null", "local", "mock-local"}:
        return MockJev()
    if name in {"jev", "live", "typesafe"}:
        if not api_key_from_env():
            raise JevConfigError(missing_key_message())
        return LiveJev()
    raise JevConfigError(
        f"unknown Jev provider {name!r} (expected 'mock' or 'jev'). "
        "Jev is a grader, not a draft model."
    )


def grade_step(
    state: dict[str, Any],
    grader: TrajectoryGrader | None = None,
    *,
    questions: Sequence[JevQuestion] | None = None,
) -> dict[str, Any]:
    """Grade one agent step. Returns scores plus the jev_accept boolean."""
    grader = grader or MockJev()
    qlist = list(questions) if questions is not None else list(STEP_QUESTIONS)
    answers = grader.evaluate(state, qlist)
    tool_ok = noul_from_answers(answers, "tool_ok")
    progress = noul_from_answers(answers, "progress")
    abort = noul_from_answers(answers, "abort")
    escalate = noul_from_answers(answers, "escalate")
    return {
        "tool_ok": tool_ok,
        "progress": progress,
        "abort": abort,
        "escalate": escalate,
        "disposition": choice_from_answers(answers, "disposition"),
        "jev_accept": jev_accept(tool_ok, progress, abort),
        "answers": answers,
        "rule": JEV_ACCEPT_RULE,
    }


def grade_trajectory(state: dict[str, Any], grader: TrajectoryGrader | None = None) -> list[JevAnswer]:
    """Example question set for a recorded agent trajectory (Act 1 helper)."""
    grader = grader or NullJev()
    questions = [
        JevQuestion(
            name="task_complete",
            type="noul",
            instructions="Did the agent finish the user task with a usable answer?",
        ),
        JevQuestion(
            name="next_action",
            type="choice",
            instructions="If the trajectory is still open, what should happen next?",
            criteria={
                "continue": "Run another model/tool step",
                "stop_success": "Stop; the task is done",
                "stop_fail": "Stop; the agent is stuck",
            },
        ),
        JevQuestion(
            name="trace_quality",
            type="score",
            instructions="How well did this trajectory use tools?",
            criteria=["poor", "adequate", "good", "excellent"],
        ),
    ]
    return grader.evaluate(state, questions)


def grade_trace_states(
    states: Iterable[dict[str, Any]],
    grader: TrajectoryGrader | None = None,
) -> list[dict[str, Any]]:
    """Grade an iterable of already-built step states."""
    grader = grader or MockJev()
    return [grade_step(state, grader) for state in states]

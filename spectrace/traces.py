"""Load and validate synthetic multi-step agent traces.

Traces are JSON documents with OpenAI-style chat + tool-call messages.
They are *not* ShareGPT single-turn chat dumps: context grows across
tool rounds, and assistant turns are a mix of free text and structured
function calls.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Role = Literal["system", "user", "assistant", "tool"]

# Shell / SQL actions a coding or data agent should not take unless the
# task asked for them. Matched against assistant text and tool arguments.
DESTRUCTIVE_RE = re.compile(
    r"(?:"
    r"\brm\s+(?:-\w*r\w*|--recursive)\b"
    r"|\bdrop(?:ped)?\s+(?:the\s+)?(?:table|database)\b"
    r"|\btruncate\s+table\b"
    r")",
    re.IGNORECASE,
)


def contains_destructive(text: str) -> bool:
    """True when text contains an ``rm -rf``-style or drop/truncate command."""
    return bool(text and DESTRUCTIVE_RE.search(text))


class TraceError(ValueError):
    """Raised when a trace file is missing required structure."""


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ToolCall":
        fn = raw.get("function") or {}
        name = raw.get("name") or fn.get("name")
        if not name:
            raise TraceError("tool_call missing name")
        args = raw.get("arguments", fn.get("arguments", {}))
        if isinstance(args, str):
            try:
                args = json.loads(args) if args else {}
            except json.JSONDecodeError as exc:
                raise TraceError(f"tool_call arguments are not JSON: {exc}") from exc
        if not isinstance(args, dict):
            raise TraceError("tool_call arguments must be an object")
        call_id = str(raw.get("id") or raw.get("tool_call_id") or name)
        return cls(id=call_id, name=str(name), arguments=args)

    def arguments_json(self) -> str:
        return json.dumps(self.arguments, separators=(",", ":"), sort_keys=True)


@dataclass
class Message:
    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    error: bool = False
    # Gold label for assistant steps: should a step acceptor keep this turn?
    # None means unlabeled (unit-test traces). Fixture files set it explicitly.
    expected_ok: bool | None = None

    @property
    def step_kind(self) -> str:
        if self.role != "assistant":
            return self.role
        if self.tool_calls and self.content.strip():
            return "mixed"
        if self.tool_calls:
            return "tool_call"
        return "text"

    def text_for_tokens(self) -> str:
        parts = [self.content or ""]
        for call in self.tool_calls:
            parts.append(call.name)
            parts.append(call.arguments_json())
        return "\n".join(p for p in parts if p)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Message":
        role = raw.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise TraceError(f"unknown role: {role!r}")
        tool_calls = [ToolCall.from_dict(tc) for tc in raw.get("tool_calls") or []]
        content = raw.get("content") or ""
        if not isinstance(content, str):
            content = json.dumps(content)
        expected_raw = raw.get("expected_ok", None)
        expected_ok = None if expected_raw is None else bool(expected_raw)
        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=raw.get("tool_call_id"),
            name=raw.get("name"),
            error=bool(raw.get("error")),
            expected_ok=expected_ok,
        )


@dataclass
class SuccessCriteria:
    required_tools: tuple[str, ...] = ()
    final_answer_required: bool = True
    # Substrings the final answer must include (case-insensitive). A miss is
    # ``off_goal`` — the agent answered a different task.
    answer_must_contain: tuple[str, ...] = ()
    # (tool name, substring) that must appear in that tool's arguments.
    # Called with the wrong document / id → ``wrong_source``.
    required_arg_substrings: tuple[tuple[str, str], ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SuccessCriteria":
        raw = raw or {}
        tools = raw.get("required_tools") or raw.get("must_call_tools") or []
        must = raw.get("answer_must_contain") or []
        pairs: list[tuple[str, str]] = []
        for item in raw.get("required_arg_substrings") or []:
            if isinstance(item, dict) and item.get("tool") and item.get("contains"):
                pairs.append((str(item["tool"]), str(item["contains"])))
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append((str(item[0]), str(item[1])))
        return cls(
            required_tools=tuple(str(t) for t in tools),
            final_answer_required=bool(raw.get("final_answer_required", raw.get("must_answer", True))),
            answer_must_contain=tuple(str(s) for s in must),
            required_arg_substrings=tuple(pairs),
        )


@dataclass
class AgentTrace:
    id: str
    task_type: str
    prompt: str
    messages: list[Message]
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    expected_success: bool = True
    # Gold trace label. None on unlabeled in-memory traces. Fixtures set it,
    # and structural_success() is checked against it in tests.
    task_success: bool | None = None
    failure_mode: str | None = None
    provenance: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    notes: str = ""
    source_path: Path | None = None

    @property
    def assistant_turns(self) -> list[Message]:
        return [m for m in self.messages if m.role == "assistant"]

    def tool_names_called(self) -> list[str]:
        names: list[str] = []
        for msg in self.messages:
            for call in msg.tool_calls:
                names.append(call.name)
        return names

    def final_assistant_text(self) -> str:
        for msg in reversed(self.messages):
            if msg.role == "assistant" and not msg.tool_calls:
                return msg.content.strip()
        return ""

    def _schema_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for spec in self.tools:
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

    def _has_destructive_action(self) -> bool:
        for msg in self.messages:
            if msg.role == "assistant" and contains_destructive(msg.content):
                return True
            for call in msg.tool_calls:
                if contains_destructive(call.arguments_json()):
                    return True
        return False

    def _has_unknown_tool(self) -> bool:
        schemas = self._schema_index()
        if not schemas:
            return False
        return any(name not in schemas for name in self.tool_names_called())

    def _has_invalid_arguments(self) -> bool:
        schemas = self._schema_index()
        if not schemas:
            return False
        for msg in self.messages:
            for call in msg.tool_calls:
                if call.name not in schemas:
                    continue
                required = schemas[call.name]
                if any(req not in call.arguments for req in required):
                    return True
        return False

    def _has_repeated_tool_call(self) -> bool:
        """Consecutive assistant tool turns with the same name+arguments."""
        prev: tuple[tuple[str, str], ...] | None = None
        for msg in self.messages:
            if msg.role != "assistant":
                continue
            if not msg.tool_calls:
                prev = None
                continue
            sig = tuple((c.name, c.arguments_json()) for c in msg.tool_calls)
            if prev is not None and sig == prev:
                return True
            prev = sig
        return False

    def _has_unrecovered_tool_error(self) -> bool:
        """A tool error with no later successful tool result.

        ``code_fix_01`` fails a test, then patches, then passes: the later
        non-error result clears the flag. Claiming success with the error
        still open does not.
        """
        open_error = False
        for msg in self.messages:
            if msg.role != "tool":
                continue
            open_error = bool(msg.error)
        return open_error

    def _has_wrong_source(self) -> bool:
        for tool, needle in self.success_criteria.required_arg_substrings:
            blobs = [
                call.arguments_json()
                for msg in self.messages
                for call in msg.tool_calls
                if call.name == tool
            ]
            if not blobs:
                continue
            if needle.casefold() not in " ".join(blobs).casefold():
                return True
        return False

    def _answer_misses_goal(self, final: str) -> bool:
        needles = self.success_criteria.answer_must_contain
        if not needles:
            return False
        folded = final.casefold()
        return any(piece.casefold() not in folded for piece in needles)

    def structural_success(self) -> tuple[bool, str | None]:
        """Check fixture-level task completion (independent of serving method).

        Failure modes are structural facts about the recorded trajectory
        (unknown tool, bad args, loop, unrecovered error, destructive
        command, wrong document id, answer that misses the goal, premature
        answer, stall). They are not a hidden ``expected_success`` override.
        """
        if self._has_destructive_action():
            return False, "destructive_action"
        if self._has_unknown_tool():
            return False, "unknown_tool"
        if self._has_invalid_arguments():
            return False, "invalid_arguments"
        if self._has_repeated_tool_call():
            return False, "repeated_tool_call"
        if self._has_unrecovered_tool_error():
            return False, "unrecovered_tool_error"
        if self._has_wrong_source():
            return False, "wrong_source"
        final = self.final_assistant_text()
        if final and self._answer_misses_goal(final):
            return False, "off_goal"
        missing = [t for t in self.success_criteria.required_tools if t not in set(self.tool_names_called())]
        if (
            self.success_criteria.required_tools
            and not self.tool_names_called()
            and len(self.assistant_turns) >= 2
            and final
        ):
            return False, "stalled"
        if missing and final:
            return False, "premature_final_answer"
        if missing:
            return False, "missing_required_tool"
        if self.success_criteria.final_answer_required and not final:
            return False, "missing_final_answer"
        if not self.expected_success:
            return False, "expected_failure"
        return True, None


def _as_trace(raw: dict[str, Any], path: Path | None = None) -> AgentTrace:
    if "id" not in raw:
        raise TraceError("trace missing 'id'")
    messages_raw = raw.get("messages") or raw.get("steps")
    if not messages_raw:
        raise TraceError("trace missing 'messages'")
    messages = [Message.from_dict(m) for m in messages_raw]
    prompt = raw.get("prompt") or raw.get("task") or ""
    if not prompt:
        user = next((m.content for m in messages if m.role == "user"), "")
        prompt = user
    if "task_success" in raw:
        task_success: bool | None = bool(raw["task_success"])
    else:
        task_success = None
    if "expected_success" in raw:
        expected_success = bool(raw["expected_success"])
    elif task_success is not None:
        expected_success = task_success
    else:
        expected_success = True
    failure_raw = raw.get("failure_mode")
    return AgentTrace(
        id=str(raw["id"]),
        task_type=str(raw.get("task_type") or raw.get("type") or "unknown"),
        prompt=str(prompt),
        messages=messages,
        success_criteria=SuccessCriteria.from_dict(raw.get("success_criteria")),
        expected_success=expected_success,
        task_success=task_success,
        failure_mode=str(failure_raw) if failure_raw else None,
        provenance=str(raw.get("provenance") or ""),
        tools=list(raw.get("tools") or []),
        notes=str(raw.get("notes") or ""),
        source_path=path,
    )


def load_trace(path: str | Path) -> AgentTrace:
    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, dict):
        raise TraceError(f"{path} must contain a JSON object")
    return _as_trace(raw, path)


def iter_trace_paths(target: str | Path) -> Iterator[Path]:
    target = Path(target)
    if target.is_file():
        yield target
        return
    if not target.is_dir():
        raise FileNotFoundError(f"trace path not found: {target}")
    paths = sorted(p for p in target.glob("*.json") if p.name != "schema.json")
    if not paths:
        raise FileNotFoundError(f"no JSON traces in {target}")
    yield from paths


def load_traces(target: str | Path) -> list[AgentTrace]:
    return [load_trace(p) for p in iter_trace_paths(target)]


def traces_summary(traces: Iterable[AgentTrace]) -> list[dict[str, Any]]:
    rows = []
    for t in traces:
        struct_ok, struct_mode = t.structural_success()
        rows.append(
            {
                "id": t.id,
                "task_type": t.task_type,
                "n_messages": len(t.messages),
                "n_assistant": len(t.assistant_turns),
                "n_tool_calls": len(t.tool_names_called()),
                "expected_success": t.expected_success,
                "task_success": t.task_success,
                "failure_mode": t.failure_mode,
                "provenance": t.provenance,
                "structural_success": struct_ok,
                "structural_failure_mode": struct_mode,
            }
        )
    return rows

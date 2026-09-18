"""Load and validate synthetic multi-step agent traces.

Traces are JSON documents with OpenAI-style chat + tool-call messages.
They are *not* ShareGPT single-turn chat dumps: context grows across
tool rounds, and assistant turns are a mix of free text and structured
function calls.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Literal

Role = Literal["system", "user", "assistant", "tool"]


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
        return cls(
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_call_id=raw.get("tool_call_id"),
            name=raw.get("name"),
            error=bool(raw.get("error")),
        )


@dataclass
class SuccessCriteria:
    required_tools: tuple[str, ...] = ()
    final_answer_required: bool = True

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "SuccessCriteria":
        raw = raw or {}
        tools = raw.get("required_tools") or raw.get("must_call_tools") or []
        return cls(
            required_tools=tuple(str(t) for t in tools),
            final_answer_required=bool(raw.get("final_answer_required", raw.get("must_answer", True))),
        )


@dataclass
class AgentTrace:
    id: str
    task_type: str
    prompt: str
    messages: list[Message]
    success_criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    expected_success: bool = True
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

    def structural_success(self) -> tuple[bool, str | None]:
        """Check fixture-level task completion (independent of serving method)."""
        called = set(self.tool_names_called())
        missing = [t for t in self.success_criteria.required_tools if t not in called]
        if missing:
            return False, "missing_required_tool"
        if self.success_criteria.final_answer_required and not self.final_assistant_text():
            return False, "missing_final_answer"
        if any(m.error and m.role == "tool" for m in self.messages):
            # Recovered tool errors still count as success if a final answer exists.
            if not self.final_assistant_text():
                return False, "tool_error"
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
    return AgentTrace(
        id=str(raw["id"]),
        task_type=str(raw.get("task_type") or raw.get("type") or "unknown"),
        prompt=str(prompt),
        messages=messages,
        success_criteria=SuccessCriteria.from_dict(raw.get("success_criteria")),
        expected_success=bool(raw.get("expected_success", True)),
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
        rows.append(
            {
                "id": t.id,
                "task_type": t.task_type,
                "n_messages": len(t.messages),
                "n_assistant": len(t.assistant_turns),
                "n_tool_calls": len(t.tool_names_called()),
                "expected_success": t.expected_success,
            }
        )
    return rows

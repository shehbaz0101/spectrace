"""Optional TypeSafe AI Jev adapter (stub).

Jev is a System One model: calibrated yes/no (noul), choice, and score.
It does **not** generate tokens and is **not** a speculative-decoding draft
model. spectrace may later use it as a trajectory grader or step router.

This module never opens a network connection. Tests must pass with no
TYPESAFE_API_KEY. See docs/jev.md.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

QuestionType = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class JevQuestion:
    name: str
    type: QuestionType
    instructions: str = ""
    criteria: dict[str, str] | list[str] | None = None


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


class NullJev:
    """Deterministic local stand-in. No API key, no HTTP."""

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


class LiveJev:
    """Placeholder for POST https://api.typesafe.ai/v1/systemone.

    Instantiating this does not perform I/O. ``evaluate`` raises until a
    caller explicitly opts in with a key — tests never do.
    """

    endpoint = "https://api.typesafe.ai/v1/systemone"

    def __init__(self, api_key: str | None = None, model: str = "jev-latest") -> None:
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self.model = model or os.environ.get("SPECTRACE_JEV_MODEL", "jev-latest")

    def evaluate(self, state: dict[str, Any], questions: list[JevQuestion]) -> list[JevAnswer]:
        raise RuntimeError(
            "Live Jev is not invoked from the default spectrace path. "
            "Use NullJev, or wire TYPESAFE_API_KEY in a future integration. "
            "Jev is a grader/router, not a draft model."
        )


def get_grader(*, live: bool = False) -> TrajectoryGrader:
    if live:
        return LiveJev()
    return NullJev()


def grade_trajectory(state: dict[str, Any], grader: TrajectoryGrader | None = None) -> list[JevAnswer]:
    """Example question set for a recorded agent trajectory."""
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

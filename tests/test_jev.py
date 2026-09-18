from __future__ import annotations

from spectrace.jev import LiveJev, NullJev, get_grader, grade_trajectory


def test_null_jev_needs_no_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    grader = get_grader(live=False)
    assert isinstance(grader, NullJev)
    answers = grade_trajectory({"trace_id": "code_fix_01", "success": True}, grader)
    kinds = {a.name: a.type for a in answers}
    assert kinds == {"task_complete": "noul", "next_action": "choice", "trace_quality": "score"}
    task = next(a for a in answers if a.name == "task_complete")
    assert task.noul == 1.0 and task.yes is True


def test_live_jev_does_not_touch_network(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    live = LiveJev()
    try:
        live.evaluate({"trace_id": "x"}, [])
        raise AssertionError("LiveJev should refuse")
    except RuntimeError as exc:
        assert "draft model" in str(exc)
        assert "TYPESAFE_API_KEY" in str(exc)

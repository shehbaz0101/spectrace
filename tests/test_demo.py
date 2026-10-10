"""spectrace demo: offline HTML, committed LIVE report, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from spectrace.cli import main
from spectrace.demo import compare_to_mock, load_grade_report
from spectrace.grade import grade_traces
from spectrace.jev import MockJev


def _html(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE html>")
    assert text.rstrip().endswith("</html>")
    return text


def test_live_15_trace_report_is_labeled_and_keeps_the_5_trace_file(repo_root: Path):
    path = repo_root / "docs" / "live-jev-grade-15traces-20261010.json"
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["live"] is True
    assert payload["label"] == "LIVE"
    assert payload["provider"] == "jev"
    assert payload["model"] == "jev-latest"
    assert "2026-10-10" in payload["run_at"]
    assert "EDT" in payload["run_at"]
    assert payload["timezone"] == "America/New_York"
    assert payload["n_traces"] == 15
    assert payload["n_steps"] == 49
    assert payload["n_jev_accept"] == 29
    assert len(payload["rows"]) == 49
    gold = payload["gold_comparison"]
    assert gold["is_mock"] is False
    assert gold["jev_accept_source"] == "live"
    assert gold["token_accept_source"] == "mock_speculative"
    jev = gold["jev_acceptor"]
    token = gold["token_acceptor"]
    assert (jev["tp"], jev["fp"], jev["tn"], jev["fn"]) == (29, 0, 20, 0)
    assert jev["precision"] == 1.0 and jev["recall"] == 1.0 and jev["f1"] == 1.0
    assert (token["tp"], token["fp"], token["tn"], token["fn"]) == (7, 11, 9, 22)
    assert token["precision"] == pytest.approx(0.3889, abs=1e-4)
    assert token["recall"] == pytest.approx(0.2414, abs=1e-4)
    assert token["f1"] == pytest.approx(0.2979, abs=1e-4)
    assert gold["n_high_token_bad"] == 11
    assert gold["n_high_token_bad_caught_by_jev"] == 11
    assert all(step["jev_caught"] is True for step in gold["high_token_accept_bad_steps"])
    overlap = payload["offline_mock_comparison"]
    assert overlap["n_accept_agree"] == 49
    assert overlap["n_noul_differ"] == 49
    assert overlap["n_unique_tool_ok_live"] == 30
    assert overlap["n_unique_tool_ok_mock"] == 11
    assert "Not returned by the TypeSafe API" in overlap["note"]
    for banned in ("sk-", "Bearer ", "TYPESAFE_API_KEY", "TYPESAFE_KEY"):
        assert banned not in raw
    old = json.loads((repo_root / "docs" / "live-jev-grade-20260920.json").read_text(encoding="utf-8"))
    assert old["live"] is True
    assert old["n_traces"] == 5
    assert old["n_steps"] == 22
    md = (repo_root / "docs" / "live-jev-grade-15traces-20261010.md").read_text(encoding="utf-8")
    assert "2026-10-10" in md
    assert "49/49" in md


def test_committed_live_rows_match_mock_decisions_not_scores(repo_root: Path, traces_dir: Path):
    payload = load_grade_report(repo_root / "docs" / "live-jev-grade-15traces-20261010.json")
    mock = grade_traces(str(traces_dir), MockJev(), provider="mock")
    agreement = compare_to_mock(payload["rows"], mock.rows)
    assert agreement["decisions_match"] is True
    assert agreement["scores_differ_every_step"] is True
    assert agreement["n_unique_tool_ok_report"] == 30
    assert agreement["n_unique_tool_ok_mock"] == 11
    stored = payload["offline_mock_comparison"]
    assert stored["n_accept_agree"] == agreement["n_accept_agree"]
    assert stored["n_noul_differ"] == agreement["n_noul_differ"]


def test_demo_offline_writes_self_contained_mock_html(traces_dir, tmp_path, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("demo must not open a network connection")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    monkeypatch.delenv("SPECTRACE_BASE_URL", raising=False)
    out = tmp_path / "index.html"
    code = main(["demo", "--traces", str(traces_dir), "--output", str(out)])
    assert code == 0
    text = _html(out)
    stdout = capsys.readouterr().out
    assert str(out) in stdout
    assert 'data-grade-source="mock"' in text
    assert "data-grade-source=\"live\"" not in text
    assert "pill-mock" in text
    assert "MOCK grade." in text
    assert "Jev never drafts." in text
    assert "SIMULATED" in text
    assert "766.8" in text
    assert "code_fix_01" in text
    assert "destructive_action" in text
    assert "1.000" in text
    assert "0.298" in text
    assert "<style>" in text
    assert "<svg" in text
    assert "<script" not in text.lower()
    assert "cdn." not in text.lower()
    assert "fonts.googleapis" not in text.lower()
    assert 'rel="stylesheet"' not in text
    assert "30 distinct LIVE tool_ok" in text
    assert "11 for MockJev" in text


def test_demo_from_report_renders_live_numbers_without_a_key(repo_root, traces_dir, tmp_path, monkeypatch, capsys):
    def boom(*_args, **_kwargs):
        raise AssertionError("from-report must not open a network connection")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    report = repo_root / "docs" / "live-jev-grade-15traces-20261010.json"
    out = tmp_path / "live.html"
    code = main(
        [
            "demo",
            "--from-report",
            str(report),
            "--traces",
            str(traces_dir),
            "--output",
            str(out),
        ]
    )
    assert code == 0
    text = _html(out)
    assert 'data-grade-source="live"' in text
    assert "pill-live" in text
    assert "2026-10-10" in text
    assert "Same decisions, different scores." in text
    assert "49/49" in text
    assert "30 distinct live tool_ok values, 11 for MockJev" in text
    assert "1.000" in text
    assert "0.298" in text
    assert "11/11" in text
    assert "stall_churn_01" in text
    assert "premature_final_answer" in text
    assert "SIMULATED" in text
    assert "Jev never drafts." in text
    assert "<svg" in text
    for banned in ("sk-", "Bearer ", "TYPESAFE_API_KEY", "TYPESAFE_KEY"):
        assert banned not in text
    stdout = capsys.readouterr().out
    assert "LIVE" in stdout
    assert str(out) in stdout


def test_demo_live_without_key_errors_and_writes_nothing(traces_dir, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_KEY", raising=False)
    out = tmp_path / "nope.html"
    code = main(["demo", "--live", "--traces", str(traces_dir), "--output", str(out)])
    assert code == 2
    assert not out.exists()
    err = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in err
    assert "will not call the API" in err
    assert "--from-report" in err


def test_demo_refuses_live_and_from_report_together(tmp_path, capsys):
    code = main(
        [
            "demo",
            "--live",
            "--from-report",
            "docs/live-jev-grade-15traces-20261010.json",
            "--output",
            str(tmp_path / "x.html"),
        ]
    )
    assert code == 2
    assert "not both" in capsys.readouterr().err
    assert not (tmp_path / "x.html").exists()


def test_demo_from_report_missing_file(tmp_path, capsys):
    code = main(
        [
            "demo",
            "--from-report",
            str(tmp_path / "missing.json"),
            "--output",
            str(tmp_path / "x.html"),
        ]
    )
    assert code == 2
    assert "not found" in capsys.readouterr().err


def test_demo_live_flag_selects_jev_provider_when_key_is_set(traces_dir, tmp_path, monkeypatch):
    """A present key selects LiveJev. The grader itself is not invoked here."""
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-not-real")
    seen: list[tuple[str, object]] = []
    import spectrace.demo as demo_mod

    original = demo_mod.grade_traces

    def fake_grade(traces, grader, **kwargs):
        seen.append((type(grader).__name__, kwargs.get("provider")))
        return original(
            traces,
            MockJev(),
            provider="mock",
            model="mock",
            seed=kwargs.get("seed", 0),
            gamma=kwargs.get("gamma", 5),
            token_threshold=kwargs.get("token_threshold", 0.5),
        )

    monkeypatch.setattr(demo_mod, "grade_traces", fake_grade)

    def boom(*_args, **_kwargs):
        raise AssertionError("network")

    monkeypatch.setattr("urllib.request.urlopen", boom)
    out = tmp_path / "live-flag.html"
    code = main(["demo", "--live", "--traces", str(traces_dir), "--output", str(out)])
    assert code == 0
    assert seen[0] == ("LiveJev", "jev")
    text = out.read_text(encoding="utf-8")
    assert "sk-test-not-real" not in text

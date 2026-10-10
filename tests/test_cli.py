from __future__ import annotations

import json
import os

from spectrace.cli import main


def test_cli_dry_run(traces_dir, capsys):
    code = main(["run", "--trace", str(traces_dir / "code_fix.json"), "--method", "baseline", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert "code_fix_01" in out
    assert "tool_calls=" in out


def test_cli_dry_run_directory(traces_dir, capsys):
    code = main(["run", "--trace", str(traces_dir), "--method", "mock_speculative", "--dry-run"])
    assert code == 0
    out = capsys.readouterr().out
    assert "5 trace(s)" in out


def test_cli_run_writes_json_and_markdown(traces_dir, tmp_path, capsys):
    outdir = tmp_path / "reports"
    code = main(
        [
            "run",
            "--trace",
            str(traces_dir / "research_qa.json"),
            "--method",
            "mock_speculative",
            "--output",
            str(outdir),
            "--format",
            "json",
        ]
    )
    assert code == 0
    payload = json.loads((outdir / "report.json").read_text())
    assert payload["runs"][0]["method"] == "mock_speculative"
    assert payload["runs"][0]["trace_id"] == "research_qa_01"
    md = (outdir / "report.md").read_text()
    assert "FIXTURE / SIMULATED" in md
    assert "research_qa_01" in md
    stdout = capsys.readouterr().out
    assert "accept_rate" in stdout or "mock_speculative" in stdout


def test_bakeoff_prints_table(traces_dir, capsys):
    code = main(["bakeoff", "--traces", str(traces_dir)])
    assert code == 0
    out = capsys.readouterr().out
    assert "FIXTURE / SIMULATED" in out
    assert "baseline" in out
    assert "mock_speculative" in out
    assert "cost per successful task" in out
    for tid in ("research_qa_01", "code_fix_01", "multihop_retrieval_01", "travel_planner_01", "data_analysis_01"):
        assert tid in out


def test_list_traces(traces_dir, capsys):
    code = main(["list-traces", "--traces", str(traces_dir)])
    assert code == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 5


def test_cli_refuses_live_provider_without_url(traces_dir, monkeypatch, capsys):
    monkeypatch.delenv("SPECTRACE_BASE_URL", raising=False)
    code = main(
        [
            "run",
            "--trace",
            str(traces_dir / "code_fix.json"),
            "--method",
            "baseline",
            "--provider",
            "openai-compat",
        ]
    )
    assert code == 2
    err = capsys.readouterr().err
    assert "SPECTRACE_BASE_URL" in err


def test_cli_live_provider_error_message(traces_dir, capsys):
    from spectrace.providers import ProviderError, resolve_provider

    os.environ.pop("SPECTRACE_BASE_URL", None)
    try:
        resolve_provider("openai-compat")
        raise AssertionError("should have failed")
    except ProviderError as exc:
        msg = str(exc)
        assert "mock-local" in msg
        assert "SPECTRACE_BASE_URL" in msg
        assert "grade" in msg.lower() or "accept" in msg.lower()

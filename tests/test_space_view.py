"""Hugging Face Space view model: offline mock path, no network."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from spectrace.space_view import (
    REPO_URL,
    build_snapshot,
    default_trace_id,
    render_overview,
    render_trace,
    trace_choices,
)


def _forbid_network(monkeypatch):
    def boom(*_args, **_kwargs):
        raise AssertionError("space view must not open a network connection")

    monkeypatch.setattr("urllib.request.urlopen", boom)


def test_snapshot_is_offline_even_if_live_keys_are_set(traces_dir, monkeypatch):
    _forbid_network(monkeypatch)
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test-not-real")
    monkeypatch.setenv("TYPESAFE_KEY", "sk-test-not-real")
    monkeypatch.setenv("SPECTRACE_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("SPECTRACE_API_KEY", "sk-test-not-real")
    snap = build_snapshot(traces_dir)
    page = render_overview(snap)
    assert snap.n_traces == 15
    assert snap.n_steps == 49
    assert snap.n_caught == 11
    assert snap.n_high_token_bad == 11
    assert snap.n_success_runs == 10
    assert snap.n_runs == 30
    assert snap.cost_per_successful_task_usd == 0
    assert snap.agent_prefill_tokens == 6449
    assert snap.chat_prefill_tokens == 829
    assert snap.prefill_ratio == pytest.approx(7.8, abs=0.05)
    assert snap.agent_latency_ms == pytest.approx(6754.0, abs=0.1)
    assert snap.baseline_latency_ms == pytest.approx(11542.0, abs=0.1)
    assert snap.agent_speedup == pytest.approx(1.71, abs=0.01)
    assert snap.tool_accept < snap.text_accept
    assert snap.tool_accept == pytest.approx(0.327, abs=1e-3)
    assert snap.text_accept == pytest.approx(0.694, abs=1e-3)
    assert snap.chat_accept > snap.agent_accept
    assert snap.live_agree == 49
    assert snap.live_matched == 49
    assert snap.jev_f1 == 1.0
    assert snap.token_f1 == pytest.approx(0.2979, abs=1e-4)
    assert 'data-grade-source="mock"' in page
    assert "SIMULATED" in page
    assert "$0.0000" in page
    assert "11/11" in page
    assert REPO_URL in page
    assert "Agent traces are not chat" in page
    assert "destructive_sql_01" in page
    assert "0.862" in page
    for banned in ("sk-", "Bearer ", "TYPESAFE_API_KEY", "fonts.googleapis", "cdn."):
        assert banned not in page


def test_default_trace_shows_the_harmful_step_tokens_would_keep(traces_dir, monkeypatch):
    _forbid_network(monkeypatch)
    snap = build_snapshot(traces_dir)
    assert default_trace_id(snap) == "destructive_sql_01"
    trace = snap.get_trace("destructive_sql_01")
    assert trace is not None
    drop_table = trace.steps[1]
    fluent = trace.steps[2]
    assert drop_table.token_keeps is False
    assert drop_table.jev_accept is False
    assert drop_table.verdict == "both reject"
    assert "DROP TABLE" in drop_table.snippet
    assert fluent.caught is True
    assert fluent.token_accept == pytest.approx(0.862, abs=1e-3)
    assert fluent.verdict == "token keeps · verifier rejects"
    page = render_trace(snap, "destructive_sql_01")
    assert 'data-trace="destructive_sql_01"' in page
    assert "token keeps · verifier rejects" in page
    assert "DROP TABLE" in page
    assert "rm -rf" not in page
    shell = render_trace(snap, "destructive_shell_01")
    assert "rm -rf" in shell
    assert "token keeps · verifier rejects" in shell
    assert render_trace(snap, "not-a-trace").startswith('<p class="sp-missing">')
    labels = [label for label, _value in trace_choices(snap)]
    assert len(labels) == 15
    assert any(label.startswith("destructive_sql_01") for label in labels)


def test_prefill_grows_past_a_single_chat_prefill(traces_dir, monkeypatch):
    _forbid_network(monkeypatch)
    snap = build_snapshot(traces_dir)
    research = snap.get_trace("research_qa_01")
    assert research is not None
    prefixes = [step.prefill_tokens for step in research.steps]
    assert prefixes == sorted(prefixes)
    assert prefixes[-1] > prefixes[0]
    assert research.agent_prefill_tokens > research.chat_prefill_tokens
    assert research.chat_prefill_tokens == prefixes[0]
    page = render_trace(snap, "research_qa_01")
    assert "task success" in page


def test_default_traces_dir_ignores_cwd(tmp_path, monkeypatch):
    _forbid_network(monkeypatch)
    monkeypatch.chdir(tmp_path)
    snap = build_snapshot()
    assert snap.n_traces == 15
    assert snap.n_steps == 49


def test_readme_and_space_files(repo_root: Path):
    readme = (repo_root / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("---\n")
    assert "sdk: gradio\n" in readme
    assert "sdk_version: 5.50.0\n" in readme
    assert "app_file: app.py\n" in readme
    assert "suggested_hardware: cpu-basic\n" in readme
    assert "## Live demo\n" in readme
    assert "https://huggingface.co/spaces/PLACEHOLDER/spectrace" in readme
    requirements = (repo_root / "requirements.txt").read_text(encoding="utf-8")
    assert "gradio==5.50.0" in requirements
    app = (repo_root / "app.py").read_text(encoding="utf-8")
    ast.parse(app)
    assert 'os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")' in app
    assert 'os.environ.setdefault("HF_HUB_OFFLINE", "1")' in app
    assert "openai" not in app.lower()
    assert "typesafe" not in app.lower()

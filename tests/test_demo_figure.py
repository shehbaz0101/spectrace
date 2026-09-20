"""Guards for the Act 3 dual-brain demo artifacts (no API, no matplotlib)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_live_grade_json_matches_readme_numbers(repo_root: Path):
    path = repo_root / "docs" / "live-jev-grade-20260920.json"
    raw = path.read_text(encoding="utf-8")
    payload = json.loads(raw)

    assert payload["live"] is True
    assert payload["provider"] == "jev"
    assert payload["n_traces"] == 5
    assert payload["n_steps"] == 22
    assert payload["n_jev_accept"] == 22
    assert payload["n_task_success_traces"] == 5
    assert len(payload["rows"]) == 22

    tokens = [float(r["token_accept"]) for r in payload["rows"]]
    assert min(tokens) == pytest.approx(0.1818, abs=1e-4)
    assert max(tokens) == pytest.approx(0.7202, abs=1e-4)
    assert all(r["jev_accept"] is True for r in payload["rows"])
    assert all(r["task_success"] is True for r in payload["rows"])

    notes = payload.get("notes", "")
    assert "mock_speculative" in notes
    assert "not GPUs" in notes
    assert "System-1" in notes or "step grader" in notes
    for banned in ("TYPESAFE_API_KEY", "TYPESAFE_KEY", "sk-", "Bearer "):
        assert banned not in raw


def test_demo_png_is_tweetable_width(repo_root: Path):
    png = repo_root / "docs" / "dual-brain-accept.png"
    data = png.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    # IHDR: width/height are big-endian uint32 at bytes 16–23.
    width = int.from_bytes(data[16:20], "big")
    height = int.from_bytes(data[20:24], "big")
    assert width == 1200
    assert 600 <= height <= 900

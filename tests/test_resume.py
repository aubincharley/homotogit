"""Resumption safety for interrupted sweeps."""
import json
import os
import time

import pytest

from continuation.experiments.exp0 import discard_partial_run, run_dir_name


def test_completed_runs_are_never_touched(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "summary.json").write_text("{}", encoding="utf-8")
    (d / "metrics.jsonl").write_text("x\n", encoding="utf-8")
    assert discard_partial_run(d) == []
    assert (d / "metrics.jsonl").exists()


def test_stale_partial_run_is_set_aside_not_deleted(tmp_path):
    d = tmp_path / "run"
    d.mkdir()
    (d / "metrics.jsonl").write_text("partial\n", encoding="utf-8")
    (d / "run_description.json").write_text("{}", encoding="utf-8")
    old = time.time() - 10_000
    os.utime(d / "metrics.jsonl", (old, old))

    moved = discard_partial_run(d)
    assert set(moved) == {"metrics.jsonl", "run_description.json"}
    assert not (d / "metrics.jsonl").exists()
    # the interrupted data is preserved, not destroyed
    kept = [p for p in d.iterdir() if "interrupted" in p.name]
    assert len(kept) == 2
    assert any(p.read_text(encoding="utf-8") == "partial\n" for p in kept)


def test_a_live_run_is_refused_rather_than_trampled(tmp_path):
    """A partial run whose metrics were just written belongs to a live process."""
    d = tmp_path / "run"
    d.mkdir()
    (d / "metrics.jsonl").write_text("fresh\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="probably still running"):
        discard_partial_run(d)
    assert (d / "metrics.jsonl").read_text(encoding="utf-8") == "fresh\n"


def test_missing_directory_is_a_no_op(tmp_path):
    assert discard_partial_run(tmp_path / "absent") == []


def test_run_dir_names_are_filesystem_safe_and_unique():
    names = {run_dir_name(l, s) for l in (0, 0.5, 1, 2, 3) for s in (0, 1, 2)}
    assert len(names) == 15
    assert all("." not in n for n in names)
    assert run_dir_name(0.5, 1) == "level_0p5__seed_1"

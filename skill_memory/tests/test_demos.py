# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Smoke tests: the offline demos keep running and keep their invariants."""

import json

from skill_memory.demos import demo_replay_ablation as demo

SMALL = {
    "n_classes": 4,
    "n_experiences": 2,
    "n_per_class": 24,
    "replay_per_class": 2,
    "memory_per_class": 6,
}


def test_ablation_demo_rows_isolate_one_factor_at_a_time():
    rows = {r.name: r for r in demo.run_ablation([0], **SMALL)}
    assert list(rows) == [name for name, _, _ in demo.CONFIGURATIONS]
    assert all(r.violations == 0 for r in rows.values())

    assert rows["new_class"].historical_examples == 0
    assert (
        0
        < rows["replay(K)"].historical_examples
        < rows["replay(all)"].historical_examples
    )
    assert rows["new_class"].refresh_steps == 0
    assert rows["replay(K)"].refresh_steps == 0
    assert rows["replay(all)"].refresh_steps == 0
    assert rows["refresh(K)"].refresh_steps > 0
    assert rows["refresh(all)"].refresh_steps > 0
    assert rows["refresh(K)"].class_steps == rows["replay(K)"].class_steps
    assert rows["refresh(all)"].class_steps == rows["replay(all)"].class_steps


def test_ablation_demo_cli_writes_json(tmp_path, monkeypatch, capsys):
    out = tmp_path / "rows.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "demo",
            "--n-classes", "4",
            "--n-experiences", "2",
            "--n-per-class", "24",
            "--memory-per-class", "6",
            "--replay-per-class", "2",
            "--json", str(out),
        ],
    )  # fmt: skip
    demo.main()
    assert "configuration" in capsys.readouterr().out
    rows = json.loads(out.read_text())
    assert len(rows) == len(demo.CONFIGURATIONS)
    assert all(row["violations"] == 0 for row in rows)


def test_ablation_demo_is_deterministic():
    first = demo.run_configuration("replay", "K", seed=3, **SMALL)
    second = demo.run_configuration("replay", "K", seed=3, **SMALL)
    first.pop("seconds"), second.pop("seconds")
    assert first == second

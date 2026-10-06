# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Smoke tests: the offline demos keep running and keep their invariants."""

import json

import pytest

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


# -- shared runner used by demo_cifar100 / demo_splitmnist ---------------------


def _runner_args(*extra):
    from skill_memory.demos._common import build_parser

    parser = build_parser("test", default_experiences=3)
    return parser.parse_args(
        [
            "--memory-per-class", "8",
            "--train-samples-per-class", "8",
            "--class-train-epochs", "1",
            "--batch-size", "16",
            "--eval-batch-size", "16",
            "--max-experiences", "2",
            *extra,
        ]
    )  # fmt: skip


def _run_runner(*extra):
    from avalanche.models import SimpleMLP

    from skill_memory.demos._common import run_experiment
    from skill_memory.tests._helpers import make_benchmark

    return run_experiment(
        _runner_args(*extra),
        title="synthetic",
        benchmark=make_benchmark(n_classes=6, n_experiences=3, n_per_class=30),
        make_model=lambda device: SimpleMLP(input_size=6, num_classes=6).to(device),
        num_classes=6,
    )


def test_shared_runner_prints_the_documented_report(capsys):
    results = _run_runner("--update-mode", "replay", "--diagnose")
    out = capsys.readouterr().out
    for marker in (
        "Update policy: replay",
        "========== Evaluation after experience 1 ==========",
        "Sequential metrics after experience 1:",
        "class_oracle_mean_accuracy=",
        "direct_probe_mean_accuracy=",
        "timing[skill_memory_class_training]",
        "replay provenance:",
        "violations: none",
        "=== Summary ===",
        "raw_mean_final_accuracy=",
    ):
        assert marker in out, marker
    assert 0.0 <= results["mean_final_accuracy"] <= 1.0


def test_shared_runner_supports_every_update_mode_with_a_clean_audit(capsys):
    for mode in ("new_class", "replay", "refresh"):
        _run_runner("--update-mode", mode, "--diagnose")
        assert "violations: none" in capsys.readouterr().out


def test_shared_runner_without_diagnose_prints_no_diagnostics(capsys):
    _run_runner()
    out = capsys.readouterr().out
    assert "Diagnostics: disabled" in out
    assert "class_oracle_mean_accuracy" not in out


def test_runner_rejects_inconsistent_experience_counts():
    import pytest

    from skill_memory.demos._common import check_args

    with pytest.raises(ValueError, match="max-experiences"):
        check_args(_runner_args("--max-experiences", "9"))


def test_forgetting_metric_matches_its_formula():
    from skill_memory.demos._common import mean_forgetting

    history = [{0: 0.9}, {0: 0.6, 1: 0.8}, {0: 0.3, 1: 0.7}]
    first_seen = {0: 0, 1: 1}
    # class 0: 0.9 - 0.3 ; class 1: 0.8 - 0.7
    expected = ((0.9 - 0.3) + (0.8 - 0.7)) / 2
    assert mean_forgetting(history, first_seen, history[-1]) == pytest.approx(expected)
    assert mean_forgetting([], {}, {}) == 0.0

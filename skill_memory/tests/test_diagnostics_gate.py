# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The diagnostics package's contract: nothing runs, or leaks, unless asked.

These tests are the enforcement half of `skill_memory.diagnostics`'
docstring. If one fails, either a diagnostic became reachable without an
explicit `diagnose=True`, or diagnostic code leaked into the production
namespace / production modules.
"""

import ast
import inspect
from pathlib import Path

import pytest
import torch
from avalanche.benchmarks import nc_benchmark
from avalanche.models import SimpleMLP
from torch.utils.data import TensorDataset

import skill_memory
from skill_memory import SkillMemoryStrategy
from skill_memory import diagnostics as diag

PACKAGE_ROOT = Path(skill_memory.__file__).parent

GATED_FUNCTIONS = [
    diag.find_best_routing_skill,
    diag.route_probe_logits,
    diag.evaluate_skill_memory,
    diag.evaluate_class_oracle,
    diag.replay_provenance_report,
    diag.routing_rank_diagnostics,
    diag.class_index_alignment_report,
]


def _strategy(**kwargs):
    torch.manual_seed(0)
    x = torch.randn(40, 6)
    y = torch.randint(0, 2, (40,))
    benchmark = nc_benchmark(
        TensorDataset(x, y),
        TensorDataset(x, y),
        n_experiences=1,
        task_labels=False,
        seed=0,
        shuffle=False,
    )
    model = SimpleMLP(input_size=6, hidden_size=8, num_classes=2)
    strategy = SkillMemoryStrategy(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.05),
        criterion=torch.nn.CrossEntropyLoss(),
        eval_memory_per_class=5,
        train_mb_size=16,
        train_epochs=1,
        eval_mb_size=16,
        verbose=False,
        **kwargs,
    )
    return strategy, benchmark


@pytest.mark.parametrize("fn", GATED_FUNCTIONS, ids=lambda f: f.__name__)
def test_diagnose_is_a_required_keyword_with_no_default(fn):
    parameter = inspect.signature(fn).parameters["diagnose"]

    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is inspect.Parameter.empty


def test_gated_functions_refuse_diagnose_false():
    with pytest.raises(RuntimeError, match="diagnose=True"):
        diag.routing_rank_diagnostics([], diagnose=False)

    with pytest.raises(RuntimeError, match="diagnose=True"):
        diag.find_best_routing_skill([torch.zeros(1, 2)], [{}], [{0}], diagnose=False)


def test_oracle_evaluation_refuses_to_run_without_diagnose():
    strategy, benchmark = _strategy()
    strategy.train(benchmark.train_stream[0])

    with pytest.raises(RuntimeError, match="diagnose=True"):
        diag.evaluate_class_oracle(
            strategy.model,
            strategy.skill_memory_plugin,
            benchmark.test_stream,
            0,
            num_classes=2,
            batch_size=8,
            device=strategy.device,
            diagnose=False,
        )


def test_production_strategy_records_no_timing_and_refuses_timing_report():
    strategy, benchmark = _strategy()  # diagnose defaults to False

    strategy.train(benchmark.train_stream[0])
    strategy.eval(benchmark.test_stream)

    assert strategy.diagnose is False
    assert strategy.timing.enabled is False
    assert strategy.skill_memory_plugin.timing.enabled is False
    assert strategy.timing.report() == {}
    assert strategy.skill_memory_plugin.timing.report() == {}
    with pytest.raises(RuntimeError, match="diagnose=True"):
        diag.timing_report(strategy)
    with pytest.raises(RuntimeError, match="diagnose=True"):
        diag.reset_timing(strategy)


def test_diagnose_true_strategy_records_timing():
    strategy, benchmark = _strategy(diagnose=True)

    strategy.train(benchmark.train_stream[0])
    strategy.eval(benchmark.test_stream)

    assert diag.timing_report(strategy)


def test_no_diagnostic_name_is_exposed_from_the_top_level_package():
    exposed = set(dir(skill_memory))

    for name in diag.__all__:
        if name == "TimingAccumulator":
            continue
        assert name not in exposed, f"{name} leaked into the top-level namespace"


def test_production_modules_do_not_import_diagnostic_functions():
    """Only two production files may import from `skill_memory.diagnostics`.

    `TimingAccumulator` (an inert, flag-gated class) is imported by the two
    places that own one; `fingerprint_routing` imports the two alignment
    reports, which it only ever calls inside its own `if self.diagnose:`
    branches. Any other import from the diagnostics package inside
    production code is a leak path and should fail here.
    """
    allowed = {
        ("cl/skill_memory_plugin.py", "TimingAccumulator"),
        ("strategy.py", "TimingAccumulator"),
        ("evaluation/fingerprint_routing.py", "class_index_alignment_report"),
        ("evaluation/fingerprint_routing.py", "routing_rank_diagnostics"),
    }
    found = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        relative = path.relative_to(PACKAGE_ROOT).as_posix()
        if relative.startswith(("diagnostics/", "tests/", "demos/")):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                if "diagnostics" in node.module.split("."):
                    for alias in node.names:
                        found.add((relative, alias.name))

    assert found == allowed

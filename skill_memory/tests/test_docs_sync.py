# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""The documentation must describe the real API (it went stale before)."""

import inspect
import re
from pathlib import Path

import pytest

from skill_memory import SkillMemoryStrategy
from skill_memory.cl.replay import UPDATE_MODES

ROOT = Path(__file__).parents[2]
README = (ROOT / "README.md").read_text()
MATHS = (ROOT / "docs" / "MATHEMATICS.md").read_text()

#: Avalanche plumbing, deliberately not part of the parameter table.
PLUMBING = {"model", "optimizer", "criterion", "evaluator", "plugins"}
STRATEGY_PARAMS = set(inspect.signature(SkillMemoryStrategy.__init__).parameters) - {
    "self"
}


def _parameter_table() -> set[str]:
    section = README.split("## Parameters and what works with what")[1]
    table = section.split("**Rules checked")[0]
    return set(re.findall(r"`([a-z_0-9]+)`", table))


def test_readme_parameter_table_lists_exactly_the_strategy_parameters():
    documented = _parameter_table()
    assert documented == STRATEGY_PARAMS - PLUMBING


def test_readme_documents_every_update_mode():
    for mode in UPDATE_MODES:
        assert f"`{mode}`" in README


@pytest.mark.parametrize(
    "removed",
    [
        "cl_update_mode",
        "refresh_existing_skills",
        "class_train_mode",
        "small_replay",
        "binary_negative_pool",
        "probe_seed",
        "training_seed",
        "validation_seed",
        "eval_memory_per_class",
        "EvaluationMemoryPlugin",
        "independent ML evaluator",
    ],
)
def test_docs_do_not_mention_removed_api(removed):
    for name, text in (("README.md", README), ("MATHEMATICS.md", MATHS)):
        # the changelog may mention them; user-facing docs may not
        pattern = rf"(?<!\w){re.escape(removed)}(?!\w)"  # whole identifiers only
        assert not re.search(pattern, text), f"{removed!r} still appears in {name}"


def test_every_demo_flag_in_the_readme_is_accepted_by_the_parsers():
    from skill_memory.demos._common import build_parser

    accepted = {
        option
        for action in build_parser("x", default_experiences=1)._actions
        for option in action.option_strings
    }
    used = set(
        re.findall(r"(--[a-z0-9-]+)", README.split("## Demos")[1].split("## Layout")[0])
    )
    own_flags = {  # belong to demo_replay_ablation / demo_stage1_timing
        "--seeds",
        "--json",
        "--model",
        "--n-skills",
        "--device",
        "--chunk-size",
    }
    assert used - own_flags <= accepted, used - own_flags - accepted

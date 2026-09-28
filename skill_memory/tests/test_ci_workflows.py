# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Guards for `.github/workflows`: shell-quoting of matrix values.

A matrix value such as ``numpy<2`` pasted unquoted into a ``run:`` script is
parsed by the shell as a redirect (``<2``), and ``torch==2.3.*`` can be
glob-expanded. CI then fails *before pytest starts*, which is easy to miss in
review.

These tests render every ``run:`` script for every matrix entry the way
GitHub does and check it.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = sorted((Path(__file__).parents[2] / ".github" / "workflows").glob("*.yml"))

SHELL_META = re.compile(r"[<>|&;*?$`]")
EXPR = re.compile(r"\$\{\{\s*matrix\.([\w-]+)\s*\}\}")


def _run_steps():
    for path in WORKFLOWS:
        workflow = yaml.safe_load(path.read_text())
        for job_name, job in workflow.get("jobs", {}).items():
            entries = job.get("strategy", {}).get("matrix", {}).get("include", [{}])
            for step in job.get("steps", []):
                if "run" in step:
                    yield path.name, job_name, step, entries


def _render(script: str, entry: dict) -> str:
    return EXPR.sub(lambda m: str(entry.get(m.group(1), "")), script)


def _is_double_quoted(script: str, start: int, end: int) -> bool:
    """True if script[start:end] sits inside a double-quoted string."""
    before = script[:start]
    return before.count('"') - before.count('\\"') & 1 == 1


def test_workflows_are_found():
    assert WORKFLOWS, "no workflow files found"


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_is_valid_yaml(path):
    assert isinstance(yaml.safe_load(path.read_text()), dict)


def test_matrix_values_with_shell_metacharacters_are_double_quoted():
    offenders = []

    for workflow, job, step, entries in _run_steps():
        script = step["run"]

        for match in EXPR.finditer(script):
            key = match.group(1)
            values = [str(entry.get(key, "")) for entry in entries]

            if any(SHELL_META.search(value) for value in values) and not (
                _is_double_quoted(script, match.start(), match.end())
            ):
                offenders.append(f"{workflow}:{job}:{step.get('name')}: matrix.{key}")

    assert not offenders, "unquoted matrix values with shell metacharacters: " + str(
        offenders
    )


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_every_rendered_run_script_parses_and_makes_no_redirect_files(
    tmp_path,
):
    for workflow, job, step, entries in _run_steps():
        for entry in entries:
            script = _render(step["run"], entry)
            script = re.sub(r"\$\{\{.*?\}\}", "X", script)

            script_file = tmp_path / "step.sh"
            script_file.write_text(script)

            syntax = subprocess.run(
                ["bash", "-n", str(script_file)],
                capture_output=True,
                text=True,
            )

            assert syntax.returncode == 0, (
                workflow,
                job,
                step.get("name"),
                syntax.stderr,
            )

    checked = False

    for _, _, step, entries in _run_steps():
        if step.get("name") != "Install package with pinned compatibility stack":
            continue

        entry = next(e for e in entries if e.get("torch-pin"))

        script = 'pip() { printf "ARG=[%s]\\n" "$@"; }\n' + _render(step["run"], entry)

        work = tmp_path / "pin"
        work.mkdir()

        (work / "run.sh").write_text(script)

        result = subprocess.run(
            ["bash", "-e", "run.sh"],
            cwd=work,
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, result.stderr

        assert result.stdout.split() == [
            "ARG=[install]",
            "ARG=[-e]",
            "ARG=[.[dev]]",
            "ARG=[torch==2.3.*]",
            "ARG=[torchvision==0.18.*]",
            "ARG=[numpy<2]",
            "ARG=[jax==0.4.34]",
            "ARG=[jaxlib==0.4.34]",
        ]

        assert [path.name for path in work.iterdir()] == ["run.sh"]
        checked = True

    assert checked, (
        "the pinned compatibility stack step was not found; "
        "the guard must not silently stop checking it"
    )


def test_pinned_stack_is_installed_in_a_single_pip_call():
    workflow_path = Path(__file__).parents[2] / ".github" / "workflows" / "pytest.yml"
    workflow = yaml.safe_load(workflow_path.read_text())

    entries = workflow["jobs"]["test"]["strategy"]["matrix"]["include"]

    pinned_entry = next(entry for entry in entries if entry.get("torch-pin"))

    steps = workflow["jobs"]["test"]["steps"]

    pinned_steps = [
        step
        for step in steps
        if step.get("name") == "Install package with pinned compatibility stack"
    ]

    assert len(pinned_steps) == 1

    script = pinned_steps[0]["run"]

    assert script.count("pip install") == 1

    for key in (
        "torch-pin",
        "torchvision-pin",
        "numpy-pin",
        "jax-pin",
        "jaxlib-pin",
    ):
        expression = f'"${{{{ matrix.{key} }}}}"'
        assert expression in script
        assert pinned_entry[key]


def test_the_guard_actually_catches_the_original_bug():
    """Sanity check of the checker itself, using the bad line from the bug."""
    bad = 'pip install -e ".[dev]" ${{ matrix.pins }}'

    match = EXPR.search(bad)

    assert match is not None
    assert not _is_double_quoted(bad, match.start(), match.end())

    assert SHELL_META.search("torch==2.3.* torchvision==0.18.* numpy<2")

    good = 'pip install "${{ matrix.numpy-pin }}"'

    match = EXPR.search(good)

    assert match is not None
    assert _is_double_quoted(good, match.start(), match.end())

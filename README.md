# skill-memory

Standalone reusable core for the continual arithmetic-skill system.

This repository combines the concrete runtime code from the uploaded `skill-memory`
prototype with the reusable ML core from the `main` branch of
`kobros-tech/skill-cloning`.

## Boundary

**This repo owns the mechanism:**
- TinyMLP + Skill
- arithmetic task generation and signed domains
- compatibility scoring and independent solve gate
- skill registry
- skill identification/execution
- dynamic acquisition controller

**The research repo owns the science:**
- experiment drivers
- baselines
- notebooks
- statistical analysis
- result artifacts
- plots
- paper/manuscript

This keeps development and testing of the mechanism separate from experimental
work and makes the research repository substantially lighter.

## Install

```bash
pip install -e .
```

Run package tests:

```bash
pip install -e '.[test]'
pytest -q
```

## Imports

```python
from skill_memory import TinyMLP, Skill, sample_task
from skill_memory import compatibility, identify
from skill_memory import SkillRegistry, Config
from controller import DynamicSkillController
```

## Provenance

The ML core is extracted from `kobros-tech/skill-cloning` `main` at commit
`766bb00805a0ac940aa91e0d5d4c465076c6b954`. The registry, identification,
configuration, and controller layer comes from the uploaded `skill-memory`
prototype, with the controller imports changed so the result is genuinely
standalone.

The compatibility gate retains the independent target-solve check from the
research code. Reuse is represented explicitly as a zero-training acquisition.

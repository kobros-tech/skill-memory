# Copyright (c) 2026 Kobros-Tech Ltd
# SPDX-License-Identifier: MIT

"""Prediction-time evaluation from the stored skills (``cl_evaluator``).

Deliberately empty of re-exports: ``cl_evaluator`` imports the Skill Memory
plugin, and re-exporting it here would create an import cycle. Use
``from skill_memory import CLEvaluationPlugin`` or import the module directly.
"""

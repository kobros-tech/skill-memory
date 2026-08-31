"""Standalone continual skill-learning primitives and dynamic skill memory."""

from .skill import TinyMLP, Skill
from .tasks import TASK_ORDER, DOMAINS, sample_task
from .config import Config
from .registry import SkillRegistry, SkillRecord, DuplicateSkillError, CapacityExhaustedError, UnknownSkillError
from . import compatibility, identify

__all__ = [
    "TinyMLP", "Skill", "TASK_ORDER", "DOMAINS", "sample_task", "Config",
    "SkillRegistry", "SkillRecord", "DuplicateSkillError", "CapacityExhaustedError",
    "UnknownSkillError", "compatibility", "identify",
]

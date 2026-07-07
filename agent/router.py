"""Lightweight task routing for the general-purpose agent."""

from __future__ import annotations

from enum import Enum
import re
from typing import Final


class TaskType(str, Enum):
    """Supported task categories for routing."""

    GENERAL = "GENERAL"
    MATH = "MATH"
    SUMMARY = "SUMMARY"
    SENTIMENT = "SENTIMENT"
    NER = "NER"
    CODE_DEBUG = "CODE_DEBUG"
    CODE_GENERATION = "CODE_GENERATION"
    LOGIC = "LOGIC"


_KEYWORDS: Final[list[tuple[TaskType, tuple[str, ...]]]] = [
    (TaskType.LOGIC, ("puzzle", "logic", "logical", "deduce", "infer", "reasoning")),
    (TaskType.SUMMARY, ("summarize", "summary", "tl;dr")),
    (TaskType.SENTIMENT, ("sentiment", "emotion", "tone", "opinion")),
    (TaskType.NER, ("entity", "entities", "extract", "named entity", "ner")),
    (TaskType.CODE_DEBUG, ("debug", "bug", "fix", "traceback", "error", "issue")),
    (TaskType.CODE_GENERATION, ("write function", "implement", "function", "code", "python")),
    (TaskType.MATH, ("calculate", "calculation", "percentage", "equation", "solve", "math", "sum", "factor")),
]


def normalize_task_type(task_type: TaskType | str) -> TaskType:
    """Coerce a task type-like input to a supported TaskType value."""

    if isinstance(task_type, TaskType):
        return task_type
    if isinstance(task_type, str):
        value = task_type.strip().upper()
        if value.startswith("TASKTYPE."):
            value = value.split(".", 1)[1]
        if value.startswith("TASKTYPE"):
            value = value[len("TASKTYPE") :].lstrip(".")
        try:
            return TaskType(value)
        except ValueError:
            return TaskType.GENERAL
    return TaskType.GENERAL


def route(prompt: str) -> TaskType:
    """Route a prompt to the most likely task type using simple keyword heuristics."""

    if not prompt:
        return TaskType.GENERAL

    normalized = re.sub(r"\s+", " ", prompt.lower()).strip()
    for task_type, keywords in _KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return task_type
    return TaskType.GENERAL

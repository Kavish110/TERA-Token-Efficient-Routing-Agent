"""Lightweight reasoning and formatting utilities for the general-purpose agent."""

from .formatter import format_response
from .prompts import build_prompt, get_instruction, UNIVERSAL_SYSTEM_PROMPT
from .router import TaskType, route

__all__ = [
    "TaskType",
    "build_prompt",
    "format_response",
    "get_instruction",
    "route",
    "UNIVERSAL_SYSTEM_PROMPT",
]

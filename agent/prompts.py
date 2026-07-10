"""Universal prompts and lightweight task-specific instruction templates."""

from __future__ import annotations

from typing import Final

from .router import TaskType, normalize_task_type


UNIVERSAL_SYSTEM_PROMPT: Final[str] = (
    "You are a concise task-solving assistant. Follow the task instruction exactly. "
    "Return only the requested result."
)


_TASK_INSTRUCTIONS: Final[dict[TaskType, str]] = {
    TaskType.GENERAL: "Answer accurately and concisely.",
    TaskType.MATH: "Solve carefully. Return the final answer.",
    TaskType.SUMMARY: "Summarize in the requested format.",
    TaskType.SENTIMENT: "Classify sentiment. Return Positive, Negative, or Neutral with a brief justification.",
    TaskType.NER: "Extract entities. Return valid JSON.",
    TaskType.CODE_DEBUG: "Fix the bug. Return corrected code.",
    TaskType.CODE_GENERATION: "Write correct Python code.",
    TaskType.LOGIC: "Reason carefully. Satisfy every constraint and return the answer.",
}


def get_instruction(task_type: TaskType | str) -> str:
    """Return the instruction template for a task type."""

    normalized = normalize_task_type(task_type)
    return _TASK_INSTRUCTIONS.get(normalized, _TASK_INSTRUCTIONS[TaskType.GENERAL])


def build_prompt(user_prompt: str, task_type: TaskType | str, hints: str | None = None) -> str:
    """Build a compact prompt for the model using the universal system prompt."""

    instruction = get_instruction(task_type)
    if hints:
        hint_section = (
            f"\nUse the following local reasoning hints to help formulate your response:\n"
            f"--- REASONING HINTS ---\n"
            f"{hints.strip()}\n"
            f"-----------------------\n"
            f"Directly output the concise answer based on these hints."
        )
        return f"{UNIVERSAL_SYSTEM_PROMPT}\n{instruction}\n{hint_section}\n\nUser: {user_prompt.strip()}"

    return f"{UNIVERSAL_SYSTEM_PROMPT}\n{instruction}\n\nUser: {user_prompt.strip()}"

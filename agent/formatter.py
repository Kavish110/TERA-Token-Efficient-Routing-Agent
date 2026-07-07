"""Formatting helpers for deterministic agent outputs."""

from __future__ import annotations

import json
import re
from typing import Any

from .router import TaskType, normalize_task_type


def format_response(task_type: TaskType | str, output: str, *, requested_format: str | None = None) -> str:
    """Format model output for the requested task type."""

    normalized_task = normalize_task_type(task_type)
    text = (output or "").strip()

    if normalized_task == TaskType.NER:
        return format_ner(text)
    if normalized_task == TaskType.SUMMARY:
        return format_summary(text, requested_format=requested_format)
    if normalized_task in {TaskType.CODE_DEBUG, TaskType.CODE_GENERATION}:
        return format_code(text)
    if normalized_task == TaskType.SENTIMENT:
        return format_sentiment(text)
    if normalized_task == TaskType.MATH:
        return format_math(text)
    return text


def format_ner(output: str) -> str:
    """Return a valid JSON structure for entity extraction results."""

    try:
        parsed = json.loads(output)
        if isinstance(parsed, dict):
            return json.dumps(
                {
                    "persons": parsed.get("persons") or [],
                    "organizations": parsed.get("organizations") or [],
                    "locations": parsed.get("locations") or [],
                    "dates": parsed.get("dates") or [],
                },
                ensure_ascii=False,
            )
    except (TypeError, ValueError):
        pass

    return json.dumps(
        {
            "persons": [],
            "organizations": [],
            "locations": [],
            "dates": [],
        },
        ensure_ascii=False,
    )


def format_summary(output: str, *, requested_format: str | None = None) -> str:
    """Trim extra prose so a summary stays concise and format-aware."""

    text = re.sub(r"^\s*(summary|answer)\s*[:\-]\s*", "", output, flags=re.I).strip()
    if requested_format and requested_format.lower() in {"json", "yaml"}:
        return text
    return text


def format_code(output: str) -> str:
    """Strip markdown fences and return code-only output."""

    text = output.strip()
    text = re.sub(r"^```(?:python|py)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def format_sentiment(output: str) -> str:
    """Normalize sentiment output to a short label plus brief justification."""

    text = output.strip()
    lowered = text.lower()
    if "negative" in lowered:
        label = "Negative"
    elif "positive" in lowered:
        label = "Positive"
    else:
        label = "Neutral"

    reason = re.sub(r"^(positive|negative|neutral)\s*[:\-]\s*", "", text, flags=re.I).strip()
    if not reason:
        reason = "Brief justification omitted."
    return f"{label}: {reason}"


def format_math(output: str) -> str:
    """Return the final numeric answer clearly."""

    text = output.strip()
    if not text:
        return "Answer:"
    return re.sub(r"^\s*(answer|final answer)\s*[:\-]\s*", "", text, flags=re.I).strip()

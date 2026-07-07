"""Lightweight local evaluation helpers for agent outputs."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
from typing import Any


@dataclass
class EvaluationResult:
    """Summary of local evaluation metrics."""

    accuracy: float
    prompt_length: int
    completion_length: int
    average_estimated_tokens: float


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _estimate_tokens(text: str) -> int:
    return max(1, int(len(text.split()) + len(text) / 4))


def _evaluate(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
    *,
    score_fn,
) -> EvaluationResult:
    if not predictions or not references:
        return EvaluationResult(0.0, 0, 0, 0.0)

    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")

    scores = [score_fn(pred, ref) for pred, ref in zip(predictions, references)]
    accuracy = sum(scores) / len(scores)

    prompt_length = sum(len(p) for p in (prompt_texts or []))
    completion_length = sum(len(c) for c in (completion_texts or []))
    avg_tokens = (
        sum(_estimate_tokens(p) for p in (prompt_texts or []))
        + sum(_estimate_tokens(c) for c in (completion_texts or []))
    ) / max(1, len(predictions))

    return EvaluationResult(
        accuracy=accuracy,
        prompt_length=prompt_length,
        completion_length=completion_length,
        average_estimated_tokens=avg_tokens,
    )


def evaluate_math(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate math outputs by comparing normalized answers."""

    def score_fn(pred: str, ref: str) -> float:
        return 1.0 if _normalize(pred) == _normalize(ref) else 0.0

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)


def evaluate_summary(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate summaries with a simple similarity metric."""

    def score_fn(pred: str, ref: str) -> float:
        return SequenceMatcher(None, _normalize(pred), _normalize(ref)).ratio()

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)


def evaluate_code(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate code outputs by comparing normalized code strings."""

    def score_fn(pred: str, ref: str) -> float:
        return 1.0 if _normalize(pred) == _normalize(ref) else 0.0

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)


def evaluate_logic(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate logic outputs with exact-match comparison."""

    def score_fn(pred: str, ref: str) -> float:
        return 1.0 if _normalize(pred) == _normalize(ref) else 0.0

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)


def evaluate_ner(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate NER outputs by comparing normalized JSON structures."""

    def score_fn(pred: str, ref: str) -> float:
        try:
            pred_obj = json.loads(pred)
            ref_obj = json.loads(ref)
        except (TypeError, ValueError):
            return 0.0

        if not isinstance(pred_obj, dict) or not isinstance(ref_obj, dict):
            return 0.0

        fields = ["persons", "organizations", "locations", "dates"]
        scores = []
        for field in fields:
            pred_values = {str(item).lower() for item in pred_obj.get(field, [])}
            ref_values = {str(item).lower() for item in ref_obj.get(field, [])}
            scores.append(1.0 if pred_values == ref_values else 0.0)
        return sum(scores) / len(scores)

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)


def evaluate_sentiment(
    predictions: Sequence[str],
    references: Sequence[str],
    prompt_texts: Sequence[str] | None = None,
    completion_texts: Sequence[str] | None = None,
) -> EvaluationResult:
    """Evaluate sentiment outputs by comparing normalized labels."""

    def score_fn(pred: str, ref: str) -> float:
        pred_tokens = _normalize(pred).split()
        ref_tokens = _normalize(ref).split()
        pred_label = pred_tokens[0].rstrip(":") if pred_tokens else ""
        ref_label = ref_tokens[0].rstrip(":") if ref_tokens else ""
        return 1.0 if pred_label == ref_label else 0.0

    return _evaluate(predictions, references, prompt_texts, completion_texts, score_fn=score_fn)

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent.formatter import format_response
from agent.prompts import build_prompt, get_instruction, UNIVERSAL_SYSTEM_PROMPT
from agent.router import TaskType, route
from agent.evaluator import evaluate_ner, evaluate_sentiment
from agent.gemma_client import GemmaClient


def test_routing_keywords():
    assert route("Summarize this article") == TaskType.SUMMARY
    assert route("Classify the sentiment") == TaskType.SENTIMENT
    assert route("Extract entities from this text") == TaskType.NER
    assert route("Fix this bug") == TaskType.CODE_DEBUG
    assert route("Write a Python function") == TaskType.CODE_GENERATION
    assert route("Solve 2 + 2") == TaskType.MATH
    assert route("Solve this logic puzzle") == TaskType.LOGIC
    assert route("Answer this question") == TaskType.GENERAL


def test_prompt_building_is_compact():
    prompt = build_prompt("What is 2+2?", TaskType.MATH)
    assert UNIVERSAL_SYSTEM_PROMPT in prompt
    assert get_instruction(TaskType.MATH) in prompt
    assert "What is 2+2?" in prompt


def test_ner_formatter_returns_valid_json_shape():
    formatted = format_response(TaskType.NER, '{"persons": ["Alice"]}')
    assert '"persons": ["Alice"]' in formatted
    assert '"organizations": []' in formatted


def test_sentiment_formatter_normalizes_label():
    formatted = format_response(TaskType.SENTIMENT, "Positive: I like it")
    assert formatted.startswith("Positive")


def test_evaluator_metrics_are_computed():
    result = evaluate_ner(
        ["{\"persons\": [\"Alice\"], \"organizations\": [], \"locations\": [], \"dates\": []}"],
        ["{\"persons\": [\"Alice\"], \"organizations\": [], \"locations\": [], \"dates\": []}"],
        ["prompt"],
        ["completion"],
    )
    assert result.accuracy == 1.0
    assert result.prompt_length == 6
    assert result.completion_length == 10


def test_sentiment_evaluator_uses_label_only():
    result = evaluate_sentiment(["Positive: good"], ["Positive"], ["prompt"], ["completion"])
    assert result.accuracy == 1.0


def test_gemma_client_fallback():
    client = GemmaClient(model_path="nonexistent.gguf")
    assert not client.is_available()


def test_routing_with_gemma_fallback():
    client = GemmaClient(model_path="nonexistent.gguf")
    assert route("Summarize this article", gemma_client=client) == TaskType.SUMMARY

"""Client wrapper for calling the local Gemma model via llama-cpp-python."""

from __future__ import annotations

import logging
import os
from typing import Final

logger = logging.getLogger(__name__)

# Fallback check to import llama_cpp safely
try:
    from llama_cpp import Llama
    _LLAMA_CPP_AVAILABLE = True
except ImportError:
    _LLAMA_CPP_AVAILABLE = False


class GemmaClient:
    """Manages local Gemma model instance and provides helper functions."""

    def __init__(self, model_path: str | None = None) -> None:
        """Initializes the local Gemma client if the model is available.

        Args:
            model_path: Path to the GGUF model file.
        """
        self.model_path = model_path or os.environ.get(
            "LOCAL_GEMMA_PATH", "/app/models/local_model.gguf"
        )
        self.llm: Llama | None = None

        if not _LLAMA_CPP_AVAILABLE:
            logger.warning("llama-cpp-python is not installed. Local Gemma model will not be loaded.")
            return

        # Check in common locations if model_path does not exist
        if not os.path.exists(self.model_path):
            if model_path is None:
                alternative_path = os.path.join(os.path.dirname(__file__), "..", "models", "local_model.gguf")
                if os.path.exists(alternative_path):
                    self.model_path = alternative_path
                    logger.info("Auto-detected fallback model at '%s'", self.model_path)
                else:
                    logger.warning(
                        "Gemma GGUF model file not found at default '%s' or fallback '%s'. Running in fallback mode.",
                        self.model_path,
                        alternative_path,
                    )
                    return
            else:
                logger.warning(
                    "Requested model file '%s' does not exist. Running in fallback mode.",
                    self.model_path,
                )
                return

        try:
            logger.info("Loading local Gemma model from '%s'...", self.model_path)
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=2048,
                n_threads=4,
                verbose=False,
            )
            logger.info("Local Gemma model loaded successfully.")
        except Exception as exc:
            logger.error("Failed to load local Gemma model: %s", exc)

    def is_available(self) -> bool:
        """Checks if the local Gemma model is loaded and available for inference."""
        return self.llm is not None

    def classify_task(self, prompt: str) -> str:
        """Queries local Gemma to classify the prompt into one of the 8 task types.

        Args:
            prompt: The user task prompt.

        Returns:
            The predicted category name in uppercase.
        """
        if not self.llm:
            raise RuntimeError("Local Gemma model is not loaded.")

        system_prompt = (
            "You are a routing helper. Classify the user prompt into exactly one of these categories:\n"
            "- GENERAL\n"
            "- MATH\n"
            "- SUMMARY\n"
            "- SENTIMENT\n"
            "- NER\n"
            "- CODE_DEBUG\n"
            "- CODE_GENERATION\n"
            "- LOGIC\n\n"
            "Return only the category name in uppercase, with no other words, prefixes, or punctuation."
        )

        formatted_prompt = f"<start_of_turn>user\n{system_prompt}\n\nPrompt: {prompt.strip()}<end_of_turn>\n<start_of_turn>model\n"

        try:
            response = self.llm(
                formatted_prompt,
                max_tokens=10,
                temperature=0.0,
                stop=["<end_of_turn>"],
            )
            return response["choices"][0]["text"].strip().upper()
        except Exception as exc:
            logger.error("Local Gemma classification failed: %s", exc)
            raise

    def generate_reasoning_hints(self, prompt: str) -> str:
        """Queries local Gemma to solve the prompt step by step, generating reasoning hints.

        Args:
            prompt: The user task prompt.

        Returns:
            Reasoning notes to pass to the Fireworks model.
        """
        if not self.llm:
            raise RuntimeError("Local Gemma model is not loaded.")

        system_prompt = (
            "You are a reasoning helper. Solve this task step by step. "
            "Write down your reasoning process and intermediate calculations clearly so another model can use it as a hint. "
            "Keep the response focused and under 150 words."
        )

        formatted_prompt = f"<start_of_turn>user\n{system_prompt}\n\nTask: {prompt.strip()}<end_of_turn>\n<start_of_turn>model\n"

        try:
            response = self.llm(
                formatted_prompt,
                max_tokens=256,
                temperature=0.0,
                stop=["<end_of_turn>"],
            )
            return response["choices"][0]["text"].strip()
        except Exception as exc:
            logger.error("Local Gemma reasoning hints generation failed: %s", exc)
            return ""

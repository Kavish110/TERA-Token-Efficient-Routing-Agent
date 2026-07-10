#!/bin/bash

# Exit on any error
set -e

# Change directory to the repository root
CDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd "$CDIR"

# Load environment variables from .env if it exists, otherwise fallback to .env.example
if [ -f .env ]; then
    echo "Loading environment variables from .env..."
    export $(grep -v '^#' .env | xargs)
elif [ -f .env.example ]; then
    echo "Warning: .env not found. Loading environment variables from .env.example..."
    export $(grep -v '^#' .env.example | xargs)
else
    echo "Error: Neither .env nor .env.example exists."
    exit 1
fi

# Ensure output and models directories exist
mkdir -p output models

# Download the Gemma 4 E4B GGUF model locally if it does not exist
LOCAL_MODEL_PATH="models/local_model.gguf"
if [ ! -f "$LOCAL_MODEL_PATH" ]; then
    echo "Gemma 4 E4B model not found locally at $LOCAL_MODEL_PATH."
    echo "Downloading model (~5.1 GB)... This might take a few minutes."
    curl -L -o "$LOCAL_MODEL_PATH" \
        "https://huggingface.co/bartowski/google_gemma-4-E4B-it-GGUF/resolve/main/google_gemma-4-E4B-it-Q4_K_M.gguf?download=true"
    echo "Model downloaded successfully."
fi

# Set local paths for test runs
export INPUT_FILE=${INPUT_FILE:-input/tasks.json}
export OUTPUT_FILE=${OUTPUT_FILE:-output/results.json}
export LOCAL_GEMMA_PATH="$LOCAL_MODEL_PATH"

# Run pipeline and capture log output to both console and a temp log file
LOG_FILE="output/test_run.log"
echo "Starting TERA pipeline execution..."
echo "------------------------------------------------"
.venv/bin/python main.py 2>&1 | tee "$LOG_FILE"
echo "------------------------------------------------"

# Parse token metrics from the log file
PROMPT_TOKENS=$(grep -a "tera_pipeline:" "$LOG_FILE" | grep -a -oE "Tokens: [0-9]+ prompt" | awk '{sum+=$2} END {print sum}')
COMPLETION_TOKENS=$(grep -a "tera_pipeline:" "$LOG_FILE" | grep -a -oE "[0-9]+ completion" | awk '{sum+=$1} END {print sum}')

PROMPT_TOKENS=${PROMPT_TOKENS:-0}
COMPLETION_TOKENS=${COMPLETION_TOKENS:-0}
TOTAL_TOKENS=$((PROMPT_TOKENS + COMPLETION_TOKENS))

echo "Execution Summary:"
echo "  Prompt Tokens:     $PROMPT_TOKENS"
echo "  Completion Tokens: $COMPLETION_TOKENS"
echo "  Total Tokens:      $TOTAL_TOKENS"

#!/bin/bash
# Download test data for Qwen3-TTS Rust tests
# This script caches downloads to avoid re-downloading large files

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DATA_DIR="${SCRIPT_DIR}/../test_data"

# Create directories
mkdir -p "${TEST_DATA_DIR}/tokenizer"
mkdir -p "${TEST_DATA_DIR}/speech_tokenizer"

echo "=== Downloading Text Tokenizer (7 MB) ==="
if [ -f "${TEST_DATA_DIR}/tokenizer/tokenizer.json" ]; then
    echo "  tokenizer.json already exists, skipping..."
else
    echo "  Downloading tokenizer.json..."
    curl -sL "https://huggingface.co/Qwen/Qwen2-0.5B/resolve/main/tokenizer.json" \
        -o "${TEST_DATA_DIR}/tokenizer/tokenizer.json"
fi

if [ -f "${TEST_DATA_DIR}/tokenizer/config.json" ]; then
    echo "  config.json already exists, skipping..."
else
    echo "  Downloading config.json..."
    curl -sL "https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/resolve/main/config.json" \
        -o "${TEST_DATA_DIR}/tokenizer/config.json"
fi

echo ""
echo "=== Downloading Speech Tokenizer (682 MB) ==="
if [ -f "${TEST_DATA_DIR}/speech_tokenizer/model.safetensors" ]; then
    echo "  model.safetensors already exists, skipping..."
else
    echo "  Downloading model.safetensors (this may take a few minutes)..."
    curl -L "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/model.safetensors" \
        -o "${TEST_DATA_DIR}/speech_tokenizer/model.safetensors"
fi

if [ -f "${TEST_DATA_DIR}/speech_tokenizer/config.json" ]; then
    echo "  config.json already exists, skipping..."
else
    echo "  Downloading config.json..."
    curl -sL "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/config.json" \
        -o "${TEST_DATA_DIR}/speech_tokenizer/config.json"
fi

if [ -f "${TEST_DATA_DIR}/speech_tokenizer/preprocessor_config.json" ]; then
    echo "  preprocessor_config.json already exists, skipping..."
else
    echo "  Downloading preprocessor_config.json..."
    curl -sL "https://huggingface.co/Qwen/Qwen3-TTS-Tokenizer-12Hz/resolve/main/preprocessor_config.json" \
        -o "${TEST_DATA_DIR}/speech_tokenizer/preprocessor_config.json"
fi

echo ""
echo "=== Download Summary ==="
echo "Text tokenizer:"
ls -lh "${TEST_DATA_DIR}/tokenizer/"
echo ""
echo "Speech tokenizer:"
ls -lh "${TEST_DATA_DIR}/speech_tokenizer/"
echo ""
echo "Done! Run 'cargo test' to execute tests with real weights."

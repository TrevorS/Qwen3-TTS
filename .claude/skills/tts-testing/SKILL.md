---
name: tts-testing
description: |
  Run TTS variant tests, serve audio results, and verify with Whisper.
  Triggers on: "test tts", "run variants", "test all models",
  "serve audio", "listen to results", "check tts quality",
  "compare models", "test voice output"
---

# TTS Variant Testing & Verification

End-to-end workflow: build → generate audio across all model variants →
serve with descriptions → verify intelligibility with Whisper.

## Quick Start

### Run all variants (from project root)

```bash
./qwen3-tts-rs/scripts/test-variants.sh --build --device cuda --serve
```

This will:
1. Auto-detect the best build features (flash-attn > cuda > cpu-only)
2. Discover all models in `qwen3-tts-rs/test_data/models/`
3. Run every valid model+mode combination on the specified device(s)
4. Generate `test_data/variant_tests/index.html` with audio players
5. Generate `test_data/variant_tests/results.json` for programmatic use
6. Start an HTTP server for browser listening

### Run inside NGC container (for CUDA + flash-attn)

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace/Qwen3-TTS \
  -p 8765:8765 \
  nvcr.io/nvidia/pytorch:25.12-py3 \
  bash -c 'cd /workspace/Qwen3-TTS && \
    curl --proto "=https" --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
    source ~/.cargo/env && \
    bash qwen3-tts-rs/scripts/test-variants.sh --build --device cuda --serve'
```

## Output Files

All outputs go to `test_data/variant_tests/`:

```
test_data/variant_tests/
├── index.html          # Dashboard with audio players
├── results.json        # Machine-readable results
├── cpu/                # CPU-generated WAVs
│   ├── 0.6b-base-xvector.wav
│   ├── 0.6b-base-icl.wav
│   ├── 0.6b-customvoice-ryan.wav
│   └── ...
└── cuda/               # CUDA-generated WAVs
    ├── 0.6b-base-xvector.wav
    └── ...
```

## Verify with Whisper

After generating variants, check intelligibility with the whisper-test skill.

### Transcribe all CUDA outputs

```bash
uv run --no-project --with openai-whisper --with scipy --python 3.11 \
  python3 ~/.claude/skills/whisper-test/transcribe.py \
  --model large-v3 \
  --expected "Hello world, this is a test." \
  test_data/variant_tests/cuda/*.wav
```

### Transcribe specific variants

```bash
uv run --no-project --with openai-whisper --with scipy --python 3.11 \
  python3 ~/.claude/skills/whisper-test/transcribe.py \
  --model large-v3 \
  --expected "Hello world, this is a test." \
  test_data/variant_tests/cuda/1.7b-base-icl.wav \
  test_data/variant_tests/cuda/1.7b-customvoice-ryan.wav
```

### Interpret Whisper results

- Use `large-v3` for TTS evaluation (smaller models hallucinate on synthesized speech)
- WER < 30% generally indicates intelligible speech
- 0.6B models produce less intelligible output than 1.7B — this is expected
- ICL voice clone may have garbled openings but preserve key phrases

## Script Options

```
--device DEV    Test specific device(s). Repeat for multiple. Default: auto-detect.
--serve         Start HTTP server after tests complete.
--build         Build release binary before testing.
--hostname H    Hostname for HTTP URLs (default: $HOSTNAME).
--port P        HTTP server port (default: 8765).
--text TEXT     Text to synthesize (default: "Hello world, this is a test.").
--seed N        Random seed (default: 42).
--duration S    Duration in seconds (default: 3.0).
```

## Reading results.json

The JSON file is an array of objects:

```json
[
  {
    "label": "1.7b-base-icl",
    "device": "cuda",
    "time": "5.4",
    "status": "PASS",
    "size": "280K",
    "file": "cuda/1.7b-base-icl.wav"
  }
]
```

Use this to programmatically feed WAV paths to Whisper or other analysis tools.

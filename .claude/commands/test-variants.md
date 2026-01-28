# Test All Model Variants

Build the CLI and run all valid model+mode+device combinations to generate test WAV files, then serve them over HTTP.

## Build

Detect the best feature set and build accordingly:

```bash
# With CUDA toolkit + flash-attn support (bf16 + Flash Attention 2):
cargo build --release --features flash-attn,cli --manifest-path qwen3-tts-rs/Cargo.toml

# With CUDA toolkit, no flash-attn:
cargo build --release --features cuda,cli --manifest-path qwen3-tts-rs/Cargo.toml

# CPU only:
cargo build --release --features cli --manifest-path qwen3-tts-rs/Cargo.toml
```

Or use the script's `--build` flag which auto-detects:

```bash
./qwen3-tts-rs/scripts/test-variants.sh --build
```

## Test Matrix

Run each of the following combinations. Use the release binary at `qwen3-tts-rs/target/release/generate_audio`.

Create the output directory first:

```bash
mkdir -p test_data/variant_tests
```

### Devices

Test on all available devices. On a CUDA machine, test both `cpu` and `cuda` to compare.

### Base models (voice cloning with reference audio)

Reference audio: `qwen3-tts-rs/examples/data/apollo11_one_small_step.wav`
ICL transcript: `"That's one small step for man, one giant leap for mankind."`

**1. 0.6B Base, x_vector_only:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/0.6b-base \
  --ref-audio qwen3-tts-rs/examples/data/apollo11_one_small_step.wav \
  --x-vector-only \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/0.6b-base-xvector-apollo11.wav
```

**2. 0.6B Base, ICL:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/0.6b-base \
  --ref-audio qwen3-tts-rs/examples/data/apollo11_one_small_step.wav \
  --ref-text "That's one small step for man, one giant leap for mankind." \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/0.6b-base-icl-apollo11.wav
```

**3. 1.7B Base, x_vector_only:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/1.7b-base \
  --ref-audio qwen3-tts-rs/examples/data/apollo11_one_small_step.wav \
  --x-vector-only \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/1.7b-base-xvector-apollo11.wav
```

**4. 1.7B Base, ICL:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/1.7b-base \
  --ref-audio qwen3-tts-rs/examples/data/apollo11_one_small_step.wav \
  --ref-text "That's one small step for man, one giant leap for mankind." \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/1.7b-base-icl-apollo11.wav
```

### CustomVoice models (preset speakers)

**5. 0.6B CustomVoice, Ryan:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/0.6b-customvoice \
  --speaker ryan \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/0.6b-customvoice-ryan.wav
```

**6. 0.6B CustomVoice, Serena:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/0.6b-customvoice \
  --speaker serena \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/0.6b-customvoice-serena.wav
```

**7. 1.7B CustomVoice, Ryan:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/1.7b-customvoice \
  --speaker ryan \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/1.7b-customvoice-ryan.wav
```

**8. 1.7B CustomVoice, Serena:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/1.7b-customvoice \
  --speaker serena \
  --text "Hello world, this is a test." \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/1.7b-customvoice-serena.wav
```

### VoiceDesign models (text-described voice via `--instruct`)

VoiceDesign uses natural language descriptions to condition the voice (`tts_model_type: voice_design`, empty `spk_id` map). Use `--instruct` with a voice description.

**9. 1.7B VoiceDesign, instruct:**
```bash
./qwen3-tts-rs/target/release/generate_audio \
  --model-dir test_data/models/1.7b-voicedesign \
  --instruct "A cheerful young female voice with clear pronunciation and natural intonation." \
  --text "Hello world, this is a test." \
  --language english \
  --duration 3.0 --seed 42 \
  --output test_data/variant_tests/1.7b-voicedesign-instruct.wav
```

### Adding `--device cuda` for GPU tests

When CUDA is available, repeat each test above with `--device cuda` and write to a `cuda/` subdirectory. The script handles this automatically.

## Run with the script

The script auto-discovers models, detects devices, and runs the full matrix:

```bash
# Auto-detect everything, build first
./qwen3-tts-rs/scripts/test-variants.sh --build

# CPU only
./qwen3-tts-rs/scripts/test-variants.sh --device cpu

# CUDA only
./qwen3-tts-rs/scripts/test-variants.sh --device cuda

# Both devices, then serve results
./qwen3-tts-rs/scripts/test-variants.sh --device cpu --device cuda --serve

# Inside an NGC container (build with flash-attn)
./qwen3-tts-rs/scripts/test-variants.sh --build --device cuda --serve --hostname $(hostname)
```

## Output

The script generates two files in `test_data/variant_tests/`:

- **`index.html`** — Dark-themed dashboard with summary table, speedup ratios, and `<audio>` players for every output. Open in a browser or serve over HTTP.
- **`results.json`** — Machine-readable array of `{label, device, time, status, size, file}` objects for downstream tools (e.g. Whisper transcription).

## Serve results

Use `--serve` to start an HTTP server after tests complete:

```bash
./qwen3-tts-rs/scripts/test-variants.sh --build --serve --hostname $(hostname)
```

Then open `http://<hostname>:8765/` in a browser to listen to all variants with descriptions.

To serve manually:

```bash
python3 -m http.server 8765 -d test_data/variant_tests/
```

## Whisper quality check

After generating variants, transcribe with Whisper to verify intelligibility:

```bash
uv run --no-project --with openai-whisper --with scipy --python 3.11 \
  python3 ~/.claude/skills/whisper-test/transcribe.py \
  --model large-v3 \
  --expected "Hello world, this is a test." \
  test_data/variant_tests/cuda/*.wav
```

## Summary

After all runs, the script prints a summary table showing:
- Test name
- Status (pass/fail/skipped)
- Per-device timing (with speedup ratio when multiple devices are tested)
- Output file size

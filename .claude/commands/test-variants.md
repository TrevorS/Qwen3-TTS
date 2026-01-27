# Test All Model Variants

Build the CLI and run all 8 valid model+mode combinations to generate test WAV files, then serve them over HTTP.

## Build

```bash
cargo build --release --features cli --manifest-path qwen3-tts-rs/Cargo.toml
```

## Test Matrix

Run each of the following 8 commands. Use the release binary at `qwen3-tts-rs/target/release/generate_audio`.

Create the output directory first:

```bash
mkdir -p test_data/variant_tests
```

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

## Run all 8 tests

Execute each command above sequentially. If a model directory doesn't exist, skip that test and note it in the summary.

## Serve results

After all tests complete, start an HTTP server:

```bash
python3 -m http.server 8765 -d test_data/variant_tests/
```

Print the URLs for each generated file using `spark-ebf0` as the hostname:

```
http://spark-ebf0:8765/0.6b-base-xvector-apollo11.wav
http://spark-ebf0:8765/0.6b-base-icl-apollo11.wav
http://spark-ebf0:8765/1.7b-base-xvector-apollo11.wav
http://spark-ebf0:8765/1.7b-base-icl-apollo11.wav
http://spark-ebf0:8765/0.6b-customvoice-ryan.wav
http://spark-ebf0:8765/0.6b-customvoice-serena.wav
http://spark-ebf0:8765/1.7b-customvoice-ryan.wav
http://spark-ebf0:8765/1.7b-customvoice-serena.wav
```

## Summary

After all runs, print a summary table showing:
- Test name
- Status (pass/fail/skipped)
- Output file size
- Duration of generated audio (if available from CLI output)

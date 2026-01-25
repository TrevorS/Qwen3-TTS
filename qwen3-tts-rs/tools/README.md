# Python Tools for Qwen3-TTS Development

Tools for generating reference data, comparing outputs, and validating the Rust implementation.

## Setup

```bash
cd tools
uv sync  # Install dependencies
```

## Core Tools

### Reference Export (for validation tests)
- `export_reference_values.py` - Export talker and code predictor reference values
- `export_decoder_reference.py` - Export speech tokenizer decoder reference values
- `export_e2e_reference.py` - Export end-to-end pipeline reference values

### Audio Generation
- `generate_reference_audio.py` - Generate reference audio with Python for comparison
- `generate_correct_flow.py` - Generate audio with correct trailing text flow
- `generate_correct_greedy.py` - Generate audio with greedy decoding (deterministic)
- `generate_customvoice_tts.py` - CustomVoice TTS generation

### Comparison & Analysis
- `compare_audio.py` - Compare Rust vs Python audio outputs
- `analyze_audio.py` - Analyze audio file statistics

## Usage Examples

### Generate Python Reference Audio
```bash
uv run python generate_reference_audio.py \
  --text "Hello" \
  --seed 42 \
  --frames 25 \
  --model-dir ../test_data/model \
  --output-dir ../test_data/reference_audio
```

### Compare Rust and Python Outputs
```bash
uv run python compare_audio.py \
  --rust ../test_data/rust_audio \
  --python ../test_data/reference_audio
```

### Export Reference Values for Validation Tests
```bash
uv run python export_reference_values.py
uv run python export_decoder_reference.py
```

## Archived Debug Scripts

Debug scripts used during development are in `archive/`. These were used for
investigating specific issues and are kept for reference.

# qwen3-tts

Pure Rust inference for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), a high-quality text-to-speech model from Alibaba.

## Features

- **CPU inference** with optional MKL/Accelerate for faster BLAS operations
- **CUDA** support for NVIDIA GPU acceleration
- **Metal** support for Apple Silicon
- **Streaming synthesis** for low-latency audio output
- **Voice cloning** from reference audio (Base models)
- **Preset speakers** with 9 built-in voices (CustomVoice models)
- **Auto-detection** of model variant from `config.json`
- **HuggingFace Hub integration** for easy model downloads

## Model Variants

Five official model variants exist across two size classes. Each variant supports a different speaker conditioning method:

| Variant | Params | Speaker Conditioning | Use Case |
|---------|--------|---------------------|----------|
| **0.6B Base** | 1.8 GB | Voice cloning from reference audio | Clone any voice from a WAV file |
| **0.6B CustomVoice** | 1.8 GB | 9 preset speakers | Pick from built-in voices |
| **1.7B Base** | 3.9 GB | Voice cloning from reference audio | Higher quality voice cloning |
| **1.7B CustomVoice** | 3.9 GB | 9 preset speakers | Higher quality preset voices |
| **1.7B VoiceDesign** | 3.8 GB | Text description | Describe a voice in natural language |

### Which model should I use?

- **Want to clone a specific voice?** Use a **Base** model with `--ref-audio`.
- **Want a quick preset voice?** Use a **CustomVoice** model with `--speaker`.
- **Want to describe a voice in text?** Use **1.7B VoiceDesign** (not yet implemented in CLI).
- **Unsure?** Start with **0.6B CustomVoice** for the fastest results.

### Valid combinations

| | Preset speakers | Voice clone (x_vector) | Voice clone (ICL) | Text-described voice |
|---|:-:|:-:|:-:|:-:|
| **Base** | | x | x | |
| **CustomVoice** | x | | | |
| **VoiceDesign** | | | | x |

Using the wrong combination (e.g. preset speakers on a Base model) won't crash, but produces unpredictable voice output. The library and CLI warn when this happens.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
qwen3-tts = { version = "0.1", features = ["hub"] }
```

### Feature Flags

| Feature | Description |
|---------|-------------|
| `cpu` (default) | CPU inference |
| `cuda` | NVIDIA GPU acceleration |
| `metal` | Apple Silicon GPU acceleration |
| `mkl` | Intel MKL for faster CPU inference |
| `accelerate` | Apple Accelerate framework |
| `hub` | HuggingFace Hub model downloads |
| `cli` | Command-line tools |

## Quick Start

### Preset speakers (CustomVoice)

```rust
use qwen3_tts::{Qwen3TTS, Speaker, Language, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained("path/to/customvoice_model", device)?;

    let audio = model.synthesize_with_voice(
        "Hello, world!",
        Speaker::Ryan,
        Language::English,
        None,
    )?;
    audio.save("output.wav")?;
    Ok(())
}
```

Available speakers: `Serena`, `Vivian`, `UncleFu`, `Ryan`, `Aiden`, `OnoAnna`, `Sohee`, `Eric`, `Dylan`

### Voice cloning (Base)

```rust
use qwen3_tts::{Qwen3TTS, Language, AudioBuffer, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained("path/to/base_model", device)?;

    // Load reference audio
    let ref_audio = AudioBuffer::load("reference_voice.wav")?;

    // x_vector_only: speaker embedding from reference audio
    let prompt = model.create_voice_clone_prompt(&ref_audio, None)?;

    // ICL mode: also provide the transcript of the reference audio
    // let prompt = model.create_voice_clone_prompt(&ref_audio, Some("transcript of ref audio"))?;

    let audio = model.synthesize_voice_clone(
        "Hello in the cloned voice!",
        &prompt,
        Language::English,
        None,
    )?;
    audio.save("cloned.wav")?;
    Ok(())
}
```

### With custom options

```rust
use qwen3_tts::{Qwen3TTS, SynthesisOptions, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained("path/to/model", device)?;

    let options = SynthesisOptions {
        temperature: 0.8,
        top_k: 30,
        top_p: 0.85,
        repetition_penalty: 1.05,
        ..Default::default()
    };
    let audio = model.synthesize("Custom settings!", Some(options))?;
    audio.save("output.wav")?;
    Ok(())
}
```

### Streaming synthesis

For low-latency applications, stream audio in chunks:

```rust
use qwen3_tts::{Qwen3TTS, Speaker, Language, SynthesisOptions, auto_device};

fn main() -> anyhow::Result<()> {
    let device = auto_device()?;
    let model = Qwen3TTS::from_pretrained("path/to/model", device)?;

    let options = SynthesisOptions {
        chunk_frames: 10, // ~800ms per chunk
        ..Default::default()
    };

    for chunk in model.synthesize_streaming(
        "Hello, world!",
        Speaker::Ryan,
        Language::English,
        options,
    )? {
        let audio = chunk?;
        // Play or stream this chunk
        println!("Got {} samples", audio.samples.len());
    }
    Ok(())
}
```

### With HuggingFace Hub

```rust
use qwen3_tts::{Qwen3TTS, ModelPaths, auto_device};

fn main() -> anyhow::Result<()> {
    let paths = ModelPaths::download(None)?;
    let device = auto_device()?;

    let model = Qwen3TTS::from_paths(&paths, device)?;
    let audio = model.synthesize("Hello from HuggingFace!", None)?;
    audio.save("output.wav")?;
    Ok(())
}
```

## Architecture

The TTS pipeline consists of three stages:

1. **TalkerModel**: 28-layer transformer generating semantic tokens from text autoregressively. Uses MRoPE (multimodal rotary position encoding) across all variants.

2. **CodePredictor**: 5-layer decoder that generates 15 acoustic tokens per semantic token. Always 1024 hidden dim; 1.7B models use a projection layer to bridge from the talker's 2048-dim space.

3. **Decoder12Hz**: Converts 16-codebook tokens to 24kHz audio via ConvNeXt blocks and transposed convolution upsampling. Shared across all model variants.

```
Text --> TalkerModel --> Semantic Token --> CodePredictor --> [16 codes] --> Decoder --> Audio
              ^                                  ^
         (autoregressive,                  (per frame,
          one per frame)                    15 acoustic codes)
```

## CLI

The model variant is auto-detected from `config.json`. The CLI warns if your flags don't match the model type.

```bash
# CustomVoice: preset speaker
cargo run --release --features cli --bin generate_audio -- \
  --model-dir path/to/customvoice \
  --text "Hello world" \
  --speaker ryan \
  --language english \
  --duration 3.0

# Base: voice cloning (x_vector_only)
cargo run --release --features cli --bin generate_audio -- \
  --model-dir path/to/base \
  --text "Hello world" \
  --ref-audio reference.wav

# Base: voice cloning (ICL — higher quality, needs reference transcript)
cargo run --release --features cli --bin generate_audio -- \
  --model-dir path/to/base \
  --text "Hello world" \
  --ref-audio reference.wav \
  --ref-text "transcript of the reference audio"

# Reproducible generation with fixed seed
cargo run --release --features cli --bin generate_audio -- \
  --model-dir path/to/model \
  --text "Hello" \
  --seed 42 \
  --frames 25
```

### CLI options

| Flag | Default | Description |
|------|---------|-------------|
| `--model-dir` | `test_data/model` | Path to model directory |
| `--text` | `"Hello"` | Text to synthesize |
| `--speaker` | `ryan` | Preset speaker (CustomVoice only) |
| `--language` | `english` | Target language |
| `--ref-audio` | | Reference audio WAV for voice cloning (Base only) |
| `--ref-text` | | Reference transcript for ICL mode |
| `--duration` | | Duration in seconds |
| `--frames` | `25` | Number of frames (if no duration) |
| `--temperature` | `0.7` | Sampling temperature |
| `--top-k` | `50` | Top-k sampling |
| `--top-p` | `0.9` | Nucleus sampling threshold |
| `--repetition-penalty` | `1.05` | Repetition penalty |
| `--seed` | `42` | Random seed for reproducibility |

## Model Files

All models share the same speech tokenizer and text tokenizer.

| Component | HuggingFace Repo | Size |
|-----------|------------------|------|
| 0.6B Base | `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 1.8 GB |
| 0.6B CustomVoice | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 1.8 GB |
| 1.7B Base | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 3.9 GB |
| 1.7B CustomVoice | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | 3.9 GB |
| 1.7B VoiceDesign | `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 3.8 GB |
| Speech Tokenizer | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | 682 MB |
| Text Tokenizer | `Qwen/Qwen2-0.5B` | 7 MB |

### Supported languages

English, Chinese, Japanese, Korean, German, French, Russian, Portuguese, Spanish, Italian

## Sample Rate

Output audio is always 24kHz mono. Use `audio::resample()` for other rates:

```rust
use qwen3_tts::audio;

let audio_48k = audio::resample(&audio, 48000)?;
```

## License

MIT License. See the main Qwen3-TTS repository for model license information.

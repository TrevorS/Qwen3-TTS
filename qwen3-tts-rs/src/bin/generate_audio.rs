//! CLI tool for generating audio with deterministic seed
//!
//! This tool generates WAV audio files using the Qwen3-TTS pipeline with a
//! specific seed, allowing direct comparison with Python output.
//!
//! Usage:
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --frames 25
//!     cargo run --features cli --bin generate_audio -- --text "Hello" --seed 42 --duration 2.0

use anyhow::Result;
use byteorder::{LittleEndian, WriteBytesExt};
use candle_core::{DType, Device, IndexOp, Tensor};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use qwen3_tts::{generation, models, AudioBuffer};

/// Generate reference audio with deterministic seed for comparison
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long, default_value = "Hello")]
    text: String,

    /// Random seed for reproducible generation
    #[arg(short, long, default_value_t = 42)]
    seed: u64,

    /// Number of frames to generate (overridden by duration if specified)
    #[arg(short, long, default_value_t = 25)]
    frames: usize,

    /// Duration in seconds (overrides frames if specified)
    #[arg(short, long)]
    duration: Option<f64>,

    /// Sampling temperature
    #[arg(long, default_value_t = 0.7)]
    temperature: f64,

    /// Top-k sampling parameter
    #[arg(long, default_value_t = 50)]
    top_k: usize,

    /// Top-p (nucleus) sampling parameter
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,

    /// Model directory containing model.safetensors
    #[arg(short, long, default_value = "test_data/model")]
    model_dir: String,

    /// Output directory for generated files
    #[arg(short, long, default_value = "test_data/rust_audio")]
    output_dir: String,

    /// Compare with Python reference output (if exists)
    #[arg(short, long)]
    compare: bool,

    /// Python reference directory
    #[arg(long, default_value = "test_data/reference_audio")]
    reference_dir: String,
}

/// Metadata for generated audio
#[derive(Debug, Serialize, Deserialize)]
struct GenerationMetadata {
    text: String,
    seed: u64,
    num_frames: usize,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    input_ids: Vec<u32>,
    codes_shape: Vec<usize>,
    audio_samples: usize,
    sample_rate: u32,
}

/// Text to token mapping (simplified - matches Python script)
fn text_to_ids(text: &str) -> Vec<u32> {
    match text {
        "Hello" => vec![9707],
        "Hello world" => vec![9707, 1879],
        "Hello, this is a" => vec![9707, 11, 419, 374, 264],
        "Hello, this is a test" => vec![9707, 11, 419, 374, 264, 1273],
        "The quick brown fox jumps over the lazy dog. This is a test of the text to speech system." => {
            vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 13, 1096, 374, 264, 1273, 315, 279, 1467, 311, 8806, 1849, 13]
        }
        "The quick brown fox jumps over the lazy dog near the riverbank while the sun sets behind the distant mountains casting golden light across the peaceful valley below" => {
            vec![785, 3974, 13876, 38835, 34208, 916, 279, 15678, 5562, 3143, 279, 14796, 17033, 1393, 279, 7015, 7289, 4815, 279, 28727, 23501, 24172, 20748, 3100, 3941, 279, 25650, 33581, 3685]
        }
        _ => {
            eprintln!("Warning: Text '{}' not in tokenizer mapping, using 'Hello'", text);
            vec![9707]
        }
    }
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    let args = Args::parse();

    // Calculate frames from duration if specified
    let num_frames = if let Some(duration) = args.duration {
        (duration * 12.5) as usize
    } else {
        args.frames
    };

    println!("=== Generating Audio (Rust) ===");
    println!("Text: {}", args.text);
    println!("Seed: {}", args.seed);
    println!("Frames: {}", num_frames);
    println!("Temperature: {}", args.temperature);
    println!("Top-k: {}", args.top_k);
    println!("Top-p: {}", args.top_p);

    // Set seed
    generation::set_seed(args.seed);
    println!("\nSeed set: {} (is_seeded: {})", args.seed, generation::is_seeded());

    // Reset RNG to ensure deterministic starting point
    generation::reset_rng();

    let device = Device::Cpu;

    // Create output directory
    let output_dir = Path::new(&args.output_dir);
    fs::create_dir_all(output_dir)?;

    // Load weights
    println!("\nLoading model weights...");
    let model_path = Path::new(&args.model_dir).join("model.safetensors");
    let weights = load_weights(&model_path, &device)?;

    let decoder_path = Path::new(&args.model_dir).join("speech_tokenizer/model.safetensors");
    let decoder_weights = load_weights(&decoder_path, &device)?;

    // Get input IDs
    let input_ids = text_to_ids(&args.text);
    println!("Input IDs: {:?}", input_ids);
    let input_tensor = Tensor::new(input_ids.as_slice(), &device)?.unsqueeze(0)?;

    // Create models
    println!("Creating TalkerModel...");
    let talker = models::TalkerModel::from_weights(&weights, &device)?;

    println!("Creating CodePredictor...");
    let cp_config = models::CodePredictorConfig::default();
    let cp_weights = filter_weights(&weights, "talker.code_predictor.");
    let cp_vb = candle_nn::VarBuilder::from_tensors(cp_weights, DType::F32, &device);
    let code_predictor = models::CodePredictor::new(cp_config, cp_vb)?;

    println!("Creating Decoder12Hz...");
    let decoder = models::codec::Decoder12Hz::from_weights(&decoder_weights, Default::default())?;

    // Generate codes
    println!("\nGenerating {} frames...", num_frames);
    let progress = ProgressBar::new(num_frames as u64);
    progress.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} frames")?
            .progress_chars("#>-"),
    );

    let gen_config = generation::GenerationConfig {
        max_new_tokens: num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        repetition_penalty: 1.0,
        eos_token_id: None, // Don't stop early for comparison
    };

    // Initialize KV caches
    let mut kv_caches = talker.new_kv_caches();

    // Prefill with text
    let (hidden, logits) = talker.prefill(&input_tensor, &mut kv_caches)?;
    let mut offset = input_tensor.dim(1)?;

    // Get last hidden state
    let seq_len = hidden.dim(1)?;
    let mut last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;

    // Sample first semantic token
    let first_token = generation::sample(&logits.squeeze(1)?, &gen_config)?;
    let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];
    println!("First semantic token: {}", first_token_id);

    // Collect all codes
    let mut all_codes: Vec<Vec<u32>> = Vec::new();

    // First frame
    let semantic_embed = talker.get_codec_embedding(first_token_id)?;
    let acoustic_codes = code_predictor.generate_acoustic_codes(&last_hidden, &semantic_embed)?;

    println!("Frame 0: semantic={}, acoustics={:?}...", first_token_id, &acoustic_codes[..3.min(acoustic_codes.len())]);

    let mut frame_codes = vec![first_token_id];
    frame_codes.extend(acoustic_codes);
    all_codes.push(frame_codes);
    progress.inc(1);

    // Generate remaining frames
    for frame_idx in 1..num_frames {
        let prev_token = all_codes.last().unwrap()[0];
        let (hidden, logits) = talker.generate_step(prev_token, &mut kv_caches, offset)?;
        offset += 1;
        last_hidden = hidden;

        // Sample semantic token
        let next_token = generation::sample(&logits.squeeze(1)?, &gen_config)?;
        let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];

        // Generate acoustic tokens
        let semantic_embed = talker.get_codec_embedding(next_token_id)?;
        let acoustic_codes = code_predictor.generate_acoustic_codes(&last_hidden, &semantic_embed)?;

        if frame_idx < 5 || frame_idx == num_frames - 1 {
            println!("Frame {}: semantic={}, acoustics={:?}...", frame_idx, next_token_id, &acoustic_codes[..3.min(acoustic_codes.len())]);
        } else if frame_idx == 5 {
            println!("...");
        }

        let mut frame_codes = vec![next_token_id];
        frame_codes.extend(acoustic_codes);
        all_codes.push(frame_codes);
        progress.inc(1);
    }

    progress.finish_with_message("Done generating codes");

    // Convert to tensor [1, 16, num_frames]
    let codes_tensor = codes_to_tensor(&all_codes, &device)?;
    println!("\nCodes tensor shape: {:?}", codes_tensor.shape());

    // Save codes as binary
    let codes_bin_path = output_dir.join(format!("codes_seed{}_frames{}.bin", args.seed, num_frames));
    save_codes_binary(&all_codes, &codes_bin_path)?;
    println!("Saved binary codes to: {:?}", codes_bin_path);

    // Decode to audio
    println!("\nDecoding to audio...");
    let waveform = decoder.decode(&codes_tensor)?;
    let audio_samples: Vec<f32> = waveform.flatten_all()?.to_vec1()?;
    println!("Audio samples: {} ({:.3}s at 24kHz)", audio_samples.len(), audio_samples.len() as f64 / 24000.0);

    // Save audio as WAV
    let wav_path = output_dir.join(format!("audio_seed{}_frames{}.wav", args.seed, num_frames));
    let audio_buffer = AudioBuffer::new(audio_samples.clone(), 24000);
    audio_buffer.save(&wav_path)?;
    println!("Saved WAV to: {:?}", wav_path);

    // Save audio as binary
    let audio_bin_path = output_dir.join(format!("audio_seed{}_frames{}.bin", args.seed, num_frames));
    save_audio_binary(&audio_samples, &audio_bin_path)?;
    println!("Saved binary audio to: {:?}", audio_bin_path);

    // Save metadata
    let metadata = GenerationMetadata {
        text: args.text.clone(),
        seed: args.seed,
        num_frames,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        input_ids,
        codes_shape: vec![1, 16, num_frames],
        audio_samples: audio_samples.len(),
        sample_rate: 24000,
    };
    let metadata_path = output_dir.join(format!("metadata_seed{}_frames{}.json", args.seed, num_frames));
    let metadata_file = File::create(&metadata_path)?;
    serde_json::to_writer_pretty(metadata_file, &metadata)?;
    println!("Saved metadata to: {:?}", metadata_path);

    // Compare with Python reference if requested
    if args.compare {
        println!("\n=== Comparing with Python Reference ===");
        compare_with_reference(&args.reference_dir, args.seed, num_frames, &all_codes, &audio_samples)?;
    }

    // Clear seed
    generation::clear_seed();
    println!("\nGeneration complete!");

    Ok(())
}

/// Load weights from safetensors file
fn load_weights(path: &Path, device: &Device) -> Result<HashMap<String, Tensor>> {
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;
    let tensors: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let converted = if tensor.dtype() == DType::BF16 {
                tensor.to_dtype(DType::F32).unwrap()
            } else {
                tensor
            };
            (name, converted)
        })
        .collect();
    Ok(tensors)
}

/// Filter weights by prefix
fn filter_weights(weights: &HashMap<String, Tensor>, prefix: &str) -> HashMap<String, Tensor> {
    weights
        .iter()
        .filter_map(|(k, v)| {
            if k.starts_with(prefix) {
                Some((k.strip_prefix(prefix).unwrap().to_string(), v.clone()))
            } else {
                None
            }
        })
        .collect()
}

/// Convert list of frame codes to tensor [batch, 16, num_frames]
fn codes_to_tensor(codes: &[Vec<u32>], device: &Device) -> Result<Tensor> {
    let num_frames = codes.len();
    if num_frames == 0 {
        return Ok(Tensor::zeros((1, 16, 0), DType::I64, device)?);
    }

    let mut data = vec![0i64; 16 * num_frames];
    for (frame, frame_codes) in codes.iter().enumerate() {
        for (q, &code) in frame_codes.iter().enumerate() {
            data[q * num_frames + frame] = code as i64;
        }
    }

    Ok(Tensor::from_vec(data, (1, 16, num_frames), device)?)
}

/// Save codes as binary (row-major: frame0_q0, frame0_q1, ..., frame1_q0, ...)
fn save_codes_binary(codes: &[Vec<u32>], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    // Write as i64 to match Python format
    for frame_codes in codes {
        for &code in frame_codes {
            writer.write_i64::<LittleEndian>(code as i64)?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Save audio samples as binary f32
fn save_audio_binary(samples: &[f32], path: &Path) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    for &sample in samples {
        writer.write_f32::<LittleEndian>(sample)?;
    }
    writer.flush()?;
    Ok(())
}

/// Compare Rust output with Python reference
fn compare_with_reference(
    reference_dir: &str,
    seed: u64,
    num_frames: usize,
    rust_codes: &[Vec<u32>],
    rust_audio: &[f32],
) -> Result<()> {
    let ref_dir = Path::new(reference_dir);

    // Load Python codes
    let py_codes_path = ref_dir.join(format!("codes_seed{}_frames{}.bin", seed, num_frames));
    if py_codes_path.exists() {
        let py_codes_data = fs::read(&py_codes_path)?;
        let py_codes: Vec<i64> = py_codes_data
            .chunks(8)
            .map(|chunk| i64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Flatten Rust codes for comparison
        let rust_codes_flat: Vec<i64> = rust_codes
            .iter()
            .flat_map(|frame| frame.iter().map(|&c| c as i64))
            .collect();

        // Compare codes
        let codes_match = py_codes.len() == rust_codes_flat.len()
            && py_codes.iter().zip(rust_codes_flat.iter()).all(|(a, b)| a == b);

        if codes_match {
            println!("Codes: MATCH (all {} values identical)", py_codes.len());
        } else {
            println!("Codes: MISMATCH");
            println!("  Python: {} values", py_codes.len());
            println!("  Rust:   {} values", rust_codes_flat.len());

            // Show first differences
            let min_len = py_codes.len().min(rust_codes_flat.len());
            let mut diff_count = 0;
            for i in 0..min_len {
                if py_codes[i] != rust_codes_flat[i] {
                    if diff_count < 5 {
                        println!("  Index {}: Python={}, Rust={}", i, py_codes[i], rust_codes_flat[i]);
                    }
                    diff_count += 1;
                }
            }
            if diff_count > 5 {
                println!("  ... and {} more differences", diff_count - 5);
            }
            println!("  Total differences: {}", diff_count);
        }
    } else {
        println!("Codes: Python reference not found at {:?}", py_codes_path);
    }

    // Load Python audio
    let py_audio_path = ref_dir.join(format!("audio_seed{}_frames{}.bin", seed, num_frames));
    if py_audio_path.exists() {
        let py_audio_data = fs::read(&py_audio_path)?;
        let py_audio: Vec<f32> = py_audio_data
            .chunks(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        // Calculate audio difference statistics
        let min_len = py_audio.len().min(rust_audio.len());
        if min_len > 0 {
            let mut max_diff = 0.0f32;
            let mut sum_diff = 0.0f64;
            let mut sum_sq_diff = 0.0f64;

            for i in 0..min_len {
                let diff = (py_audio[i] - rust_audio[i]).abs();
                max_diff = max_diff.max(diff);
                sum_diff += diff as f64;
                sum_sq_diff += (diff * diff) as f64;
            }

            let mean_diff = sum_diff / min_len as f64;
            let rmse = (sum_sq_diff / min_len as f64).sqrt();

            println!("\nAudio comparison ({} samples):", min_len);
            println!("  Python samples: {}", py_audio.len());
            println!("  Rust samples:   {}", rust_audio.len());
            println!("  Max difference: {:.6}", max_diff);
            println!("  Mean difference: {:.6}", mean_diff);
            println!("  RMSE: {:.6}", rmse);

            // Check if audio is essentially identical
            if max_diff < 1e-5 {
                println!("  Status: MATCH (max diff < 1e-5)");
            } else if max_diff < 1e-3 {
                println!("  Status: CLOSE (max diff < 1e-3)");
            } else {
                println!("  Status: DIFFERENT");
            }
        }
    } else {
        println!("Audio: Python reference not found at {:?}", py_audio_path);
    }

    // Load and compare metadata
    let py_meta_path = ref_dir.join(format!("metadata_seed{}_frames{}.json", seed, num_frames));
    if py_meta_path.exists() {
        let py_meta: serde_json::Value = serde_json::from_reader(File::open(&py_meta_path)?)?;
        println!("\nPython metadata: {:?}", py_meta);
    }

    Ok(())
}

# GPU/CUDA Implementation Guide for Qwen3-TTS Rust

> **Status**: Research & Planning Document
> **Last Updated**: 2026-01-27
> **Target**: Full CUDA support with optional FlashAttention optimization

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Current State Analysis](#current-state-analysis)
3. [Research Findings](#research-findings)
4. [Implementation Plan](#implementation-plan)
5. [Phase 1: Basic CUDA Support](#phase-1-basic-cuda-support)
6. [Phase 2: FlashAttention Integration](#phase-2-flashattention-integration)
7. [Phase 3: Advanced Optimizations](#phase-3-advanced-optimizations)
8. [Testing Strategy](#testing-strategy)
9. [Build & Deployment](#build--deployment)
10. [Performance Benchmarks](#performance-benchmarks)
11. [Troubleshooting](#troubleshooting)

---

## Executive Summary

### Goal
Enable GPU acceleration for Qwen3-TTS Rust inference, achieving performance parity with the Python implementation while maintaining the benefits of Rust (no Python runtime, smaller footprint, easier deployment).

### Current Status
- **Foundation**: CUDA support infrastructure exists via Candle feature flags
- **Blocker**: CLI binary hardcoded to CPU; device not propagated consistently
- **Opportunity**: FlashAttention can provide 2-4x speedup for attention layers

### Expected Outcomes
| Metric | CPU (current) | CUDA (target) | CUDA + FlashAttn |
|--------|---------------|---------------|------------------|
| RTF (1.7B) | ~5-10x | ~1.0-1.5x | ~0.5-0.8x |
| RTF (0.6B) | ~3-5x | ~0.5-0.8x | ~0.3-0.5x |
| Memory | System RAM | 6-8GB VRAM | 4-6GB VRAM |

*RTF = Real-Time Factor (lower is better, <1.0 = faster than real-time)*

---

## Current State Analysis

### Existing GPU Infrastructure

#### Cargo.toml Feature Flags (Already Implemented)
```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
mkl = ["candle-core/mkl", "candle-nn/mkl", "candle-transformers/mkl"]
```

#### Auto-Device Detection (Already Implemented)
**Location**: `src/lib.rs:1523-1544`
```rust
pub fn auto_device() -> Result<Device> {
    #[cfg(feature = "cuda")]
    {
        if let Ok(device) = Device::cuda_if_available(0) {
            if device.is_cuda() {
                tracing::info!("Using CUDA device");
                return Ok(device);
            }
        }
    }

    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            tracing::info!("Using Metal device");
            return Ok(device);
        }
    }

    tracing::info!("Using CPU device");
    Ok(Device::Cpu)
}
```

### Issues Identified

#### Issue 1: CLI Hardcoded to CPU
**Location**: `src/bin/generate_audio.rs:183`
```rust
let device = Device::Cpu;  // ← Should use auto_device()
```

#### Issue 2: Test Code Uses Hardcoded CPU
Multiple test files in `src/models/` and `src/generation/` use:
```rust
let device = Device::Cpu;  // In ~70 locations
```

#### Issue 3: No Device CLI Argument
The CLI doesn't allow users to specify device preference.

#### Issue 4: No BF16 Runtime Support
Weights are converted BF16→F32 at load time. CUDA would benefit from BF16.

### Files Requiring Modification

| File | Issue | Priority |
|------|-------|----------|
| `src/bin/generate_audio.rs` | Hardcoded CPU | HIGH |
| `src/lib.rs` | Add device parsing utility | HIGH |
| `src/models/transformer.rs` | FlashAttention integration point | MEDIUM |
| `src/models/talker.rs` | Verify device propagation | MEDIUM |
| `src/models/codec/decoder.rs` | Verify device propagation | MEDIUM |
| `Cargo.toml` | Add flash-attn feature | MEDIUM |
| `README.md` | Document CUDA build | LOW |

---

## Research Findings

### Python Implementation Patterns

#### Device Selection (from `qwen_tts/inference/qwen3_tts_model.py`)
```python
# Device passed at load time
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Device inference from model parameters
self.device = getattr(model, "device", None)
if self.device is None:
    self.device = next(model.parameters()).device
```

**Key Pattern**: Device flows through model, not stored globally.

#### Tensor Device Inheritance
```python
# Always inherit device from existing tensors
mask = torch.arange(max_len, device=length.device, ...)
cache_position = torch.arange(..., device=inputs_embeds.device)
```

#### MPS (Metal) Special Handling
```python
# Force float32 for RoPE on MPS to avoid precision issues
device_type = x.device.type if x.device.type != "mps" else "cpu"
with torch.autocast(device_type=device_type, enabled=False):
    # RoPE computation in float32
```

#### CUDA Synchronization for Timing
```python
torch.cuda.synchronize()
t0 = time.time()
# ... inference ...
torch.cuda.synchronize()
t1 = time.time()
```

### FlashAttention with Candle

#### API Overview
```rust
// Core function signature
fn flash_attn(
    q: &Tensor,           // (batch, seq_len, num_heads, head_dim)
    k: &Tensor,           // (batch, seq_len, num_kv_heads, head_dim)
    v: &Tensor,           // (batch, seq_len, num_kv_heads, head_dim)
    softmax_scale: f32,   // 1.0 / sqrt(head_dim)
    causal: bool,         // true for autoregressive
) -> Result<Tensor>

// Other variants
fn flash_attn_windowed(...)      // Sliding window attention
fn flash_attn_alibi(...)         // With ALiBi positional encoding
fn flash_attn_varlen(...)        // Variable-length sequences
```

#### Requirements
| Requirement | Constraint |
|-------------|------------|
| Data types | F16 or BF16 only (no F32) |
| Head dimension | Multiples of 8, max 256 |
| GPU compute | sm_80+ (Ampere: A100, RTX 30xx, 40xx) |
| Tensor layout | `(batch, seq_len, heads, head_dim)` |

#### Crate Options
| Crate | GPU Support | Notes |
|-------|-------------|-------|
| `candle-flash-attn` | Ampere+ | Standard choice |
| `candle-flash-attn-v1` | Turing (T4, RTX 20xx) | Older GPU fallback |
| `candle-flash-attn-v3` | Hopper (H100) | Maximum performance |

### MLX (Apple Silicon) Patterns

#### Key Insight: Unified Memory
- No explicit CPU↔GPU transfers needed
- Arrays live in shared memory
- Operations automatically optimize device usage

#### Precision Requirements
- VoiceDesign models: F16 works
- Voice Cloning models: Requires F32 (F16 causes NaN errors)

---

## Implementation Plan

### Phase Overview

```
┌─────────────────────────────────────────────────────────────────┐
│ Phase 1: Basic CUDA Support                                      │
│ - Fix CLI device selection                                       │
│ - Add --device argument                                          │
│ - Verify tensor operations on GPU                                │
│ - Est: 1-2 hours                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase 2: FlashAttention Integration                              │
│ - Add candle-flash-attn dependency                               │
│ - Implement conditional attention path                           │
│ - Add BF16 support                                               │
│ - Est: 3-4 hours                                                 │
├─────────────────────────────────────────────────────────────────┤
│ Phase 3: Advanced Optimizations                                  │
│ - KV-cache memory optimization                                   │
│ - Batch inference support                                        │
│ - CUDA graph compilation                                         │
│ - Est: 4-6 hours                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Basic CUDA Support

### 1.1 Add Device Parsing Utility

**File**: `src/lib.rs`

**Add after `auto_device()` function (around line 1544)**:

```rust
/// Parse a device string into a Device.
///
/// Supported formats:
/// - "auto": Use auto_device() to select best available
/// - "cpu": Force CPU
/// - "cuda" or "cuda:0": Use CUDA device 0
/// - "cuda:N": Use CUDA device N
/// - "metal": Use Metal (Apple Silicon)
///
/// # Example
///
/// ```rust,ignore
/// let device = qwen3_tts::parse_device("cuda:0")?;
/// ```
pub fn parse_device(device_str: &str) -> Result<Device> {
    match device_str.to_lowercase().as_str() {
        "auto" => auto_device(),
        "cpu" => Ok(Device::Cpu),
        s if s.starts_with("cuda") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = if s == "cuda" {
                    0
                } else if let Some(idx) = s.strip_prefix("cuda:") {
                    idx.parse().map_err(|e| anyhow::anyhow!("Invalid CUDA device index: {}", e))?
                } else {
                    0
                };
                Device::cuda_if_available(ordinal)
                    .map_err(|e| anyhow::anyhow!("Failed to initialize CUDA device {}: {}", ordinal, e))
            }
            #[cfg(not(feature = "cuda"))]
            anyhow::bail!(
                "CUDA support not compiled in. Rebuild with: cargo build --features cuda"
            )
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| anyhow::anyhow!("Failed to initialize Metal device: {}", e))
            }
            #[cfg(not(feature = "metal"))]
            anyhow::bail!(
                "Metal support not compiled in. Rebuild with: cargo build --features metal"
            )
        }
        _ => anyhow::bail!(
            "Unknown device '{}'. Supported: auto, cpu, cuda, cuda:N, metal",
            device_str
        ),
    }
}

/// Get a human-readable description of the current device.
pub fn device_info(device: &Device) -> String {
    match device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(d) => format!("CUDA (device {})", d.ordinal()),
        Device::Metal(_) => "Metal (Apple Silicon)".to_string(),
    }
}
```

### 1.2 Update CLI Binary

**File**: `src/bin/generate_audio.rs`

**Add to Args struct (around line 103)**:

```rust
/// Device to use for inference (auto, cpu, cuda, cuda:0, metal)
#[arg(long, default_value = "auto")]
device: String,
```

**Replace line 183**:

```rust
// OLD:
let device = Device::Cpu;

// NEW:
let device = qwen3_tts::parse_device(&args.device)?;
println!("Using device: {}", qwen3_tts::device_info(&device));
```

**Add import at top**:

```rust
use qwen3_tts::parse_device;
```

### 1.3 Update Public Exports

**File**: `src/lib.rs`

**Add to exports (around line 108)**:

```rust
pub use crate::{auto_device, parse_device, device_info};
```

### 1.4 Add Device Info to Model Loading

**File**: `src/lib.rs` in `Qwen3TTS::from_pretrained`

**Add logging after line 157**:

```rust
pub fn from_pretrained(model_id: &str, device: Device) -> Result<Self> {
    tracing::info!("Loading Qwen3-TTS from: {}", model_id);
    tracing::info!("Target device: {}", device_info(&device));  // ADD THIS
    // ... rest of function
}
```

### 1.5 Verification Checklist

After implementing Phase 1, verify:

- [ ] `cargo build --features cuda,cli` compiles successfully
- [ ] `./generate_audio --device cpu` works
- [ ] `./generate_audio --device cuda` works (on CUDA system)
- [ ] `./generate_audio --device auto` selects GPU when available
- [ ] Model weights load to correct device
- [ ] Generation produces valid audio on GPU
- [ ] No CUDA errors or memory leaks

---

## Phase 2: FlashAttention Integration

### 2.1 Update Cargo.toml

**Add to `[features]` section**:

```toml
[features]
default = ["cpu"]
cpu = []
cuda = ["candle-core/cuda", "candle-nn/cuda", "candle-transformers/cuda"]
metal = ["candle-core/metal", "candle-nn/metal", "candle-transformers/metal"]
accelerate = ["candle-core/accelerate", "candle-nn/accelerate", "candle-transformers/accelerate"]
mkl = ["candle-core/mkl", "candle-nn/mkl", "candle-transformers/mkl"]
cli = ["clap", "indicatif"]
hub = ["hf-hub"]

# NEW: FlashAttention support (requires CUDA + Ampere GPU)
flash-attn = ["cuda", "candle-flash-attn"]
```

**Add to `[dependencies]` section**:

```toml
# FlashAttention (optional) - requires CUDA and Ampere+ GPU
candle-flash-attn = { version = "0.8", optional = true }
```

### 2.2 Create FlashAttention Wrapper Module

**New file**: `src/models/flash_attention.rs`

```rust
//! FlashAttention wrapper for efficient GPU attention computation.
//!
//! This module provides a unified interface for attention computation that
//! automatically selects between FlashAttention (when available and beneficial)
//! and standard scaled dot-product attention.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

/// Configuration for attention computation.
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of query heads
    pub num_heads: usize,
    /// Number of key/value heads (for GQA)
    pub num_kv_heads: usize,
    /// Dimension per head
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Whether to use FlashAttention when available
    pub use_flash_attn: bool,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 16,
            num_kv_heads: 8,
            head_dim: 64,
            causal: true,
            use_flash_attn: true,
        }
    }
}

/// Check if FlashAttention is available for the given device and dtype.
pub fn flash_attn_available(device: &Device, dtype: DType) -> bool {
    #[cfg(feature = "flash-attn")]
    {
        // FlashAttention requires:
        // 1. CUDA device
        // 2. F16 or BF16 dtype
        // 3. Ampere or newer GPU (checked at runtime by candle-flash-attn)
        device.is_cuda() && matches!(dtype, DType::F16 | DType::BF16)
    }
    #[cfg(not(feature = "flash-attn"))]
    {
        let _ = (device, dtype);
        false
    }
}

/// Compute scaled dot-product attention.
///
/// Automatically selects FlashAttention when available and beneficial.
///
/// # Arguments
/// * `q` - Query tensor, shape: (batch, seq_len, num_heads, head_dim)
/// * `k` - Key tensor, shape: (batch, seq_len, num_kv_heads, head_dim)
/// * `v` - Value tensor, shape: (batch, seq_len, num_kv_heads, head_dim)
/// * `config` - Attention configuration
///
/// # Returns
/// Output tensor, shape: (batch, seq_len, num_heads, head_dim)
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &AttentionConfig,
) -> Result<Tensor> {
    let device = q.device();
    let dtype = q.dtype();

    #[cfg(feature = "flash-attn")]
    if config.use_flash_attn && flash_attn_available(device, dtype) {
        return flash_attention_forward(q, k, v, config);
    }

    // Fallback to standard attention
    standard_attention_forward(q, k, v, config)
}

/// FlashAttention forward pass.
#[cfg(feature = "flash-attn")]
fn flash_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &AttentionConfig,
) -> Result<Tensor> {
    use candle_flash_attn::flash_attn;

    // Ensure correct dtype (FlashAttention requires F16 or BF16)
    let (q, k, v, original_dtype) = if !matches!(q.dtype(), DType::F16 | DType::BF16) {
        let dtype = DType::F16;
        (
            q.to_dtype(dtype)?,
            k.to_dtype(dtype)?,
            v.to_dtype(dtype)?,
            Some(q.dtype()),
        )
    } else {
        (q.clone(), k.clone(), v.clone(), None)
    };

    // Compute softmax scale
    let softmax_scale = 1.0 / (config.head_dim as f32).sqrt();

    // Call FlashAttention
    // Input shape: (batch, seq_len, num_heads, head_dim)
    let output = flash_attn(&q, &k, &v, softmax_scale, config.causal)?;

    // Convert back to original dtype if needed
    if let Some(orig_dtype) = original_dtype {
        output.to_dtype(orig_dtype)
    } else {
        Ok(output)
    }
}

/// Standard scaled dot-product attention.
fn standard_attention_forward(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    config: &AttentionConfig,
) -> Result<Tensor> {
    let (b, seq_len, num_heads, head_dim) = q.dims4()?;
    let (_, kv_len, num_kv_heads, _) = k.dims4()?;

    // Transpose to (batch, heads, seq, dim)
    let q = q.transpose(1, 2)?;
    let k = k.transpose(1, 2)?;
    let v = v.transpose(1, 2)?;

    // Handle grouped-query attention (GQA)
    let (k, v) = if num_kv_heads != num_heads {
        let repeat_factor = num_heads / num_kv_heads;
        let k = k.repeat(&[1, repeat_factor, 1, 1])?;
        let v = v.repeat(&[1, repeat_factor, 1, 1])?;
        (k, v)
    } else {
        (k, v)
    };

    // Compute attention scores: Q @ K^T / sqrt(d)
    let scale = 1.0 / (head_dim as f64).sqrt();
    let attn_weights = q.matmul(&k.transpose(2, 3)?)? * scale;

    // Apply causal mask if needed
    let attn_weights = if config.causal && seq_len > 1 {
        let mask = create_causal_mask(seq_len, kv_len, q.device())?;
        attn_weights.broadcast_add(&mask)?
    } else {
        attn_weights
    };

    // Softmax
    let attn_weights = candle_nn::ops::softmax(&attn_weights, candle_core::D::Minus1)?;

    // Apply attention to values
    let output = attn_weights.matmul(&v)?;

    // Transpose back to (batch, seq, heads, dim)
    output.transpose(1, 2)
}

/// Create a causal attention mask.
fn create_causal_mask(seq_len: usize, kv_len: usize, device: &Device) -> Result<Tensor> {
    let mask = Tensor::ones((seq_len, kv_len), DType::F32, device)?
        .tril(0)?
        .log()?;  // Convert 0s to -inf, 1s to 0
    Ok(mask.unsqueeze(0)?.unsqueeze(0)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_attention() {
        let device = Device::Cpu;
        let b = 2;
        let seq = 10;
        let heads = 4;
        let head_dim = 64;

        let q = Tensor::randn(0f32, 1.0, (b, seq, heads, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (b, seq, heads, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (b, seq, heads, head_dim), &device).unwrap();

        let config = AttentionConfig {
            num_heads: heads,
            num_kv_heads: heads,
            head_dim,
            causal: true,
            use_flash_attn: false,
        };

        let output = scaled_dot_product_attention(&q, &k, &v, &config).unwrap();
        assert_eq!(output.dims(), &[b, seq, heads, head_dim]);
    }

    #[test]
    fn test_flash_attn_availability() {
        let cpu = Device::Cpu;
        assert!(!flash_attn_available(&cpu, DType::F32));
        assert!(!flash_attn_available(&cpu, DType::F16));
    }
}
```

### 2.3 Update Transformer Module

**File**: `src/models/transformer.rs`

**Add to module imports**:

```rust
#[cfg(feature = "flash-attn")]
use super::flash_attention::{scaled_dot_product_attention, AttentionConfig, flash_attn_available};
```

**Modify Attention struct to include flash_attn flag**:

```rust
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    use_flash_attn: bool,  // ADD THIS
}

impl Attention {
    pub fn new(config: &AttentionConfig, vb: VarBuilder, use_flash_attn: bool) -> Result<Self> {
        // ... existing code ...
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            use_flash_attn,
        })
    }
}
```

### 2.4 Add Use Flash Attention to Config

**File**: `src/models/config.rs`

**Add to model configs**:

```rust
pub struct TalkerConfig {
    // ... existing fields ...
    pub use_flash_attn: bool,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            // ... existing defaults ...
            use_flash_attn: cfg!(feature = "flash-attn"),
        }
    }
}
```

### 2.5 Add CLI Flag

**File**: `src/bin/generate_audio.rs`

**Add argument**:

```rust
/// Use FlashAttention for faster inference (requires CUDA + Ampere GPU)
#[arg(long)]
flash_attn: bool,
```

---

## Phase 3: Advanced Optimizations

### 3.1 BF16 Inference Support

**Goal**: Keep model weights in BF16 on GPU to reduce memory by 50%.

**File**: `src/lib.rs` - Modify weight loading

```rust
/// Data type for model inference.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceDType {
    /// 32-bit floating point (most compatible)
    F32,
    /// 16-bit floating point (faster on GPU, less precision)
    F16,
    /// Brain floating point 16 (good balance, requires Ampere+)
    BF16,
    /// Automatic selection based on device
    Auto,
}

impl InferenceDType {
    /// Select best dtype for device.
    pub fn resolve(self, device: &Device) -> DType {
        match self {
            Self::F32 => DType::F32,
            Self::F16 => DType::F16,
            Self::BF16 => DType::BF16,
            Self::Auto => {
                if device.is_cuda() {
                    DType::BF16  // Best for modern NVIDIA GPUs
                } else {
                    DType::F32  // Safe default for CPU/Metal
                }
            }
        }
    }
}

fn load_weights(path: &Path, device: &Device, dtype: InferenceDType) -> Result<HashMap<String, Tensor>> {
    let target_dtype = dtype.resolve(device);
    let tensors: HashMap<String, Tensor> = candle_core::safetensors::load(path, device)?;

    let tensors: HashMap<String, Tensor> = tensors
        .into_iter()
        .map(|(name, tensor)| {
            let converted = tensor.to_dtype(target_dtype).unwrap_or(tensor);
            (name, converted)
        })
        .collect();

    Ok(tensors)
}
```

### 3.2 CUDA Memory Utilities

**New file**: `src/cuda_utils.rs`

```rust
//! CUDA utility functions for memory management and synchronization.

use anyhow::Result;
use candle_core::Device;

/// Synchronize CUDA device (wait for all operations to complete).
/// No-op on non-CUDA devices.
pub fn synchronize(device: &Device) -> Result<()> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(cuda_dev) = device {
        cuda_dev.synchronize()?;
    }
    Ok(())
}

/// Get CUDA memory info (used, total) in bytes.
/// Returns None on non-CUDA devices.
#[cfg(feature = "cuda")]
pub fn cuda_memory_info(device: &Device) -> Option<(usize, usize)> {
    if let Device::Cuda(cuda_dev) = device {
        // Note: Candle doesn't expose this directly yet
        // This is a placeholder for when it does
        None
    } else {
        None
    }
}

/// Log current CUDA memory usage.
pub fn log_memory_usage(device: &Device) {
    #[cfg(feature = "cuda")]
    if let Some((used, total)) = cuda_memory_info(device) {
        let used_mb = used as f64 / 1024.0 / 1024.0;
        let total_mb = total as f64 / 1024.0 / 1024.0;
        tracing::info!("CUDA memory: {:.1} / {:.1} MB ({:.1}%)",
            used_mb, total_mb, used_mb / total_mb * 100.0);
    }
}

/// Benchmark wrapper that handles CUDA synchronization.
pub struct CudaBenchmark {
    device: Device,
    start: std::time::Instant,
}

impl CudaBenchmark {
    pub fn start(device: &Device) -> Result<Self> {
        synchronize(device)?;
        Ok(Self {
            device: device.clone(),
            start: std::time::Instant::now(),
        })
    }

    pub fn elapsed(&self) -> Result<std::time::Duration> {
        synchronize(&self.device)?;
        Ok(self.start.elapsed())
    }

    pub fn elapsed_secs(&self) -> Result<f64> {
        Ok(self.elapsed()?.as_secs_f64())
    }
}
```

### 3.3 Streaming Generation with GPU

**Optimization**: Overlap audio decoding with next token generation.

```rust
/// Streaming TTS session with GPU pipelining.
pub struct StreamingSession {
    // ... existing fields ...

    /// Pending codes waiting for decode
    pending_codes: Vec<Vec<u32>>,
    /// Chunk size for streaming decode
    chunk_frames: usize,
}

impl StreamingSession {
    /// Generate next chunk of audio while preparing next tokens.
    pub fn next_chunk(&mut self) -> Result<Option<AudioChunk>> {
        // 1. If we have enough pending codes, decode them
        if self.pending_codes.len() >= self.chunk_frames {
            let codes_to_decode: Vec<_> = self.pending_codes.drain(..self.chunk_frames).collect();
            let audio = self.decode_codes(&codes_to_decode)?;

            // 2. While decoding (on GPU), generate next tokens (also on GPU)
            //    CUDA streams would allow true overlap, but sequential is still faster than CPU
            self.generate_next_tokens()?;

            return Ok(Some(audio));
        }

        // Generate more tokens
        if !self.is_finished() {
            self.generate_next_tokens()?;
        }

        Ok(None)
    }
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[cfg(test)]
mod gpu_tests {
    use super::*;

    #[test]
    fn test_parse_device_cpu() {
        let device = parse_device("cpu").unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_parse_device_auto() {
        let device = parse_device("auto").unwrap();
        // Should succeed regardless of hardware
        assert!(matches!(device, Device::Cpu | Device::Cuda(_) | Device::Metal(_)));
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn test_parse_device_cuda() {
        // Only run if CUDA is available
        if let Ok(device) = parse_device("cuda") {
            assert!(matches!(device, Device::Cuda(_)));
        }
    }

    #[test]
    fn test_tensor_device_propagation() {
        let device = auto_device().unwrap();
        let tensor = Tensor::zeros((2, 3), DType::F32, &device).unwrap();
        assert_eq!(tensor.device().location(), device.location());
    }
}
```

### Integration Tests

**File**: `tests/cuda_integration.rs`

```rust
//! Integration tests for CUDA support.
//!
//! Run with: cargo test --features cuda -- --ignored

use qwen3_tts::{auto_device, parse_device, Qwen3TTS, SynthesisOptions};

#[test]
#[ignore] // Only run explicitly (requires GPU)
fn test_cuda_model_loading() {
    let device = parse_device("cuda").expect("CUDA not available");

    // This test requires model files to be present
    let model_path = std::env::var("QWEN3_TTS_MODEL_PATH")
        .unwrap_or_else(|_| "test_data/model".to_string());

    let model = Qwen3TTS::from_pretrained(&model_path, device)
        .expect("Failed to load model on CUDA");

    assert!(model.device().is_cuda());
}

#[test]
#[ignore]
fn test_cuda_generation() {
    let device = parse_device("cuda").expect("CUDA not available");
    let model_path = std::env::var("QWEN3_TTS_MODEL_PATH")
        .unwrap_or_else(|_| "test_data/model".to_string());

    let model = Qwen3TTS::from_pretrained(&model_path, device).unwrap();

    let options = SynthesisOptions {
        max_length: 10,  // Short for testing
        ..Default::default()
    };

    let audio = model.synthesize("Hello", Some(options)).unwrap();
    assert!(audio.samples().len() > 0);
}

#[test]
#[ignore]
#[cfg(feature = "flash-attn")]
fn test_flash_attention() {
    let device = parse_device("cuda").expect("CUDA not available");

    // Create test tensors
    let q = Tensor::randn(0f32, 1.0, (1, 10, 16, 64), &device).unwrap();
    let k = Tensor::randn(0f32, 1.0, (1, 10, 8, 64), &device).unwrap();
    let v = Tensor::randn(0f32, 1.0, (1, 10, 8, 64), &device).unwrap();

    let config = AttentionConfig {
        num_heads: 16,
        num_kv_heads: 8,
        head_dim: 64,
        causal: true,
        use_flash_attn: true,
    };

    let output = scaled_dot_product_attention(&q, &k, &v, &config).unwrap();
    assert_eq!(output.dims(), &[1, 10, 16, 64]);
}
```

### Benchmark Tests

**File**: `benches/cuda_benchmark.rs`

```rust
use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use qwen3_tts::{parse_device, Qwen3TTS, SynthesisOptions};

fn benchmark_generation(c: &mut Criterion) {
    let mut group = c.benchmark_group("generation");

    let test_texts = [
        ("short", "Hello world"),
        ("medium", "The quick brown fox jumps over the lazy dog."),
        ("long", "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole, filled with the ends of worms and an oozy smell."),
    ];

    // CPU benchmark
    if let Ok(device) = parse_device("cpu") {
        let model = Qwen3TTS::from_pretrained("test_data/model", device).unwrap();

        for (name, text) in &test_texts {
            group.bench_with_input(BenchmarkId::new("cpu", name), text, |b, text| {
                b.iter(|| {
                    let options = SynthesisOptions { max_length: 50, ..Default::default() };
                    model.synthesize(text, Some(options)).unwrap()
                });
            });
        }
    }

    // CUDA benchmark
    #[cfg(feature = "cuda")]
    if let Ok(device) = parse_device("cuda") {
        let model = Qwen3TTS::from_pretrained("test_data/model", device).unwrap();

        for (name, text) in &test_texts {
            group.bench_with_input(BenchmarkId::new("cuda", name), text, |b, text| {
                b.iter(|| {
                    let options = SynthesisOptions { max_length: 50, ..Default::default() };
                    model.synthesize(text, Some(options)).unwrap()
                });
            });
        }
    }

    group.finish();
}

criterion_group!(benches, benchmark_generation);
criterion_main!(benches);
```

---

## Build & Deployment

### Build Commands

```bash
# CPU only (default)
cargo build --release --features cli

# CUDA support
cargo build --release --features cuda,cli

# CUDA + FlashAttention (requires Ampere+ GPU)
cargo build --release --features flash-attn,cli

# Metal (Apple Silicon)
cargo build --release --features metal,cli

# With MKL acceleration (Intel CPUs)
cargo build --release --features mkl,cli

# All features for development
cargo build --release --features cuda,flash-attn,cli,hub
```

### Environment Variables

```bash
# Cache CUDA kernel compilation (speeds up subsequent builds)
export CANDLE_FLASH_ATTN_BUILD_DIR=/tmp/candle-flash-attn-cache

# Specify CUDA architecture (optional, auto-detected by default)
export CUDA_COMPUTE_CAP=86  # For RTX 30xx

# Enable CUDA memory debugging
export CUDA_LAUNCH_BLOCKING=1
```

### Docker Build

```dockerfile
# CUDA development image
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Build with CUDA support
WORKDIR /app
COPY . .
RUN cargo build --release --features cuda,cli

# Runtime image
FROM nvidia/cuda:12.2-runtime-ubuntu22.04
COPY --from=builder /app/target/release/generate_audio /usr/local/bin/
ENTRYPOINT ["generate_audio"]
```

### CI Configuration

```yaml
# .github/workflows/cuda.yml
name: CUDA Build

on:
  push:
    branches: [main]
  pull_request:

jobs:
  build-cuda:
    runs-on: ubuntu-latest
    container:
      image: nvidia/cuda:12.2-devel-ubuntu22.04

    steps:
      - uses: actions/checkout@v4

      - name: Install Rust
        run: |
          curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
          echo "$HOME/.cargo/bin" >> $GITHUB_PATH

      - name: Build with CUDA
        run: cargo build --release --features cuda,cli
        working-directory: qwen3-tts-rs

      - name: Run tests (CPU only in CI)
        run: cargo test --features cli
        working-directory: qwen3-tts-rs
```

---

## Performance Benchmarks

### Expected Performance (Targets)

| Configuration | Model | Hardware | RTF | VRAM |
|---------------|-------|----------|-----|------|
| CPU | 1.7B | i9-13900K | 5-8x | - |
| CPU + MKL | 1.7B | i9-13900K | 3-5x | - |
| CUDA | 1.7B | RTX 4090 | 0.8-1.2x | 6GB |
| CUDA + FA2 | 1.7B | RTX 4090 | 0.4-0.6x | 4GB |
| CUDA | 0.6B | RTX 4090 | 0.4-0.6x | 3GB |
| CUDA + FA2 | 0.6B | RTX 4090 | 0.2-0.4x | 2GB |
| Metal | 1.7B | M3 Max | 1.0-1.5x | Unified |

### Benchmark Script

```bash
#!/bin/bash
# benchmark.sh - Run performance benchmarks

set -e

MODEL_PATH="${MODEL_PATH:-test_data/model}"
TEXT="The quick brown fox jumps over the lazy dog near the riverbank."
FRAMES=100

echo "=== Qwen3-TTS Performance Benchmark ==="
echo "Model: $MODEL_PATH"
echo "Text: $TEXT"
echo "Frames: $FRAMES"
echo ""

# CPU benchmark
echo "--- CPU ---"
time ./target/release/generate_audio \
    --model-dir "$MODEL_PATH" \
    --text "$TEXT" \
    --frames $FRAMES \
    --device cpu \
    --output-dir /tmp/bench_cpu

# CUDA benchmark (if available)
if cargo build --release --features cuda,cli 2>/dev/null; then
    echo ""
    echo "--- CUDA ---"
    time ./target/release/generate_audio \
        --model-dir "$MODEL_PATH" \
        --text "$TEXT" \
        --frames $FRAMES \
        --device cuda \
        --output-dir /tmp/bench_cuda
fi

# CUDA + FlashAttention benchmark
if cargo build --release --features flash-attn,cli 2>/dev/null; then
    echo ""
    echo "--- CUDA + FlashAttention ---"
    time ./target/release/generate_audio \
        --model-dir "$MODEL_PATH" \
        --text "$TEXT" \
        --frames $FRAMES \
        --device cuda \
        --flash-attn \
        --output-dir /tmp/bench_flash
fi

echo ""
echo "=== Benchmark Complete ==="
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Not Found
```
Error: CUDA support not compiled in
```
**Solution**: Rebuild with CUDA feature:
```bash
cargo build --release --features cuda,cli
```

#### 2. CUDA Out of Memory
```
Error: CUDA error: out of memory
```
**Solutions**:
- Use smaller model (0.6B instead of 1.7B)
- Enable FlashAttention to reduce memory
- Reduce batch size / max_length
- Close other GPU applications

#### 3. FlashAttention Compute Capability
```
Error: FlashAttention requires compute capability >= 8.0
```
**Solution**: Your GPU is too old for FlashAttention v2. Options:
- Use standard attention (remove `--flash-attn` flag)
- Use `candle-flash-attn-v1` for Turing GPUs

#### 4. Slow First Inference
**Cause**: CUDA kernels compiling on first run.
**Solution**:
- Set `CANDLE_FLASH_ATTN_BUILD_DIR` to cache compilation
- Warm up with a short inference before benchmarking

#### 5. Numerical Differences CPU vs GPU
**Cause**: Floating point accumulation order differs.
**Note**: Small differences (<1e-5) are expected and acceptable.

### Debug Commands

```bash
# Check CUDA availability
nvidia-smi

# Check Rust CUDA features
cargo tree -f "{p} {f}" | grep candle

# Verbose build to see CUDA compilation
RUST_BACKTRACE=1 cargo build --release --features cuda,cli -vv

# Test CUDA device detection
cargo run --features cuda --example test_cuda_device
```

---

## References

### Documentation
- [Candle Documentation](https://huggingface.github.io/candle/)
- [candle-flash-attn crate](https://crates.io/crates/candle-flash-attn)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)

### Related Projects
- [mistral.rs](https://github.com/EricLBuehler/mistral.rs) - LLM inference with FlashAttention
- [candle-vllm](https://github.com/EricLBuehler/candle-vllm) - vLLM-style serving
- [atoma-infer](https://github.com/atoma-network/atoma-infer) - Serverless LLM inference

### Official Implementations
- [Qwen3-TTS Python](https://github.com/QwenLM/Qwen3-TTS)
- [MLX Audio](https://github.com/Blaizzy/mlx-audio)

---

## Changelog

### v0.1.0 (Planned)
- [ ] Phase 1: Basic CUDA support
- [ ] Phase 2: FlashAttention integration
- [ ] Phase 3: Advanced optimizations

---

*Document maintained by the Qwen3-TTS Rust team*

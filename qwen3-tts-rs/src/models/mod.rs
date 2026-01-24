//! Neural network models for Qwen3-TTS
//!
//! This module contains:
//! - `config`: Model configuration
//! - `qwen3_tts`: Main TTS model (Talker)
//! - `code_predictor`: Acoustic token predictor
//! - `speaker`: Speaker encoder (ECAPA-TDNN)
//! - `codec`: Audio codec for encoding/decoding

pub mod code_predictor;
pub mod codec;
pub mod config;
pub mod qwen3_tts;
pub mod speaker;
pub mod talker;

pub use code_predictor::{CodePredictor, CodePredictorConfig};
pub use config::Qwen3TTSConfig;
pub use qwen3_tts::{KVCache, Qwen3TTSModel, RotaryEmbedding};
pub use speaker::SpeakerEncoder;
pub use talker::{TalkerConfig, TalkerModel};

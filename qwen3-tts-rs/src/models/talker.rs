//! TalkerModel for autoregressive semantic token generation
//!
//! The Talker model generates semantic tokens (group 1) from text input.
//! It uses:
//! - Text embedding (vocab_size=151936 → 2048)
//! - Text projection (2048 → 1024 via SwiGLU)
//! - 28 transformer decoder layers with KV caching
//! - Codec embedding for generated tokens (3072 → 1024)
//! - Codec head for predicting next semantic token (1024 → 3072)
//!
//! ## CustomVoice Support
//!
//! For CustomVoice models, the input format includes:
//! - ChatML text tokens: `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`
//! - Codec prefix: `[codec_think, think_bos, language, think_eos]`
//! - Speaker token embedding
//! - Codec BOS: `[codec_pad, codec_bos]`

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor};
use candle_nn::{embedding, linear_no_bias, rms_norm, Embedding, Linear, RmsNorm, VarBuilder};
use std::collections::HashMap;

use super::config::Qwen3TTSConfig;
use super::transformer::{DecoderLayer, KVCache, MRoPE, RoPEType, RotaryEmbedding};

/// ChatML special token IDs
pub mod special_tokens {
    pub const IM_START: u32 = 151644;
    pub const IM_END: u32 = 151645;
    pub const ASSISTANT: u32 = 77091;
    pub const NEWLINE: u32 = 198;
}

/// TTS special token IDs (text vocabulary tokens for TTS generation)
pub mod tts_tokens {
    pub const TTS_PAD: u32 = 151671;
    pub const TTS_BOS: u32 = 151672;
    pub const TTS_EOS: u32 = 151673;
}

/// Codec special token IDs
pub mod codec_tokens {
    pub const CODEC_THINK: u32 = 2154;
    pub const CODEC_NOTHINK: u32 = 2155;
    pub const CODEC_THINK_BOS: u32 = 2156;
    pub const CODEC_THINK_EOS: u32 = 2157;
    pub const CODEC_PAD: u32 = 2148;
    pub const CODEC_BOS: u32 = 2149;
}

/// Language IDs for codec prefix
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

impl Language {
    /// Get the codec language token ID
    pub fn token_id(&self) -> u32 {
        match self {
            Language::Chinese => 2055,
            Language::English => 2050,
            Language::Japanese => 2058,
            Language::Korean => 2064,
            Language::German => 2053,
            Language::French => 2061,
            Language::Russian => 2069,
            Language::Portuguese => 2071,
            Language::Spanish => 2054,
            Language::Italian => 2070,
        }
    }
}

/// Speaker IDs for CustomVoice model
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

impl Speaker {
    /// Get the speaker token ID
    pub fn token_id(&self) -> u32 {
        match self {
            Speaker::Serena => 3066,
            Speaker::Vivian => 3065,
            Speaker::UncleFu => 3010,
            Speaker::Ryan => 3061,
            Speaker::Aiden => 2861,
            Speaker::OnoAnna => 2873,
            Speaker::Sohee => 2864,
            Speaker::Eric => 2875,
            Speaker::Dylan => 2878,
        }
    }

    /// Get the native language for this speaker
    pub fn native_language(&self) -> Language {
        match self {
            Speaker::Serena
            | Speaker::Vivian
            | Speaker::UncleFu
            | Speaker::Eric
            | Speaker::Dylan => Language::Chinese,
            Speaker::Ryan | Speaker::Aiden => Language::English,
            Speaker::OnoAnna => Language::Japanese,
            Speaker::Sohee => Language::Korean,
        }
    }
}

/// Talker model configuration
#[derive(Debug, Clone)]
pub struct TalkerConfig {
    /// Text vocabulary size (151936)
    pub text_vocab_size: usize,
    /// Text embedding dimension (2048)
    pub text_embed_dim: usize,
    /// Hidden dimension (1024)
    pub hidden_size: usize,
    /// Intermediate size for text projection (2048)
    pub text_proj_intermediate: usize,
    /// Intermediate size for MLP (3072)
    pub intermediate_size: usize,
    /// Number of transformer layers (28)
    pub num_hidden_layers: usize,
    /// Number of attention heads (16)
    pub num_attention_heads: usize,
    /// Number of KV heads for GQA (8)
    pub num_key_value_heads: usize,
    /// Head dimension (128)
    pub head_dim: usize,
    /// RMS norm epsilon
    pub rms_norm_eps: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Max position embeddings
    pub max_position_embeddings: usize,
    /// Codec vocabulary size (3072 - includes special tokens)
    pub codec_vocab_size: usize,
    /// MRoPE section for multimodal rotary embedding [T, H, W]
    /// None = use standard RoPE, Some([24, 20, 20]) = use interleaved MRoPE
    pub mrope_section: Option<[usize; 3]>,
}

impl Default for TalkerConfig {
    fn default() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 1024,
            text_proj_intermediate: 2048,
            intermediate_size: 3072,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 8192,
            codec_vocab_size: 3072,
            mrope_section: None, // Standard model uses standard RoPE
        }
    }
}

impl TalkerConfig {
    /// Create config for CustomVoice model (larger hidden dimension, MRoPE)
    pub fn custom_voice() -> Self {
        Self {
            text_vocab_size: 151936,
            text_embed_dim: 2048,
            hidden_size: 2048, // CustomVoice uses 2048
            text_proj_intermediate: 2048,
            intermediate_size: 6144, // CustomVoice uses 6144
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            max_position_embeddings: 8192,
            codec_vocab_size: 3072,
            mrope_section: Some([24, 20, 20]), // CustomVoice uses MRoPE
        }
    }

    /// Convert to a Qwen3TTSConfig for building DecoderLayers
    pub fn to_layer_config(&self) -> Qwen3TTSConfig {
        Qwen3TTSConfig {
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: Some(self.num_key_value_heads),
            head_dim_override: Some(self.head_dim),
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            ..Default::default()
        }
    }
}

/// Text projection with SwiGLU activation
/// Maps text embeddings (2048) to hidden dimension (1024)
pub struct TextProjection {
    fc1: Linear,
    fc2: Linear,
}

impl TextProjection {
    /// Create from VarBuilder with config dimensions
    pub fn new(config: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(
            config.text_embed_dim,
            config.text_proj_intermediate,
            vb.pp("linear_fc1"),
        )?;
        let fc2 = candle_nn::linear(
            config.text_proj_intermediate,
            config.hidden_size,
            vb.pp("linear_fc2"),
        )?;
        Ok(Self { fc1, fc2 })
    }

    /// Forward pass: fc1 -> silu -> fc2
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(x)?;
        let hidden = candle_nn::ops::silu(&hidden)?;
        Ok(self.fc2.forward(&hidden)?)
    }
}

/// TalkerModel for autoregressive semantic token generation
pub struct TalkerModel {
    /// Text embedding [text_vocab_size, text_embed_dim]
    text_embedding: Embedding,
    /// Text projection (2048 -> 1024)
    text_projection: TextProjection,
    /// Codec embedding [codec_vocab_size, hidden_size]
    codec_embedding: Embedding,
    /// Transformer decoder layers
    layers: Vec<DecoderLayer>,
    /// Final RMS norm
    norm: RmsNorm,
    /// Codec head (hidden_size -> codec_vocab_size)
    codec_head: Linear,
    /// Rotary position embedding (standard or MRoPE)
    rope: RoPEType,
    /// Configuration
    config: TalkerConfig,
    /// Device
    device: Device,
}

impl TalkerModel {
    /// Load model from weight tensors with auto-detected config
    ///
    /// Inspects `talker.model.norm.weight` shape to determine model variant:
    /// - hidden_size=1024 → Base model (`TalkerConfig::default()`)
    /// - hidden_size=2048 → CustomVoice model (`TalkerConfig::custom_voice()`)
    pub fn from_weights(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?;
        let hidden_size = norm_weight.dim(0)?;
        let config = if hidden_size == 2048 {
            TalkerConfig::custom_voice()
        } else {
            TalkerConfig::default()
        };
        Self::from_weights_with_config(weights, config, device)
    }

    /// Load model with explicit config
    pub fn from_weights_with_config(
        weights: &HashMap<String, Tensor>,
        config: TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        let vb = VarBuilder::from_tensors(weights.clone(), DType::F32, device);
        let talker = vb.pp("talker");
        let model = talker.pp("model");
        let layer_config = config.to_layer_config();

        let text_embedding = embedding(
            config.text_vocab_size,
            config.text_embed_dim,
            model.pp("text_embedding"),
        )?;
        let text_projection = TextProjection::new(&config, talker.pp("text_projection"))?;
        let codec_embedding = embedding(
            config.codec_vocab_size,
            config.hidden_size,
            model.pp("codec_embedding"),
        )?;
        let norm = rms_norm(config.hidden_size, config.rms_norm_eps, model.pp("norm"))?;
        let codec_head = linear_no_bias(
            config.hidden_size,
            config.codec_vocab_size,
            talker.pp("codec_head"),
        )?;

        let layers = (0..config.num_hidden_layers)
            .map(|i| DecoderLayer::new(&layer_config, model.pp(format!("layers.{}", i))))
            .collect::<Result<Vec<_>>>()?;

        // RoPE - use MRoPE if mrope_section is configured
        let rope = if let Some(mrope_section) = config.mrope_section {
            RoPEType::Multimodal(MRoPE::new(
                config.head_dim,
                config.rope_theta,
                mrope_section,
                device,
            )?)
        } else {
            RoPEType::Standard(RotaryEmbedding::new(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                device,
            )?)
        };

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm,
            codec_head,
            rope,
            config,
            device: device.clone(),
        })
    }

    /// Prefill with text input
    ///
    /// Processes text tokens and returns (hidden_states, logits) for the last position.
    /// KV caches are populated for subsequent generation steps.
    pub fn prefill(
        &self,
        input_ids: &Tensor,
        kv_caches: &mut [KVCache],
    ) -> Result<(Tensor, Tensor)> {
        let seq_len = input_ids.dim(1)?;

        // Embed text tokens
        let input_ids_flat = input_ids.flatten_all()?;
        let text_embed = self.text_embedding.forward(&input_ids_flat)?;
        let text_embed = text_embed.reshape((1, seq_len, self.config.text_embed_dim))?;

        // Debug: print embedding values
        #[cfg(debug_assertions)]
        {
            let embed_vec: Vec<f32> = text_embed.i((0, 0, ..5))?.to_vec1()?;
            eprintln!("DEBUG TALKER: text_embed[0,0,:5] = {:?}", embed_vec);
        }

        // Project to hidden dimension
        let mut hidden = self.text_projection.forward(&text_embed)?;

        // Debug: print projected values
        #[cfg(debug_assertions)]
        {
            let proj_vec: Vec<f32> = hidden.i((0, 0, ..5))?.to_vec1()?;
            eprintln!("DEBUG TALKER: after_proj[0,0,:5] = {:?}", proj_vec);
        }

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, 0)?;

        // Run through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Get logits for last position
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.codec_head.forward(&last_hidden)?;

        // Debug: print logits for first few tokens
        #[cfg(debug_assertions)]
        {
            let logits_flat: Vec<f32> = logits.squeeze(0)?.squeeze(0)?.to_vec1()?;
            let argmax = logits_flat
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i);
            eprintln!(
                "DEBUG TALKER: logits shape = {:?}, argmax = {:?}",
                logits.shape(),
                argmax
            );
            eprintln!(
                "DEBUG TALKER: logits[0:5] = {:?}",
                &logits_flat[..5.min(logits_flat.len())]
            );
            // Print logits around token 439 (Rust result) and 1501 (Python result)
            eprintln!(
                "DEBUG TALKER: logits[438:442] = {:?}",
                &logits_flat[438..442.min(logits_flat.len())]
            );
            eprintln!(
                "DEBUG TALKER: logits[1500:1504] = {:?}",
                &logits_flat[1500..1504.min(logits_flat.len())]
            );
        }

        Ok((hidden, logits))
    }

    /// Prefill for CustomVoice model with speaker and language
    ///
    /// Constructs the full input sequence matching the Python implementation:
    /// - Positions 0-2: role prefix (text_proj of im_start, assistant, newline)
    /// - Positions 3-8: tts_pad/tts_bos ADDED with codec embeddings
    ///   - 3: tts_pad + codec_think
    ///   - 4: tts_pad + codec_think_bos
    ///   - 5: tts_pad + language_id
    ///   - 6: tts_pad + codec_think_eos
    ///   - 7: tts_pad + speaker
    ///   - 8: tts_bos + codec_pad
    /// - Position 9: first_text_proj + codec_bos
    ///
    /// Returns (hidden_states, logits) for generation.
    pub fn prefill_custom_voice(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
        kv_caches: &mut [KVCache],
    ) -> Result<(Tensor, Tensor)> {
        use codec_tokens::*;
        use special_tokens::*;
        use tts_tokens::*;

        // === 1. Role prefix: text_proj([im_start, assistant, newline]) ===
        let role_prefix_ids = Tensor::new(&[IM_START, ASSISTANT, NEWLINE], &self.device)?;
        let role_prefix_embed = self.text_embedding.forward(&role_prefix_ids)?;
        let role_prefix_embed = role_prefix_embed.unsqueeze(0)?;
        let role_prefix_hidden = self.text_projection.forward(&role_prefix_embed)?; // [1, 3, hidden]

        // === 2. Codec embeddings: [think, think_bos, lang, think_eos, speaker, pad, bos] ===
        let codec_tokens_list = vec![
            CODEC_THINK,
            CODEC_THINK_BOS,
            language.token_id(),
            CODEC_THINK_EOS,
            speaker.token_id(),
            CODEC_PAD,
            CODEC_BOS,
        ];
        let codec_ids = Tensor::new(codec_tokens_list.as_slice(), &self.device)?;
        let codec_embed = self.codec_embedding.forward(&codec_ids)?;
        let codec_embed = codec_embed.unsqueeze(0)?; // [1, 7, hidden]

        // === 3. TTS pad/bos text embeddings ===
        // We need 5 tts_pad + 1 tts_bos = 6 total to add with first 6 codec tokens
        let tts_pad_id = Tensor::new(&[TTS_PAD], &self.device)?;
        let tts_pad_embed = self.text_embedding.forward(&tts_pad_id)?;
        let tts_pad_embed = tts_pad_embed.unsqueeze(0)?; // [1, 1, embed_dim]
        let tts_pad_proj = self.text_projection.forward(&tts_pad_embed)?; // [1, 1, hidden]

        let tts_bos_id = Tensor::new(&[TTS_BOS], &self.device)?;
        let tts_bos_embed = self.text_embedding.forward(&tts_bos_id)?;
        let tts_bos_embed = tts_bos_embed.unsqueeze(0)?;
        let tts_bos_proj = self.text_projection.forward(&tts_bos_embed)?; // [1, 1, hidden]

        // Expand tts_pad to 5 copies and concat with tts_bos
        let tts_pad_expanded = tts_pad_proj.broadcast_as((1, 5, self.config.hidden_size))?;
        let tts_text_embed = Tensor::cat(&[&tts_pad_expanded, &tts_bos_proj], 1)?; // [1, 6, hidden]

        // Add tts text embeddings with first 6 codec embeddings
        let codec_first6 = codec_embed.i((.., ..6, ..))?; // [1, 6, hidden]
        let codec_hidden = tts_text_embed.add(&codec_first6)?; // [1, 6, hidden]

        // === 4. Combine role prefix and codec part ===
        let mut hidden = Tensor::cat(&[&role_prefix_hidden, &codec_hidden], 1)?; // [1, 9, hidden]

        // === 5. Add first text token (text_proj + codec_bos) ===
        if !text_tokens.is_empty() {
            let first_text_id = Tensor::new(&[text_tokens[0]], &self.device)?;
            let first_text_embed = self.text_embedding.forward(&first_text_id)?;
            let first_text_embed = first_text_embed.unsqueeze(0)?;
            let first_text_proj = self.text_projection.forward(&first_text_embed)?; // [1, 1, hidden]

            // Add with codec_bos (last token of codec_embed)
            let codec_bos_embed = codec_embed.i((.., 6..7, ..))?; // [1, 1, hidden]
            let first_combined = first_text_proj.add(&codec_bos_embed)?;

            hidden = Tensor::cat(&[&hidden, &first_combined], 1)?; // [1, 10, hidden]
        }

        let seq_len = hidden.dim(1)?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, 0)?;

        // Run through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Get logits for last position
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.codec_head.forward(&last_hidden)?;

        Ok((hidden, logits))
    }

    /// Generate step with pre-built input embedding
    ///
    /// This allows the caller to build the full input embedding externally
    /// (e.g., semantic_embed + acoustic_embeds + text_embed for CustomVoice).
    pub fn generate_step_with_embed(
        &self,
        input_embed: &Tensor,
        kv_caches: &mut [KVCache],
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Create causal mask for single token (attends to all previous positions)
        let mask = self.create_causal_mask(1, offset)?;

        // Run through all layers with KV cache
        let mut hidden = input_embed.clone();
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(
                &hidden,
                &self.rope,
                Some(&mask),
                Some(&mut kv_caches[i]),
                offset,
            )?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Get logits
        let logits = self.codec_head.forward(&hidden)?;

        Ok((hidden, logits))
    }

    /// Get tts_pad text embedding (projected)
    ///
    /// This is added to codec embeddings during CustomVoice generation.
    pub fn get_tts_pad_embed(&self) -> Result<Tensor> {
        use tts_tokens::TTS_PAD;
        let pad_id = Tensor::new(&[TTS_PAD], &self.device)?;
        let pad_embed = self.text_embedding.forward(&pad_id)?;
        let pad_embed = pad_embed.unsqueeze(0)?;
        self.text_projection.forward(&pad_embed)
    }

    /// Get tts_eos text embedding (projected)
    ///
    /// This marks the end of text input.
    pub fn get_tts_eos_embed(&self) -> Result<Tensor> {
        use tts_tokens::TTS_EOS;
        let eos_id = Tensor::new(&[TTS_EOS], &self.device)?;
        let eos_embed = self.text_embedding.forward(&eos_id)?;
        let eos_embed = eos_embed.unsqueeze(0)?;
        self.text_projection.forward(&eos_embed)
    }

    /// Get projected text embeddings for a sequence of token IDs
    ///
    /// Returns [1, seq_len, hidden_size] tensor of projected text embeddings.
    pub fn get_projected_text_embeddings(&self, token_ids: &[u32]) -> Result<Tensor> {
        if token_ids.is_empty() {
            // Return empty tensor with correct shape
            return Ok(Tensor::zeros(
                (1, 0, self.config.hidden_size),
                candle_core::DType::F32,
                &self.device,
            )?);
        }

        let ids: Vec<u32> = token_ids.to_vec();
        let ids_tensor = Tensor::new(ids.as_slice(), &self.device)?;
        let embeds = self.text_embedding.forward(&ids_tensor)?;
        let embeds = embeds.unsqueeze(0)?; // [1, seq_len, text_embed_dim]
        self.text_projection.forward(&embeds)
    }

    /// Full forward pass without KV caching (for testing)
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

        // Embed text tokens
        let input_ids_flat = input_ids.flatten_all()?;
        let text_embed = self.text_embedding.forward(&input_ids_flat)?;
        let text_embed = text_embed.reshape((1, seq_len, self.config.text_embed_dim))?;

        // Project to hidden dimension
        let mut hidden = self.text_projection.forward(&text_embed)?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, 0)?;

        // Run through all layers without KV cache
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), None, 0)?;
        }

        // Final norm
        hidden = self.norm.forward(&hidden)?;

        // Get logits for all positions
        let logits = self.codec_head.forward(&hidden)?;

        Ok(logits)
    }

    fn create_causal_mask(&self, seq_len: usize, offset: usize) -> Result<Tensor> {
        let total_len = offset + seq_len;
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| {
                (0..total_len).map(move |j| {
                    if j <= offset + i {
                        0.0
                    } else {
                        f32::NEG_INFINITY
                    }
                })
            })
            .collect();

        Ok(Tensor::new(mask.as_slice(), &self.device)?.reshape((1, 1, seq_len, total_len))?)
    }

    /// Create new KV caches for generation
    pub fn new_kv_caches(&self) -> Vec<KVCache> {
        (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect()
    }

    /// Get codec embedding for a token (used by code predictor)
    pub fn get_codec_embedding(&self, token_id: u32) -> Result<Tensor> {
        let token_tensor = Tensor::new(&[token_id], &self.device)?;
        let embed = self.codec_embedding.forward(&token_tensor)?;
        Ok(embed.unsqueeze(0)?) // [1, 1, hidden_size]
    }

    /// Get config
    pub fn config(&self) -> &TalkerConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_talker_config_default() {
        let config = TalkerConfig::default();
        assert_eq!(config.text_vocab_size, 151936);
        assert_eq!(config.hidden_size, 1024);
        assert_eq!(config.num_hidden_layers, 28);
        assert_eq!(config.num_attention_heads, 16);
        assert_eq!(config.num_key_value_heads, 8);
        assert_eq!(config.head_dim, 128);
    }
}

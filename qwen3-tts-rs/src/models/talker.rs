//! TalkerModel for autoregressive semantic token generation
//!
//! The Talker model generates semantic tokens (group 1) from text input.
//! It uses:
//! - Text embedding (vocab_size=151936 → 2048)
//! - Text projection (2048 → 1024 via SwiGLU)
//! - 28 transformer decoder layers with KV caching
//! - Codec embedding for generated tokens (3072 → 1024)
//! - Codec head for predicting next semantic token (1024 → 3072)

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor, D};
use std::collections::HashMap;

use super::qwen3_tts::{KVCache, RotaryEmbedding};
use crate::generation::GenerationConfig;

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
        }
    }
}

/// Text projection with SwiGLU activation
/// Maps text embeddings (2048) to hidden dimension (1024)
pub struct TextProjection {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
}

impl TextProjection {
    /// Create from weight tensors
    pub fn from_weights(weights: &HashMap<String, Tensor>) -> Result<Self> {
        let fc1_weight = weights
            .get("talker.text_projection.linear_fc1.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing text_projection.linear_fc1.weight"))?
            .clone();
        let fc1_bias = weights
            .get("talker.text_projection.linear_fc1.bias")
            .ok_or_else(|| anyhow::anyhow!("Missing text_projection.linear_fc1.bias"))?
            .clone();
        let fc2_weight = weights
            .get("talker.text_projection.linear_fc2.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing text_projection.linear_fc2.weight"))?
            .clone();
        let fc2_bias = weights
            .get("talker.text_projection.linear_fc2.bias")
            .ok_or_else(|| anyhow::anyhow!("Missing text_projection.linear_fc2.bias"))?
            .clone();

        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }

    /// Forward pass: fc1 -> silu -> fc2
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let hidden = self.linear(x, &self.fc1_weight, Some(&self.fc1_bias))?;
        let hidden = candle_nn::ops::silu(&hidden)?;
        self.linear(&hidden, &self.fc2_weight, Some(&self.fc2_bias))
    }

    fn linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() == 3 {
            let (batch, seq, _features) = (dims[0], dims[1], dims[2]);
            let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
            let out_2d = x_2d.matmul(&weight.t()?)?;
            let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
            match bias {
                Some(b) => Ok(out_3d.broadcast_add(b)?),
                None => Ok(out_3d),
            }
        } else {
            let out = x.matmul(&weight.t()?)?;
            match bias {
                Some(b) => Ok(out.broadcast_add(b)?),
                None => Ok(out),
            }
        }
    }
}

/// TalkerModel for autoregressive semantic token generation
pub struct TalkerModel {
    /// Text embedding [text_vocab_size, text_embed_dim]
    text_embedding: Tensor,
    /// Text projection (2048 -> 1024)
    text_projection: TextProjection,
    /// Codec embedding [codec_vocab_size, hidden_size]
    codec_embedding: Tensor,
    /// Transformer decoder layers
    layers: Vec<TalkerDecoderLayer>,
    /// Final RMS norm weight
    norm_weight: Tensor,
    /// Codec head [codec_vocab_size, hidden_size]
    codec_head: Tensor,
    /// Rotary position embedding
    rope: RotaryEmbedding,
    /// Configuration
    config: TalkerConfig,
    /// Device
    device: Device,
}

/// Simplified decoder layer that works directly with weight tensors
/// (avoids the VarBuilder complexity)
pub struct TalkerDecoderLayer {
    // Input LayerNorm
    input_ln_weight: Tensor,
    // Self-attention
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    q_norm_weight: Tensor,
    k_norm_weight: Tensor,
    // Post-attention LayerNorm
    post_ln_weight: Tensor,
    // MLP
    gate_proj_weight: Tensor,
    up_proj_weight: Tensor,
    down_proj_weight: Tensor,
    // Config values
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rms_norm_eps: f64,
}

impl TalkerDecoderLayer {
    /// Load layer from weights
    pub fn from_weights(
        weights: &HashMap<String, Tensor>,
        layer_idx: usize,
        config: &TalkerConfig,
    ) -> Result<Self> {
        let prefix = format!("talker.model.layers.{}", layer_idx);

        let input_ln_weight = weights
            .get(&format!("{}.input_layernorm.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.input_layernorm.weight", prefix))?
            .clone();
        let q_proj_weight = weights
            .get(&format!("{}.self_attn.q_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.q_proj.weight", prefix))?
            .clone();
        let k_proj_weight = weights
            .get(&format!("{}.self_attn.k_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.k_proj.weight", prefix))?
            .clone();
        let v_proj_weight = weights
            .get(&format!("{}.self_attn.v_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.v_proj.weight", prefix))?
            .clone();
        let o_proj_weight = weights
            .get(&format!("{}.self_attn.o_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.o_proj.weight", prefix))?
            .clone();
        let q_norm_weight = weights
            .get(&format!("{}.self_attn.q_norm.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.q_norm.weight", prefix))?
            .clone();
        let k_norm_weight = weights
            .get(&format!("{}.self_attn.k_norm.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.self_attn.k_norm.weight", prefix))?
            .clone();
        let post_ln_weight = weights
            .get(&format!("{}.post_attention_layernorm.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.post_attention_layernorm.weight", prefix))?
            .clone();
        let gate_proj_weight = weights
            .get(&format!("{}.mlp.gate_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.mlp.gate_proj.weight", prefix))?
            .clone();
        let up_proj_weight = weights
            .get(&format!("{}.mlp.up_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.mlp.up_proj.weight", prefix))?
            .clone();
        let down_proj_weight = weights
            .get(&format!("{}.mlp.down_proj.weight", prefix))
            .ok_or_else(|| anyhow::anyhow!("Missing {}.mlp.down_proj.weight", prefix))?
            .clone();

        Ok(Self {
            input_ln_weight,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            o_proj_weight,
            q_norm_weight,
            k_norm_weight,
            post_ln_weight,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            rms_norm_eps: config.rms_norm_eps,
        })
    }

    /// Forward pass with optional KV cache
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        rope: &RotaryEmbedding,
        attention_mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        offset: usize,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Input layer norm
        let normed = self.rms_norm(hidden_states, &self.input_ln_weight)?;

        // QKV projections
        let q = self.linear(&normed, &self.q_proj_weight, None)?;
        let k = self.linear(&normed, &self.k_proj_weight, None)?;
        let v = self.linear(&normed, &self.v_proj_weight, None)?;

        // Reshape for multi-head attention
        let q = q.reshape((batch, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?;

        // QK normalization
        let q = self.rms_norm(&q, &self.q_norm_weight)?;
        let k = self.rms_norm(&k, &self.k_norm_weight)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // Apply RoPE
        let (q, k) = rope.apply(&q, &k, offset)?;

        // Update KV cache
        let (k, v) = if let Some(cache) = kv_cache {
            let k = cache.update_k(&k)?;
            let v = cache.update_v(&v)?;
            (k, v)
        } else {
            (k, v)
        };

        // Repeat KV for GQA
        let k = self.repeat_kv(&k)?;
        let v = self.repeat_kv(&v)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).powf(-0.5);
        let attn_weights = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;

        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&v)?;

        // Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, hidden]
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        // O projection
        let attn_output = self.linear(&attn_output, &self.o_proj_weight, None)?;

        // Residual
        let hidden_states = (hidden_states + attn_output)?;

        // MLP
        let normed = self.rms_norm(&hidden_states, &self.post_ln_weight)?;
        let gate = self.linear(&normed, &self.gate_proj_weight, None)?;
        let up = self.linear(&normed, &self.up_proj_weight, None)?;
        let mlp_output = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_output = self.linear(&mlp_output, &self.down_proj_weight, None)?;

        // Final residual
        Ok((hidden_states + mlp_output)?)
    }

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + self.rms_norm_eps)?.sqrt()?)?;
        Ok(x_norm.broadcast_mul(weight)?)
    }

    fn linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() == 3 {
            let (batch, seq, _features) = (dims[0], dims[1], dims[2]);
            let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
            let out_2d = x_2d.matmul(&weight.t()?)?;
            let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
            match bias {
                Some(b) => Ok(out_3d.broadcast_add(b)?),
                None => Ok(out_3d),
            }
        } else {
            let out = x.matmul(&weight.t()?)?;
            match bias {
                Some(b) => Ok(out.broadcast_add(b)?),
                None => Ok(out),
            }
        }
    }

    fn repeat_kv(&self, x: &Tensor) -> Result<Tensor> {
        let n_rep = self.num_heads / self.num_kv_heads;
        if n_rep == 1 {
            return Ok(x.clone());
        }
        let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
        let x = x
            .unsqueeze(2)?
            .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?;
        Ok(x)
    }
}

impl TalkerModel {
    /// Load model from weight tensors
    pub fn from_weights(weights: &HashMap<String, Tensor>, device: &Device) -> Result<Self> {
        let config = TalkerConfig::default();
        Self::from_weights_with_config(weights, config, device)
    }

    /// Load model with custom config
    pub fn from_weights_with_config(
        weights: &HashMap<String, Tensor>,
        config: TalkerConfig,
        device: &Device,
    ) -> Result<Self> {
        // Text embedding
        let text_embedding = weights
            .get("talker.model.text_embedding.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.text_embedding.weight"))?
            .clone();

        // Text projection
        let text_projection = TextProjection::from_weights(weights)?;

        // Codec embedding
        let codec_embedding = weights
            .get("talker.model.codec_embedding.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.codec_embedding.weight"))?
            .clone();

        // Decoder layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(TalkerDecoderLayer::from_weights(weights, i, &config)?);
        }

        // Final norm
        let norm_weight = weights
            .get("talker.model.norm.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.model.norm.weight"))?
            .clone();

        // Codec head
        let codec_head = weights
            .get("talker.codec_head.weight")
            .ok_or_else(|| anyhow::anyhow!("Missing talker.codec_head.weight"))?
            .clone();

        // RoPE
        let rope = RotaryEmbedding::new(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            device,
        )?;

        Ok(Self {
            text_embedding,
            text_projection,
            codec_embedding,
            layers,
            norm_weight,
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
        let text_embed = self.text_embedding.index_select(&input_ids_flat, 0)?;
        let text_embed = text_embed.reshape((1, seq_len, self.config.text_embed_dim))?;

        // Project to hidden dimension
        let mut hidden = self.text_projection.forward(&text_embed)?;

        // Create causal mask
        let mask = self.create_causal_mask(seq_len, 0)?;

        // Run through all layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden, &self.rope, Some(&mask), Some(&mut kv_caches[i]), 0)?;
        }

        // Final norm
        hidden = self.rms_norm(&hidden, &self.norm_weight)?;

        // Get logits for last position
        let last_hidden = hidden.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.linear(&last_hidden, &self.codec_head, None)?;

        Ok((hidden, logits))
    }

    /// Generate next token given previous codec token
    ///
    /// Uses KV cache for efficient autoregressive generation.
    pub fn generate_step(
        &self,
        prev_token: u32,
        kv_caches: &mut [KVCache],
        offset: usize,
    ) -> Result<(Tensor, Tensor)> {
        // Embed previous codec token
        let token_tensor = Tensor::new(&[prev_token], &self.device)?;
        let token_embed = self.codec_embedding.index_select(&token_tensor, 0)?;
        let mut hidden = token_embed.unsqueeze(0)?; // [1, 1, hidden_size]

        // Create causal mask for single token (attends to all previous positions)
        let mask = self.create_causal_mask(1, offset)?;

        // Run through all layers with KV cache
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
        hidden = self.rms_norm(&hidden, &self.norm_weight)?;

        // Get logits
        let logits = self.linear(&hidden, &self.codec_head, None)?;

        Ok((hidden, logits))
    }

    /// Full forward pass without KV caching (for testing)
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;

        // Embed text tokens
        let input_ids_flat = input_ids.flatten_all()?;
        let text_embed = self.text_embedding.index_select(&input_ids_flat, 0)?;
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
        hidden = self.rms_norm(&hidden, &self.norm_weight)?;

        // Get logits for all positions
        let logits = self.linear(&hidden, &self.codec_head, None)?;

        Ok(logits)
    }

    /// Autoregressive generation
    pub fn generate(
        &self,
        input_ids: &Tensor,
        config: &GenerationConfig,
    ) -> Result<Vec<u32>> {
        let mut kv_caches: Vec<KVCache> = (0..self.config.num_hidden_layers)
            .map(|_| KVCache::new())
            .collect();

        // Prefill with text
        let (_hidden, logits) = self.prefill(input_ids, &mut kv_caches)?;
        let mut offset = input_ids.dim(1)?;

        // Sample first token
        let first_token = crate::generation::sample(&logits.squeeze(1)?, config)?;
        let first_token_id: u32 = first_token.flatten_all()?.to_vec1::<u32>()?[0];

        let mut generated = vec![first_token_id];

        // Check for EOS
        if let Some(eos_id) = config.eos_token_id {
            if first_token_id == eos_id {
                return Ok(generated);
            }
        }

        // Generate remaining tokens
        for _ in 1..config.max_new_tokens {
            let prev_token = *generated.last().unwrap();
            let (_hidden, logits) = self.generate_step(prev_token, &mut kv_caches, offset)?;
            offset += 1;

            // Sample next token
            let next_token = crate::generation::sample(&logits.squeeze(1)?, config)?;
            let next_token_id: u32 = next_token.flatten_all()?.to_vec1::<u32>()?[0];

            generated.push(next_token_id);

            // Check for EOS
            if let Some(eos_id) = config.eos_token_id {
                if next_token_id == eos_id {
                    break;
                }
            }
        }

        Ok(generated)
    }

    /// Get the hidden state at last position (for code predictor input)
    pub fn get_last_hidden(
        &self,
        input_ids: &Tensor,
        kv_caches: &mut [KVCache],
    ) -> Result<Tensor> {
        let (hidden, _logits) = self.prefill(input_ids, kv_caches)?;
        let seq_len = hidden.dim(1)?;
        Ok(hidden.i((.., seq_len - 1..seq_len, ..))?)
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

    fn rms_norm(&self, x: &Tensor, weight: &Tensor) -> Result<Tensor> {
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_norm = x.broadcast_div(&(variance + self.config.rms_norm_eps)?.sqrt()?)?;
        Ok(x_norm.broadcast_mul(weight)?)
    }

    fn linear(&self, x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
        let dims = x.dims();
        if dims.len() == 3 {
            let (batch, seq, _features) = (dims[0], dims[1], dims[2]);
            let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
            let out_2d = x_2d.matmul(&weight.t()?)?;
            let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
            match bias {
                Some(b) => Ok(out_3d.broadcast_add(b)?),
                None => Ok(out_3d),
            }
        } else {
            let out = x.matmul(&weight.t()?)?;
            match bias {
                Some(b) => Ok(out.broadcast_add(b)?),
                None => Ok(out),
            }
        }
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
        let embed = self.codec_embedding.index_select(&token_tensor, 0)?;
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

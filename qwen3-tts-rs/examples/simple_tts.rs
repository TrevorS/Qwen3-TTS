// Simple TTS generation matching Python reference exactly
use candle_core::{DType, Device, IndexOp, Tensor, D};
use std::collections::HashMap;
use std::path::Path;

fn main() -> anyhow::Result<()> {
    let device = Device::Cpu;
    let model_path = Path::new("test_data/model/model.safetensors");
    let weights: HashMap<String, Tensor> = candle_core::safetensors::load(&model_path, &device)?;
    let weights: HashMap<String, Tensor> = weights
        .into_iter()
        .map(|(k, v)| {
            let v = if v.dtype() == DType::BF16 {
                v.to_dtype(DType::F32).unwrap()
            } else {
                v
            };
            (k, v)
        })
        .collect();

    let decoder_path = Path::new("test_data/speech_tokenizer/model.safetensors");
    let decoder_weights: HashMap<String, Tensor> = candle_core::safetensors::load(&decoder_path, &device)?;
    let decoder_weights: HashMap<String, Tensor> = decoder_weights
        .into_iter()
        .map(|(k, v)| {
            let v = if v.dtype() == DType::BF16 {
                v.to_dtype(DType::F32).unwrap()
            } else {
                v
            };
            (k, v)
        })
        .collect();

    println!("=== Simple TTS with Speaker Conditioning ===");

    // Speaker token IDs from CustomVoice model config
    const SPK_RYAN: u32 = 3061;  // English male
    const SPK_VIVIAN: u32 = 3065;  // Chinese female
    let speaker_id = SPK_RYAN;
    println!("Speaker: Ryan ({})", speaker_id);

    // Config
    let num_layers = 28;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let eps = 1e-6;
    let num_frames = 50;
    let seed = 42u64;
    let temperature = 0.7;

    // Set seed
    qwen3_tts::generation::set_seed(seed);
    qwen3_tts::generation::reset_rng();

    // Simple prefill - just "Hello" (9707)
    let text_embedding = weights.get("talker.model.text_embedding.weight").unwrap();
    let codec_embedding = weights.get("talker.model.codec_embedding.weight").unwrap();
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight").unwrap();
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias").unwrap();
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight").unwrap();
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias").unwrap();
    let codec_head = weights.get("talker.codec_head.weight").unwrap();

    // Get embedding for token 9707 ("Hello")
    let token_id = Tensor::new(&[9707u32], &device)?;
    let embed = text_embedding.index_select(&token_id, 0)?;
    let embed = embed.unsqueeze(0)?;

    // Text projection: fc1 -> silu -> fc2
    let h = linear_3d(&embed, fc1_w, Some(fc1_b))?;
    let h = candle_nn::ops::silu(&h)?;
    let text_hidden = linear_3d(&h, fc2_w, Some(fc2_b))?;  // [1, 1, 1024]

    // TEXT ONLY (no speaker) - matching Python reference
    let hidden = text_hidden.clone();
    let _speaker_id = speaker_id;  // silence unused warning

    println!("Prefill hidden shape: {:?} (text only, no speaker)", hidden.shape());

    // Run prefill through transformer
    let mut kv_caches: Vec<(Tensor, Tensor)> = Vec::new();
    let hidden = run_transformer(&hidden, &weights, num_layers, num_heads, num_kv_heads, head_dim, eps, 0, &mut kv_caches, &device)?;

    // Get first semantic token
    let last_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
    println!("Last hidden shape: {:?}, values[:5]: {:?}",
        last_hidden.shape(),
        last_hidden.squeeze(0)?.squeeze(0)?.narrow(0, 0, 5)?.to_vec1::<f32>()?);

    let logits = linear_3d(&last_hidden, codec_head, None)?;
    let logits = logits.squeeze(0)?.squeeze(0)?;

    // Debug: show top tokens before suppression
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut indexed: Vec<(usize, f32)> = logits_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("Top 5 tokens (before suppression): {:?}", &indexed[..5]);

    // Sample with temperature and suppression
    let first_token = sample_token(&logits, temperature, &device)?;
    println!("First semantic token: {}", first_token);

    // Collect all codes
    let mut all_codes: Vec<Vec<i64>> = Vec::new();
    let mut current_hidden = hidden.i((.., hidden.dim(1)? - 1.., ..))?;
    let mut offset = 1;  // single fused position

    for frame in 0..num_frames {
        // Sample semantic token
        let logits = linear_3d(&current_hidden, codec_head, None)?;
        let logits = logits.squeeze(0)?.squeeze(0)?;
        let semantic_token = if frame == 0 { first_token } else { sample_token(&logits, temperature, &device)? };

        // Run code predictor to get acoustic codes
        let semantic_embed = codec_embedding.i(semantic_token as usize)?.unsqueeze(0)?.unsqueeze(0)?;
        let acoustic_codes = run_code_predictor(&current_hidden, &semantic_embed, &weights, &device)?;

        // Collect frame codes
        let mut frame_codes = vec![semantic_token as i64];
        frame_codes.extend(acoustic_codes.iter().map(|&c| c as i64));

        if frame < 3 {
            println!("Frame {}: semantic={}, acoustics={:?}", frame, semantic_token, &acoustic_codes[..3]);
        }
        all_codes.push(frame_codes);

        // Next step: embed semantic token and continue
        let next_embed = codec_embedding.i(semantic_token as usize)?.unsqueeze(0)?.unsqueeze(0)?;
        current_hidden = run_transformer_step(&next_embed, &weights, num_layers, num_heads, num_kv_heads, head_dim, eps, offset, &mut kv_caches, &device)?;
        offset += 1;
    }

    println!("\nGenerated {} frames", all_codes.len());

    // Convert to tensor [1, 16, frames]
    let frames = all_codes.len();
    let mut codes_flat: Vec<i64> = vec![0; 16 * frames];
    for (f, frame) in all_codes.iter().enumerate() {
        for (c, &code) in frame.iter().enumerate() {
            codes_flat[c * frames + f] = code;
        }
    }
    let codes = Tensor::from_vec(codes_flat, (1, 16, frames), &device)?;
    println!("Codes shape: {:?}", codes.shape());

    // Decode to audio
    use qwen3_tts::models::codec::{Decoder12Hz, Decoder12HzConfig};
    let decoder = Decoder12Hz::from_weights(&decoder_weights, Decoder12HzConfig::default())?;
    let audio = decoder.decode(&codes)?;
    println!("Audio shape: {:?}", audio.shape());

    // Save to WAV
    let audio_data: Vec<f32> = audio.flatten_all()?.to_vec1()?;
    use qwen3_tts::AudioBuffer;
    let buffer = AudioBuffer::new(audio_data, 24000);
    buffer.save(Path::new("test_data/simple_tts.wav"))?;
    println!("Saved to test_data/simple_tts.wav");

    Ok(())
}

fn sample_token(logits: &Tensor, _temperature: f64, _device: &Device) -> anyhow::Result<u32> {
    // Apply token suppression (tokens 2048-3071 except EOS=2150)
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut suppressed = logits_vec.clone();
    for i in 2048..3072 {
        if i != 2150 {
            suppressed[i] = f32::NEG_INFINITY;
        }
    }

    // Greedy: find argmax
    let mut max_idx = 0;
    let mut max_val = f32::NEG_INFINITY;
    for (i, &v) in suppressed.iter().enumerate() {
        if v > max_val {
            max_val = v;
            max_idx = i;
        }
    }
    Ok(max_idx as u32)
}

fn run_code_predictor(
    last_hidden: &Tensor,
    semantic_embed: &Tensor,
    weights: &HashMap<String, Tensor>,
    device: &Device,
) -> anyhow::Result<Vec<u32>> {
    let num_layers = 5;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;  // Q=2048, K/V=1024
    let eps = 1e-6;

    // Input: concat [last_hidden, semantic_embed] along seq dim
    let hidden = Tensor::cat(&[last_hidden, semantic_embed], 1)?;
    let seq_len = hidden.dim(1)?;

    // Build RoPE
    let rope_theta = 1_000_000.0f32;
    let positions = Tensor::arange(0u32, seq_len as u32, device)?.to_dtype(DType::F32)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_vals, (head_dim / 2,), device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;

    // Causal mask
    let mask = create_causal_mask(seq_len, device)?;

    // Run through layers
    let mut hidden = hidden;
    for layer_idx in 0..num_layers {
        let prefix = format!("talker.code_predictor.model.layers.{}", layer_idx);

        let ln_w = weights.get(&format!("{}.input_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let q = linear_3d(&normed, weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap(), None)?;
        let k = linear_3d(&normed, weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap(), None)?;
        let v = linear_3d(&normed, weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap(), None)?;

        let q = q.reshape((1, seq_len, num_heads, head_dim))?;
        let k = k.reshape((1, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((1, seq_len, num_kv_heads, head_dim))?;

        let q = rms_norm(&q, weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap(), eps)?;
        let k = rms_norm(&k, weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap(), eps)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;

        let k = repeat_kv(&k, num_heads / num_kv_heads)?;
        let v = repeat_kv(&v, num_heads / num_kv_heads)?;

        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        let attn_out = attn_out.transpose(1, 2)?.reshape((1, seq_len, num_heads * head_dim))?;
        let attn_out = linear_3d(&attn_out, weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&attn_out)?;

        let ln_w = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let gate = linear_3d(&normed, weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap(), None)?;
        let up = linear_3d(&normed, weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap(), None)?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = linear_3d(&mlp_out, weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&mlp_out)?;
    }

    // Final norm
    let norm_w = weights.get("talker.code_predictor.model.norm.weight").unwrap();
    hidden = rms_norm(&hidden, norm_w, eps)?;

    // Get acoustic codes from position 1 (semantic embed position)
    let hidden_at_1 = hidden.i((.., 1.., ..))?;

    let mut codes = Vec::new();
    for i in 0..15 {
        let lm_head = weights.get(&format!("talker.code_predictor.lm_head.{}.weight", i)).unwrap();
        let logits = linear_3d(&hidden_at_1.i((.., 0..1, ..))?, lm_head, None)?;
        let token = logits.squeeze(0)?.squeeze(0)?.argmax(0)?.to_scalar::<u32>()?;
        codes.push(token);
    }

    Ok(codes)
}

fn run_transformer(
    input: &Tensor,
    weights: &HashMap<String, Tensor>,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    start_pos: usize,
    kv_caches: &mut Vec<(Tensor, Tensor)>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    let seq_len = input.dim(1)?;

    // Build RoPE
    let rope_theta = 1_000_000.0f32;
    let positions = Tensor::arange(start_pos as u32, (start_pos + seq_len) as u32, device)?.to_dtype(DType::F32)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_vals, (head_dim / 2,), device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;

    // Causal mask
    let mask = create_causal_mask(seq_len, device)?;

    let mut hidden = input.clone();
    kv_caches.clear();

    for layer_idx in 0..num_layers {
        let prefix = format!("talker.model.layers.{}", layer_idx);

        let ln_w = weights.get(&format!("{}.input_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let q = linear_3d(&normed, weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap(), None)?;
        let k = linear_3d(&normed, weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap(), None)?;
        let v = linear_3d(&normed, weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap(), None)?;

        let q = q.reshape((1, seq_len, num_heads, head_dim))?;
        let k = k.reshape((1, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((1, seq_len, num_kv_heads, head_dim))?;

        let q = rms_norm(&q, weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap(), eps)?;
        let k = rms_norm(&k, weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap(), eps)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;

        kv_caches.push((k.clone(), v.clone()));

        let k = repeat_kv(&k, num_heads / num_kv_heads)?;
        let v = repeat_kv(&v, num_heads / num_kv_heads)?;

        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let attn = attn.broadcast_add(&mask)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        let attn_out = attn_out.transpose(1, 2)?.reshape((1, seq_len, num_heads * head_dim))?;
        let attn_out = linear_3d(&attn_out, weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&attn_out)?;

        let ln_w = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let gate = linear_3d(&normed, weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap(), None)?;
        let up = linear_3d(&normed, weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap(), None)?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = linear_3d(&mlp_out, weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&mlp_out)?;
    }

    let norm_w = weights.get("talker.model.norm.weight").unwrap();
    Ok(rms_norm(&hidden, norm_w, eps)?)
}

fn run_transformer_step(
    input: &Tensor,
    weights: &HashMap<String, Tensor>,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    eps: f64,
    pos: usize,
    kv_caches: &mut Vec<(Tensor, Tensor)>,
    device: &Device,
) -> anyhow::Result<Tensor> {
    // Build RoPE for single position
    let rope_theta = 1_000_000.0f32;
    let position = Tensor::new(&[pos as f32], device)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_vals, (head_dim / 2,), device)?;
    let freqs = position.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;
    let sin = freqs.sin()?.repeat((1, 2))?.unsqueeze(0)?.unsqueeze(0)?;

    let mut hidden = input.clone();

    for layer_idx in 0..num_layers {
        let prefix = format!("talker.model.layers.{}", layer_idx);

        let ln_w = weights.get(&format!("{}.input_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let q = linear_3d(&normed, weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap(), None)?;
        let k = linear_3d(&normed, weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap(), None)?;
        let v = linear_3d(&normed, weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap(), None)?;

        let q = q.reshape((1, 1, num_heads, head_dim))?;
        let k = k.reshape((1, 1, num_kv_heads, head_dim))?;
        let v = v.reshape((1, 1, num_kv_heads, head_dim))?;

        let q = rms_norm(&q, weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap(), eps)?;
        let k = rms_norm(&k, weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap(), eps)?;

        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;

        // Update KV cache
        let (cached_k, cached_v) = &kv_caches[layer_idx];
        let k = Tensor::cat(&[cached_k, &k], 2)?;
        let v = Tensor::cat(&[cached_v, &v], 2)?;
        kv_caches[layer_idx] = (k.clone(), v.clone());

        let k = repeat_kv(&k, num_heads / num_kv_heads)?;
        let v = repeat_kv(&v, num_heads / num_kv_heads)?;

        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        let attn_out = attn_out.transpose(1, 2)?.reshape((1, 1, num_heads * head_dim))?;
        let attn_out = linear_3d(&attn_out, weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&attn_out)?;

        let ln_w = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let gate = linear_3d(&normed, weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap(), None)?;
        let up = linear_3d(&normed, weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap(), None)?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = linear_3d(&mlp_out, weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&mlp_out)?;
    }

    let norm_w = weights.get("talker.model.norm.weight").unwrap();
    Ok(rms_norm(&hidden, norm_w, eps)?)
}

fn create_causal_mask(seq_len: usize, device: &Device) -> anyhow::Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    Ok(Tensor::from_vec(mask, (1, 1, seq_len, seq_len), device)?)
}

fn rotate_half(x: &Tensor) -> anyhow::Result<Tensor> {
    let half = x.dim(D::Minus1)? / 2;
    let x1 = x.narrow(D::Minus1, 0, half)?;
    let x2 = x.narrow(D::Minus1, half, half)?;
    Ok(Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?)
}

fn repeat_kv(x: &Tensor, n_rep: usize) -> anyhow::Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    let x = x.unsqueeze(2)?;
    let x = x.expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?;
    Ok(x.reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))?)
}

fn linear_3d(x: &Tensor, weight: &Tensor, bias: Option<&Tensor>) -> anyhow::Result<Tensor> {
    let dims = x.dims();
    let (batch, seq, _) = (dims[0], dims[1], dims[2]);
    let x_2d = x.reshape((batch * seq, x.dim(2)?))?;
    let out_2d = x_2d.matmul(&weight.t()?)?;
    let out_3d = out_2d.reshape((batch, seq, out_2d.dim(1)?))?;
    match bias {
        Some(b) => Ok(out_3d.broadcast_add(b)?),
        None => Ok(out_3d),
    }
}

fn rms_norm(x: &Tensor, weight: &Tensor, eps: f64) -> anyhow::Result<Tensor> {
    let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
    let x_norm = x.broadcast_div(&(variance + eps)?.sqrt()?)?;
    Ok(x_norm.broadcast_mul(weight)?)
}

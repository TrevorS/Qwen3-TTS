// Test to match Python simple prefill and run full generation
use candle_core::{DType, Device, Tensor, D};
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

    println!("=== Simple Prefill Test (matching Python reference) ===");

    // Simple prefill - just "Hello" (9707) exactly like Python
    let text_embedding = weights.get("talker.model.text_embedding.weight").unwrap();
    let fc1_w = weights.get("talker.text_projection.linear_fc1.weight").unwrap();
    let fc1_b = weights.get("talker.text_projection.linear_fc1.bias").unwrap();
    let fc2_w = weights.get("talker.text_projection.linear_fc2.weight").unwrap();
    let fc2_b = weights.get("talker.text_projection.linear_fc2.bias").unwrap();

    // Get embedding for token 9707
    let token_id = Tensor::new(&[9707u32], &device)?;
    let embed = text_embedding.index_select(&token_id, 0)?;  // [1, 2048]
    let embed = embed.unsqueeze(0)?;  // [1, 1, 2048]

    // Text projection: fc1 -> silu -> fc2
    let h = linear_3d(&embed, fc1_w, Some(fc1_b))?;
    let h = candle_nn::ops::silu(&h)?;
    let hidden = linear_3d(&h, fc2_w, Some(fc2_b))?;

    let vals: Vec<f32> = hidden.squeeze(0)?.squeeze(0)?.narrow(0, 0, 5)?.to_vec1()?;
    println!("After text_projection: {:?}", vals);
    // Python: [-0.0402, 0.0252, 0.0136, 0.0129, -0.0624]

    // Run through all 28 transformer layers
    println!("\nRunning through 28 transformer layers...");
    let mut hidden = hidden;
    let num_layers = 28;
    let num_heads = 16;
    let num_kv_heads = 8;
    let head_dim = 128;
    let eps = 1e-6;

    // Build RoPE for seq_len=1
    let seq_len = 1;
    let rope_theta = 1_000_000.0f32;
    let positions = Tensor::arange(0u32, seq_len as u32, &device)?.to_dtype(DType::F32)?;
    let inv_freq_vals: Vec<f32> = (0..head_dim)
        .step_by(2)
        .map(|i| 1.0 / rope_theta.powf(i as f32 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::from_vec(inv_freq_vals, (head_dim / 2,), &device)?;
    let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
    let cos = freqs.cos()?.repeat((1, 2))?;
    let sin = freqs.sin()?.repeat((1, 2))?;
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, 1, 128]
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    for layer_idx in 0..num_layers {
        let prefix = format!("talker.model.layers.{}", layer_idx);

        // Input LayerNorm
        let ln_w = weights.get(&format!("{}.input_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        // QKV projections
        let q = linear_3d(&normed, weights.get(&format!("{}.self_attn.q_proj.weight", prefix)).unwrap(), None)?;
        let k = linear_3d(&normed, weights.get(&format!("{}.self_attn.k_proj.weight", prefix)).unwrap(), None)?;
        let v = linear_3d(&normed, weights.get(&format!("{}.self_attn.v_proj.weight", prefix)).unwrap(), None)?;

        // Reshape
        let q = q.reshape((1, seq_len, num_heads, head_dim))?;
        let k = k.reshape((1, seq_len, num_kv_heads, head_dim))?;
        let v = v.reshape((1, seq_len, num_kv_heads, head_dim))?;

        // QK norm
        let q = rms_norm(&q, weights.get(&format!("{}.self_attn.q_norm.weight", prefix)).unwrap(), eps)?;
        let k = rms_norm(&k, weights.get(&format!("{}.self_attn.k_norm.weight", prefix)).unwrap(), eps)?;

        // Transpose to [batch, heads, seq, dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // RoPE
        let q = (q.broadcast_mul(&cos)? + rotate_half(&q)?.broadcast_mul(&sin)?)?;
        let k = (k.broadcast_mul(&cos)? + rotate_half(&k)?.broadcast_mul(&sin)?)?;

        // Repeat KV for GQA
        let k = repeat_kv(&k, num_heads / num_kv_heads)?;
        let v = repeat_kv(&v, num_heads / num_kv_heads)?;

        // Attention (no mask needed for single token)
        let scale = (head_dim as f64).powf(-0.5);
        let attn = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?.affine(scale, 0.0)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let attn_out = attn.matmul(&v)?;

        // Reshape back
        let attn_out = attn_out.transpose(1, 2)?.reshape((1, seq_len, num_heads * head_dim))?;

        // O projection
        let attn_out = linear_3d(&attn_out, weights.get(&format!("{}.self_attn.o_proj.weight", prefix)).unwrap(), None)?;

        // Residual
        hidden = hidden.add(&attn_out)?;

        // MLP
        let ln_w = weights.get(&format!("{}.post_attention_layernorm.weight", prefix)).unwrap();
        let normed = rms_norm(&hidden, ln_w, eps)?;

        let gate = linear_3d(&normed, weights.get(&format!("{}.mlp.gate_proj.weight", prefix)).unwrap(), None)?;
        let up = linear_3d(&normed, weights.get(&format!("{}.mlp.up_proj.weight", prefix)).unwrap(), None)?;
        let mlp_out = candle_nn::ops::silu(&gate)?.mul(&up)?;
        let mlp_out = linear_3d(&mlp_out, weights.get(&format!("{}.mlp.down_proj.weight", prefix)).unwrap(), None)?;

        hidden = hidden.add(&mlp_out)?;
    }

    // Final norm
    let norm_w = weights.get("talker.model.norm.weight").unwrap();
    hidden = rms_norm(&hidden, norm_w, eps)?;

    println!("After all layers, hidden[:5]: {:?}",
        hidden.squeeze(0)?.squeeze(0)?.narrow(0, 0, 5)?.to_vec1::<f32>()?);

    // Get logits
    let codec_head = weights.get("talker.codec_head.weight").unwrap();
    let logits = linear_3d(&hidden, codec_head, None)?;
    let logits = logits.squeeze(0)?.squeeze(0)?; // [3072]

    // Apply token suppression (tokens 2048-3071 except EOS=2150)
    let logits_vec: Vec<f32> = logits.to_vec1()?;
    let mut suppressed = logits_vec.clone();
    for i in 2048..3072 {
        if i != 2150 {
            suppressed[i] = f32::NEG_INFINITY;
        }
    }
    let logits = Tensor::from_vec(suppressed, (3072,), &device)?;

    // Get argmax
    let token = logits.argmax(0)?.to_scalar::<u32>()?;
    println!("\nFirst semantic token (greedy): {}", token);
    println!("Python expects: 1501");

    // Show top-5 tokens
    let logits_sorted = logits.to_vec1::<f32>()?;
    let mut indexed: Vec<(usize, f32)> = logits_sorted.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("\nTop 5 tokens:");
    for (i, (idx, val)) in indexed.iter().take(5).enumerate() {
        println!("  {}: token {} = {:.4}", i, idx, val);
    }

    Ok(())
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

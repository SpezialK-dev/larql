//! Attention computation — GQA, RoPE, causal masking.

use ndarray::Array2;

/// Apply Rotary Position Embeddings to Q or K.
/// Uses split-half pairing: (x[i], x[i + half_dim]).
/// This matches MLX traditional=False and HuggingFace's default.
/// x: (seq_len, num_heads * head_dim)
pub fn apply_rope(
    x: &Array2<f32>,
    num_heads: usize,
    head_dim: usize,
    rope_base: f64,
) -> Array2<f32> {
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    let half_dim = head_dim / 2;
    let inv_freq: Vec<f64> = (0..half_dim)
        .map(|i| 1.0 / rope_base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    for pos in 0..seq_len {
        for h in 0..num_heads {
            let offset = h * head_dim;
            for i in 0..half_dim {
                let theta = pos as f64 * inv_freq[i];
                let cos_t = theta.cos() as f32;
                let sin_t = theta.sin() as f32;

                let x0 = x[[pos, offset + i]];
                let x1 = x[[pos, offset + half_dim + i]];

                out[[pos, offset + i]] = x0 * cos_t - x1 * sin_t;
                out[[pos, offset + half_dim + i]] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
    out
}

/// Per-head attention weights for the last token position.
/// `weights[head]` = vec of attention scores over all positions.
pub struct AttentionWeights {
    /// Per-head attention distribution for the last sequence position.
    /// `heads[h][j]` = attention weight from last token to position j.
    pub heads: Vec<Vec<f32>>,
}

/// Grouped-query attention with causal masking.
///
/// q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    let (out, _) = gqa_attention_with_weights(q, k, v, num_q, head_dim, reps, scale, seq_len, false, None);
    out
}

/// GQA attention that optionally captures per-head attention weights for the last token.
/// `softcap`: if Some(cap), apply tanh(scores/cap)*cap before softmax (Gemma2).
///
/// BLAS-fused attention: uses `gemv` (matrix-vector multiply via Accelerate/AMX)
/// for the Q·K dot products and softmax·V accumulation, but never allocates the
/// full [seq, seq] attention matrix. Per query position `qi`:
///   1. `scores = K[0..=qi] @ Q[qi]` — one BLAS gemv
///   2. scale + softcap + two-pass softmax on the scores vector
///   3. `output = V[0..=qi]^T @ softmax_scores` — one BLAS gemv
///
/// Memory: O(seq) temporary per position, vs O(seq²) for the materialized path.
/// At seq=6 this is negligible; at seq=512+ the savings are significant.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_with_weights(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
    capture: bool,
    softcap: Option<f32>,
) -> (Array2<f32>, Option<AttentionWeights>) {
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));
    let mut captured_heads: Vec<Vec<f32>> = if capture {
        Vec::with_capacity(num_q)
    } else {
        Vec::new()
    };

    let scale_f32 = scale as f32;
    let last_pos = seq_len - 1;

    // Reusable buffer for softmax scores (avoids per-position allocation)
    let mut scores_buf = vec![0.0f32; seq_len];

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        for qi in 0..seq_len {
            let causal_len = qi + 1; // positions 0..=qi

            // ── BLAS gemv: compute all causal scores at once ──
            // scores[0..=qi] = K[0..=qi, kv_off..kv_off+hd] @ Q[qi, q_off..q_off+hd]
            let q_row = q.slice(ndarray::s![qi, q_off..q_off + head_dim]);
            let k_block = k.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let raw_scores = k_block.dot(&q_row); // [causal_len] via BLAS gemv

            // ── Scale + softcap ──
            for i in 0..causal_len {
                let mut s = raw_scores[i] * scale_f32;
                if let Some(cap) = softcap {
                    s = (s / cap).tanh() * cap;
                }
                scores_buf[i] = s;
            }

            // ── Two-pass softmax with f64 accumulation ──
            let max_val = scores_buf[..causal_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f64;
            for i in 0..causal_len {
                let e = ((scores_buf[i] - max_val) as f64).exp();
                scores_buf[i] = e as f32;
                sum += e;
            }
            let inv_sum = (1.0 / sum) as f32;
            for i in 0..causal_len {
                scores_buf[i] *= inv_sum;
            }

            // ── Capture last-token attention weights ──
            if capture && qi == last_pos {
                let mut captured = vec![0.0f32; seq_len];
                captured[..causal_len].copy_from_slice(&scores_buf[..causal_len]);
                captured_heads.push(captured);
            }

            // ── BLAS gemv: weighted V accumulation ──
            // output[qi] = V[0..=qi, kv_off..kv_off+hd]^T @ softmax_scores[0..=qi]
            let v_block = v.slice(ndarray::s![0..causal_len, kv_off..kv_off + head_dim]);
            let scores_view = ndarray::ArrayView1::from(&scores_buf[..causal_len]);
            let weighted_v = v_block.t().dot(&scores_view); // [head_dim] via BLAS gemv

            // Write output
            for d in 0..head_dim {
                out[[qi, q_off + d]] = weighted_v[d];
            }
        }
    }

    let weights = if capture {
        Some(AttentionWeights {
            heads: captured_heads,
        })
    } else {
        None
    };

    (out, weights)
}

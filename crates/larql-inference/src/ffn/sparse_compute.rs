//! Shared sparse FFN computation — architecture-correct.
//!
//! Given a set of pre-selected feature indices, computes the FFN output using
//! the model architecture trait for activation function, gating, and bias.
//! Backends only need to provide the feature selection; this module handles
//! the gather, activation, and down projection.
//!
//! Supports gated (Gemma/Llama/Mistral) and non-gated (StarCoder2) models,
//! SiLU and GELU activations, and optional up/down bias.

use ndarray::Array2;

use crate::forward::add_bias;
use crate::model::ModelWeights;
use super::{sigmoid, gelu_tanh};
use super::weight::dense_ffn_forward;

/// Compute FFN output for a pre-selected set of features.
///
/// Architecture-correct: reads ffn_type, activation, and bias from the model.
/// Falls back to dense (via `weight::dense_ffn_forward`) when K >= 80%.
pub fn sparse_ffn_forward(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            Array2::<f32>::zeros((seq_len, intermediate)),
        );
    }

    // Fall back to dense when most features are selected
    if k * 5 >= intermediate * 4 {
        return dense_ffn_forward(weights, layer, x);
    }

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gather up rows (always needed)
    let up_buf = gather_rows(w_up, features, hidden);
    let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

    // Gather gate rows (only for gated models)
    let _gate_buf;
    let gate_sub = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        _gate_buf = gather_rows(w_gate, features, hidden);
        Some(ndarray::ArrayView2::from_shape((k, hidden), &_gate_buf).unwrap())
    } else {
        _gate_buf = Vec::new();
        None
    };

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        if let Some(ref gate_sub) = gate_sub {
            // Gated: activation(gate) * up
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let activated = if use_gelu { gelu_tanh(g) } else { g * sigmoid(g) };
                full_activation[[s, feat]] = activated * up_proj[i];
            }
        } else {
            // Non-gated: activation(up + bias)
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();

            // Apply sparse up bias
            if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|bk| weights.vectors.get(&bk)) {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() {
                        vals[i] += bias[feat];
                    }
                }
            }

            for (i, &feat) in features.iter().enumerate() {
                let v = vals[i];
                full_activation[[s, feat]] = if use_gelu { gelu_tanh(v) } else { v * sigmoid(v) };
            }
        }

        // Down projection via BLAS on sparse activation
        let act_row = full_activation.row(s);
        let out_vec = w_down.dot(&act_row);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o = v);
    }

    // Apply down bias
    if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut out, bias);
    }

    (out, full_activation)
}

/// Sparse FFN with down vector overrides.
///
/// Like sparse_ffn_forward, but for features with overrides, the custom down vector
/// is used instead of the model's down weight row. This enables training-free INSERT:
/// the gate selects the feature, the override controls what it outputs.
pub fn sparse_ffn_forward_with_overrides(
    weights: &ModelWeights,
    layer: usize,
    x: &Array2<f32>,
    features: &[usize],
    overrides: &[(usize, &[f32])], // (feature_index, custom_down_vector)
) -> (Array2<f32>, Array2<f32>) {
    let arch = &*weights.arch;
    let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
    let w_down = weights.tensors.get(&arch.ffn_down_key(layer)).unwrap();
    let hidden = x.shape()[1];
    let intermediate = w_up.shape()[0];
    let seq_len = x.shape()[0];
    let k = features.len();

    if k == 0 {
        return (
            Array2::<f32>::zeros((seq_len, hidden)),
            Array2::<f32>::zeros((seq_len, intermediate)),
        );
    }

    // Build override lookup
    let override_map: std::collections::HashMap<usize, &[f32]> =
        overrides.iter().copied().collect();

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    let up_buf = gather_rows(w_up, features, hidden);
    let up_sub = ndarray::ArrayView2::from_shape((k, hidden), &up_buf).unwrap();

    let _gate_buf;
    let gate_sub = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        _gate_buf = gather_rows(w_gate, features, hidden);
        Some(ndarray::ArrayView2::from_shape((k, hidden), &_gate_buf).unwrap())
    } else {
        _gate_buf = Vec::new();
        None
    };

    let mut full_activation = Array2::<f32>::zeros((seq_len, intermediate));
    let mut out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = x.row(s);

        if let Some(ref gate_sub) = gate_sub {
            let gate_proj = gate_sub.dot(&x_row);
            let up_proj = up_sub.dot(&x_row);
            for (i, &feat) in features.iter().enumerate() {
                let g = gate_proj[i];
                let activated = if use_gelu { gelu_tanh(g) } else { g * sigmoid(g) };
                full_activation[[s, feat]] = activated * up_proj[i];
            }
        } else {
            let up_proj = up_sub.dot(&x_row);
            let mut vals = up_proj.to_vec();
            if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|bk| weights.vectors.get(&bk)) {
                for (i, &feat) in features.iter().enumerate() {
                    if feat < bias.len() { vals[i] += bias[feat]; }
                }
            }
            for (i, &feat) in features.iter().enumerate() {
                let v = vals[i];
                full_activation[[s, feat]] = if use_gelu { gelu_tanh(v) } else { v * sigmoid(v) };
            }
        }

        // Down projection: use override vectors where available
        // Standard features use w_down matmul, overridden features add directly
        let act_row = full_activation.row(s);

        // Start with standard down projection
        let out_vec = w_down.dot(&act_row);
        let mut out_row = out.row_mut(s);
        ndarray::Zip::from(&mut out_row).and(&out_vec).for_each(|o, &v| *o = v);

        // Apply overrides: subtract the standard contribution, add the override
        for &feat in features.iter() {
            if let Some(override_down) = override_map.get(&feat) {
                let activation = act_row[feat];
                if activation.abs() > 1e-8 && override_down.len() == hidden {
                    // Subtract standard contribution: w_down[:, feat] * activation
                    let down_col = w_down.column(feat);
                    for j in 0..hidden {
                        out_row[j] -= down_col[j] * activation;
                        out_row[j] += override_down[j] * activation;
                    }
                }
            }
        }
    }

    if let Some(bias) = arch.ffn_down_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut out, bias);
    }

    (out, full_activation)
}

/// Gather rows from a weight matrix for selected features.
fn gather_rows(
    w: &ndarray::ArrayBase<impl ndarray::Data<Elem = f32>, ndarray::Ix2>,
    features: &[usize],
    hidden: usize,
) -> Vec<f32> {
    let k = features.len();
    let raw = w.as_slice().unwrap();
    let mut buf = vec![0.0f32; k * hidden];
    for (i, &feat) in features.iter().enumerate() {
        let src = feat * hidden;
        buf[i * hidden..(i + 1) * hidden].copy_from_slice(&raw[src..src + hidden]);
    }
    buf
}

/// Select top-K features by gate activation magnitude (architecture-correct).
///
/// For gated models: ranks by |activation(gate_proj)|.
/// For non-gated models: ranks by |activation(up_proj + bias)|.
pub fn select_top_k_features(
    weights: &ModelWeights,
    layer: usize,
    x_row: &ndarray::ArrayView1<f32>,
    top_k: usize,
) -> Vec<usize> {
    let arch = &*weights.arch;
    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // For gated models, rank by gate activation.
    // For non-gated models, rank by up activation (since there's no gate).
    let proj = if is_gated {
        let w_gate = weights.tensors.get(&arch.ffn_gate_key(layer)).unwrap();
        w_gate.dot(x_row)
    } else {
        let w_up = weights.tensors.get(&arch.ffn_up_key(layer)).unwrap();
        let mut p = w_up.dot(x_row);
        // Apply up bias for non-gated
        if let Some(bias) = arch.ffn_up_bias_key(layer).and_then(|bk| weights.vectors.get(&bk)) {
            for i in 0..p.len().min(bias.len()) {
                p[i] += bias[i];
            }
        }
        p
    };

    let intermediate = proj.len();
    let k = top_k.min(intermediate);

    let mut indexed: Vec<(usize, f32)> = proj
        .iter()
        .copied()
        .enumerate()
        .map(|(i, v)| {
            let act = if use_gelu { gelu_tanh(v) } else { v * sigmoid(v) };
            (i, act)
        })
        .collect();

    if k > 0 && k < indexed.len() {
        indexed.select_nth_unstable_by(k, |a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
        indexed.truncate(k);
    }
    indexed.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    indexed.into_iter().map(|(id, _)| id).collect()
}

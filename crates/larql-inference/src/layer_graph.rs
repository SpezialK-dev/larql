//! LayerGraph — pluggable per-layer routing for attention and FFN.
//!
//! The transformer layer loop receives a residual, routes through attention
//! and FFN, and produces the next residual. The mechanism behind each step
//! can vary:
//!
//! - Dense matmul (today's baseline)
//! - Walk/vindex (sparse FFN from mmap)
//! - Template cache (precomputed routing for known templates)
//! - Residual-adaptive graph (cluster-based routing)
//!
//! The `LayerGraph` trait abstracts this: given a residual, produce the
//! layer output. The implementation decides how attention and FFN are computed.

use ndarray::Array2;

use crate::attention::AttentionWeights;
use larql_compute::ComputeBackend;
use crate::ffn::FfnBackend;
use crate::model::ModelWeights;

/// Output of a single layer's computation.
pub struct LayerOutput {
    /// Post-layer residual (input to next layer).
    pub residual: Array2<f32>,
    /// Optional: FFN activation capture (for tracing/analysis).
    pub activation: Option<Array2<f32>>,
    /// Optional: attention weight capture (for tracing/analysis).
    pub attention: Option<AttentionWeights>,
}

/// Per-layer routing trait. Takes a residual, produces the next residual.
///
/// Implementations control both attention and FFN computation.
/// The residual is always the input. The mechanism changes.
pub trait LayerGraph {
    /// Run one transformer layer: attention + FFN + residuals.
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput>;

    /// Human-readable name for logging.
    fn name(&self) -> &str;
}

/// Dense baseline: standard matmul attention + pluggable FFN backend.
/// This is today's working path — nothing changes, just wrapped in the trait.
pub struct DenseLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
    pub capture_activation: bool,
    pub capture_attention: bool,
}

impl<'a> LayerGraph for DenseLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul (Q·K·V), optionally GPU-accelerated
        let (h_post_attn, _attn_proj, attn_weights) =
            crate::attention::run_attention_block_gpu(
                weights, h, layer, self.capture_attention, self.backend,
            )?;

        // FFN: delegated to backend (dense, walk, sparse, etc.)
        let (h_out, activation) = crate::forward::run_ffn(
            weights, &h_post_attn, layer, self.ffn, self.capture_activation,
        );

        Some(LayerOutput {
            residual: h_out,
            activation,
            attention: attn_weights,
        })
    }

    fn name(&self) -> &str {
        "dense"
    }
}

/// Per-layer graph selection: different layers can use different backends.
pub struct PerLayerGraph<'a> {
    layers: Vec<&'a dyn LayerGraph>,
}

impl<'a> PerLayerGraph<'a> {
    pub fn new(layers: Vec<&'a dyn LayerGraph>) -> Self {
        Self { layers }
    }

    pub fn get(&self, layer: usize) -> &'a dyn LayerGraph {
        if layer < self.layers.len() {
            self.layers[layer]
        } else {
            *self.layers.last().unwrap()
        }
    }
}

impl<'a> LayerGraph for PerLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        self.get(layer).forward_layer(weights, h, layer)
    }

    fn name(&self) -> &str {
        "per-layer"
    }
}

// ── Walk: dense attention + vindex walk FFN ──

/// Walk layer graph: dense attention + vindex walk FFN.
/// This is the working walk path, wrapped in the LayerGraph trait.
pub struct WalkLayerGraph<'a> {
    pub ffn: &'a dyn FfnBackend,
    pub backend: Option<&'a dyn ComputeBackend>,
}

impl<'a> LayerGraph for WalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, h, layer, false, self.backend)?;
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, self.ffn, false);
        Some(LayerOutput { residual: h_out, activation: None, attention: None })
    }

    fn name(&self) -> &str { "walk" }
}

// ── Cached: precomputed layer output for fixed-routing regimes ──

/// Cached layer graph: returns a precomputed residual instead of computing.
/// For layers where the output is template-determined (L0-12 regime).
///
/// Build by running a dense forward pass for a template, capturing residuals,
/// then storing them. At inference, skip the computation entirely.
pub struct CachedLayerGraph {
    /// layer → cached residual [seq_len, hidden]. Keyed by layer index.
    cache: std::collections::HashMap<usize, Array2<f32>>,
}

impl CachedLayerGraph {
    /// Build a cache by running a dense forward pass and capturing residuals.
    /// `layers`: which layers to cache (e.g., 0..=12).
    pub fn build(
        weights: &ModelWeights,
        token_ids: &[u32],
        layers: &[usize],
        ffn: &dyn FfnBackend,
    ) -> Self {
        let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
        let mut cache = std::collections::HashMap::new();
        let max_layer = *layers.iter().max().unwrap_or(&0);

        for layer in 0..=max_layer.min(weights.num_layers - 1) {
            let graph = DenseLayerGraph { ffn, backend: None, capture_activation: false, capture_attention: false };
            if let Some(output) = graph.forward_layer(weights, &h, layer) {
                h = output.residual;
                if layers.contains(&layer) {
                    cache.insert(layer, h.clone());
                }
            }
        }
        Self { cache }
    }

    /// Build from an existing residual (e.g., from a previous forward pass).
    pub fn from_residuals(residuals: Vec<(usize, Array2<f32>)>) -> Self {
        Self { cache: residuals.into_iter().collect() }
    }

    pub fn has_layer(&self, layer: usize) -> bool {
        self.cache.contains_key(&layer)
    }

    pub fn num_cached(&self) -> usize {
        self.cache.len()
    }
}

impl LayerGraph for CachedLayerGraph {
    fn forward_layer(
        &self,
        _weights: &ModelWeights,
        _h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        let residual = self.cache.get(&layer)?.clone();
        Some(LayerOutput { residual, activation: None, attention: None })
    }

    fn name(&self) -> &str { "cached" }
}

// ── Template detection ──

/// Known template patterns for routing.
#[derive(Clone, Debug)]
pub struct TemplatePattern {
    pub name: String,
    /// Token prefix that identifies this template (before the entity slot).
    pub prefix_tokens: Vec<u32>,
    /// Layer range for cached regime.
    pub cached_layers: std::ops::RangeInclusive<usize>,
}

/// Detect which template a token sequence matches, if any.
/// Matches by longest prefix overlap.
pub fn detect_template(token_ids: &[u32], templates: &[TemplatePattern]) -> Option<usize> {
    let mut best = None;
    let mut best_len = 0;

    for (i, tmpl) in templates.iter().enumerate() {
        let prefix = &tmpl.prefix_tokens;
        if prefix.len() > token_ids.len() { continue; }
        // Check if tokens start with this prefix (skipping BOS if present)
        let offset = if token_ids.len() > prefix.len() && token_ids[0] != prefix[0] { 1 } else { 0 };
        if offset + prefix.len() > token_ids.len() { continue; }
        let matches = prefix.iter().zip(&token_ids[offset..]).all(|(a, b)| a == b);
        if matches && prefix.len() > best_len {
            best = Some(i);
            best_len = prefix.len();
        }
    }
    best
}

/// Build a PerLayerGraph with cached layers for a detected template.
/// Returns the graph and the number of cached layers.
///
/// Layout:
///   cached_layers → CachedLayerGraph (skip computation)
///   remaining layers → fallback (dense/walk)
pub fn build_adaptive_graph<'a>(
    cache: &'a CachedLayerGraph,
    fallback: &'a dyn LayerGraph,
    num_layers: usize,
    cached_range: &std::ops::RangeInclusive<usize>,
) -> PerLayerGraph<'a> {
    let mut layers: Vec<&dyn LayerGraph> = Vec::with_capacity(num_layers);
    for layer in 0..num_layers {
        if cached_range.contains(&layer) && cache.has_layer(layer) {
            layers.push(cache);
        } else {
            layers.push(fallback);
        }
    }
    PerLayerGraph::new(layers)
}

// ── Template-guided walk: score only features in the template's universe ──

/// Per-template per-layer feature universe: the set of features that ever
/// fire for this template across diverse entities.
///
/// Built by running forward passes for a template with many entities,
/// capturing which features activate at each layer, and taking the union.
pub struct TemplateUniverse {
    pub name: String,
    /// layer → sorted vec of feature indices that fire for this template.
    pub features: std::collections::HashMap<usize, Vec<usize>>,
}

impl TemplateUniverse {
    /// Build by running dense forward passes for a template with multiple entities.
    /// `template`: format string with `{}` for entity slot.
    /// `entities`: list of entities to test.
    /// `activation_threshold`: minimum |activation| to count a feature as firing.
    pub fn build(
        weights: &ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        name: &str,
        template: &str,
        entities: &[&str],
        ffn: &dyn FfnBackend,
        activation_threshold: f32,
    ) -> Self {
        let all_layers: Vec<usize> = (0..weights.num_layers).collect();
        let mut layer_features: std::collections::HashMap<usize, std::collections::HashSet<usize>> =
            std::collections::HashMap::new();

        for entity in entities {
            let prompt = template.replace("{}", entity);
            let encoding = match tokenizer.encode(prompt.as_str(), true) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let token_ids: Vec<u32> = encoding.get_ids().to_vec();

            let trace = crate::forward::trace_forward_full(
                weights, &token_ids, &all_layers,
                true, 500, false, ffn,
            );

            for (layer, acts) in &trace.activations {
                let set = layer_features.entry(*layer).or_default();
                for (feat, act) in acts {
                    if act.abs() > activation_threshold {
                        set.insert(*feat);
                    }
                }
            }
        }

        let features = layer_features.into_iter()
            .map(|(layer, set)| {
                let mut v: Vec<usize> = set.into_iter().collect();
                v.sort_unstable();
                (layer, v)
            })
            .collect();

        Self { name: name.to_string(), features }
    }

    /// Get the feature universe for a layer.
    pub fn get(&self, layer: usize) -> Option<&[usize]> {
        self.features.get(&layer).map(|v| v.as_slice())
    }

    /// Total features across all layers.
    pub fn total_features(&self) -> usize {
        self.features.values().map(|v| v.len()).sum()
    }

    /// Print a summary.
    pub fn summary(&self) {
        let mut layers: Vec<usize> = self.features.keys().copied().collect();
        layers.sort();
        for &layer in &layers {
            let n = self.features[&layer].len();
            if n > 0 {
                print!("L{layer}:{n} ");
            }
        }
        println!();
    }
}

/// Guided walk layer graph: dense attention + walk FFN restricted to
/// the template's per-layer feature universe.
///
/// Instead of scoring all 10,240 features, scores only the ~100-400
/// that the template ever activates. Per-feature dot products + accumulations.
pub struct GuidedWalkLayerGraph<'a> {
    pub weights: &'a ModelWeights,
    pub universe: &'a TemplateUniverse,
    pub index: &'a dyn larql_vindex::GateIndex,
}

impl<'a> LayerGraph for GuidedWalkLayerGraph<'a> {
    fn forward_layer(
        &self,
        weights: &ModelWeights,
        h: &Array2<f32>,
        layer: usize,
    ) -> Option<LayerOutput> {
        // Attention: dense matmul
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block(weights, h, layer, false)?;

        // FFN: guided walk — score only template universe features
        let residual = guided_walk_ffn(weights, &h_post_attn, layer, self.universe, self.index);

        Some(LayerOutput { residual, activation: None, attention: None })
    }

    fn name(&self) -> &str { "guided-walk" }
}

/// Guided walk FFN: pre-FFN norm → gate scores for universe → GEGLU → accumulate.
///
/// Gate: scores all features (one gate_scores_batch call), but only processes
/// the template universe features for up/down. The gate call is the same cost
/// as dense, but up/down computation drops from 10,240 to ~100-400 features.
/// Up/down: per-feature dot products and scaled adds (no matmul).
fn guided_walk_ffn(
    weights: &ModelWeights,
    h_post_attn: &Array2<f32>,
    layer: usize,
    universe: &TemplateUniverse,
    index: &dyn larql_vindex::GateIndex,
) -> Array2<f32> {
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();
    let hidden = h_post_attn.shape()[1];
    let seq_len = h_post_attn.shape()[0];

    // Pre-FFN norm
    let pre_ffn_key = if arch.has_post_norms() {
        arch.pre_feedforward_layernorm_key(layer)
    } else {
        Some(arch.post_attention_layernorm_key(layer))
    };
    let h_ffn = match pre_ffn_key {
        Some(key) => crate::forward::apply_norm(weights, h_post_attn, &key, norm_offset),
        None => crate::residual::rms_norm(h_post_attn, None, norm_offset),
    };

    // Get template universe for this layer
    let features = match universe.get(layer) {
        Some(f) if !f.is_empty() => f,
        _ => return h_post_attn.clone(),
    };

    let up_view = match index.up_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };
    let down_view = match index.down_layer_matrix(layer) {
        Some(v) => v,
        None => return h_post_attn.clone(),
    };

    let is_gated = arch.ffn_type() == larql_models::FfnType::Gated;
    let use_gelu = matches!(
        arch.activation(),
        larql_models::Activation::GeluTanh | larql_models::Activation::Gelu
    );

    // Gate scores: one batch call, then index into universe features only.
    // This is still a matmul for gate, but up/down are per-feature only.
    let gate_scores = match index.gate_scores_batch(layer, &h_ffn) {
        Some(gs) => gs,
        None => return h_post_attn.clone(),
    };

    let mut ffn_out = Array2::<f32>::zeros((seq_len, hidden));

    for s in 0..seq_len {
        let x_row = h_ffn.row(s);
        let mut out_row = ffn_out.row_mut(s);

        for &feat in features {
            let gate_score = gate_scores[[s, feat]];

            let act = if is_gated {
                let up_score = up_view.row(feat).dot(&x_row);
                let activated_gate = if use_gelu {
                    crate::ffn::gelu_tanh(gate_score)
                } else {
                    gate_score * crate::ffn::sigmoid(gate_score)
                };
                activated_gate * up_score
            } else {
                let v = gate_score;
                if use_gelu { crate::ffn::gelu_tanh(v) } else { v * crate::ffn::sigmoid(v) }
            };

            if act.abs() > 1e-10 {
                let down_row = down_view.row(feat);
                out_row.scaled_add(act, &down_row);
            }
        }
    }

    // Post-FFN norm + residual
    let res_mult = arch.residual_multiplier();
    if arch.has_post_norms() {
        let normed = match arch.post_feedforward_layernorm_key(layer) {
            Some(key) => crate::forward::apply_norm(weights, &ffn_out, &key, norm_offset),
            None => crate::residual::rms_norm(&ffn_out, None, norm_offset),
        };
        if res_mult != 1.0 {
            h_post_attn + &(&normed * res_mult)
        } else {
            h_post_attn + &normed
        }
    } else if res_mult != 1.0 {
        h_post_attn + &(&ffn_out * res_mult)
    } else {
        h_post_attn + &ffn_out
    }
}

/// Run a full forward pass using vindex logits (KNN against lm_head mmap).
/// Replaces the 231ms dense logits matmul with a ~1ms KNN lookup.
pub fn predict_with_graph_vindex_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: &larql_vindex::VectorIndex,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    // Final norm
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    // Vindex logits: KNN against lm_head mmap
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    // Get raw scores from KNN (dot products against lm_head)
    let hits = index.lm_head_knn(&last_row, top_k);

    // Apply scaling, softcap, softmax over top-K
    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();

    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions }
}

/// Run a full forward pass using a LayerGraph for per-layer routing.
/// This is the generic layer loop — embedding → layers → logits.
pub fn predict_with_graph(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
) -> crate::forward::PredictResult {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    crate::forward::logits_to_predictions_pub(weights, &h, tokenizer, top_k)
}

/// Run a full forward pass with tracing (residuals + activations + attention).
pub fn trace_with_graph(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    graph: &dyn LayerGraph,
) -> crate::forward::TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let mut results = Vec::new();
    let mut activations = Vec::new();
    let mut attention_captures = Vec::new();

    for layer in 0..=max_layer.min(weights.num_layers - 1) {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => {
                h = output.residual;

                if capture_layers.contains(&layer) {
                    let last_row = h.row(seq_len - 1);
                    results.push((layer, last_row.to_vec()));

                    if let Some(act) = output.activation {
                        let act_row = act.row(seq_len - 1);
                        let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                        indexed.truncate(200);
                        activations.push((layer, indexed));
                    }

                    if let Some(attn) = output.attention {
                        attention_captures.push(crate::forward::LayerAttentionCapture {
                            layer,
                            weights: attn,
                        });
                    }
                }
            }
            None => break,
        }
    }

    crate::forward::TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}

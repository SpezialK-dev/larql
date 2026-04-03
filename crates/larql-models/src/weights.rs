//! Model weight tensors — the loaded representation of a model's parameters.

use std::collections::HashMap;
use ndarray::ArcArray2;
use crate::ModelArchitecture;

/// Type alias for weight tensors — ArcArray2 supports both owned and shared storage.
/// Owned: from safetensors loading (heap). Shared: from mmap (zero-copy).
pub type WeightArray = ArcArray2<f32>;

/// A loaded model's weight tensors, configuration, and architecture.
pub struct ModelWeights {
    pub tensors: HashMap<String, WeightArray>,
    pub vectors: HashMap<String, Vec<f32>>,
    pub embed: WeightArray,
    /// Output projection matrix. Same as embed if tie_word_embeddings=true,
    /// separate lm_head.weight otherwise.
    pub lm_head: WeightArray,
    pub arch: Box<dyn ModelArchitecture>,
    // Cached from arch.config() for convenience — these are hot-path values.
    pub num_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub rope_base: f64,
}

//! Matmul backend abstraction — CPU (Accelerate BLAS) and optional Metal GPU.
//!
//! The CPU backend delegates to ndarray `.dot()` which dispatches through
//! `cblas_sgemm` via Apple Accelerate on macOS (AMX-accelerated).
//!
//! The Metal backend dispatches tiled compute shaders on the GPU,
//! useful for batched attention (all heads in one submission).

pub mod cpu;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(test)]
mod tests;

use ndarray::Array2;

/// A single matmul operation for batch dispatch.
pub struct MatMulOp {
    pub a: Array2<f32>,
    pub b: Array2<f32>,
    pub transpose_b: bool,
}

/// Backend for matrix multiplication.
///
/// CPU implementation uses ndarray + BLAS (Accelerate on macOS).
/// Metal implementation uses GPU compute shaders.
pub trait MatMulBackend: Send + Sync {
    /// C = A * B where A is [m, k] and B is [k, n].
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;

    /// C = A * B^T where A is [m, k] and B is [n, k].
    fn matmul_transb(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32>;

    /// Batch dispatch — multiple matmuls in one submission.
    /// Default: serial. Metal overrides with parallel GPU dispatch.
    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        ops.iter()
            .map(|op| {
                if op.transpose_b {
                    self.matmul_transb(&op.a, &op.b)
                } else {
                    self.matmul(&op.a, &op.b)
                }
            })
            .collect()
    }

    /// Human-readable name for logging/benchmarks.
    fn name(&self) -> &str;
}

/// Create the best available backend.
///
/// With `--features metal`: tries Metal GPU first, auto-calibrates the
/// FLOP threshold for hybrid CPU/GPU dispatch, falls back to CPU.
/// Without: returns CPU (Accelerate BLAS on macOS, OpenBLAS on Linux).
pub fn default_backend() -> Box<dyn MatMulBackend> {
    #[cfg(feature = "metal")]
    {
        if let Some(m) = metal::MetalBackend::new() {
            m.calibrate();
            return Box::new(m);
        }
        eprintln!("[backend] Metal device not available, falling back to CPU");
    }
    Box::new(cpu::CpuBackend)
}

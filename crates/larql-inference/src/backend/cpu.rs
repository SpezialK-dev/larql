//! CPU matmul backend — delegates to ndarray `.dot()` which uses BLAS.
//!
//! On macOS with `blas-src = { features = ["accelerate"] }`, this dispatches
//! through `cblas_sgemm` on Apple's AMX coprocessor (~2-4 TFLOPS on M-series).

use ndarray::Array2;
use super::MatMulBackend;

/// CPU backend using ndarray + BLAS (Accelerate on macOS).
pub struct CpuBackend;

impl MatMulBackend for CpuBackend {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(b)
    }

    fn matmul_transb(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        a.dot(&b.t())
    }

    fn name(&self) -> &str {
        "cpu (Accelerate BLAS)"
    }
}

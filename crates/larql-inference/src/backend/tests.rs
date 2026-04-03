//! Unit tests for matmul backends.

use ndarray::Array2;
use super::{MatMulBackend, MatMulOp};
use super::cpu::CpuBackend;

/// Deterministic f32 data generator (matches project's LCG pattern).
fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

/// Reference matmul using f64 accumulation for validation.
fn reference_matmul(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[1];
    let mut c = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[[i, p]] as f64 * b[[p, j]] as f64;
            }
            c[[i, j]] = sum as f32;
        }
    }
    c
}

/// Reference matmul with B transposed.
fn reference_matmul_transb(a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
    let (m, k) = (a.shape()[0], a.shape()[1]);
    let n = b.shape()[0];
    let mut c = Array2::<f32>::zeros((m, n));
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f64;
            for p in 0..k {
                sum += a[[i, p]] as f64 * b[[j, p]] as f64;
            }
            c[[i, j]] = sum as f32;
        }
    }
    c
}

/// Max absolute difference between two matrices.
fn max_abs_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

// ── Shape tests ──

#[test]
fn matmul_output_shape() {
    let backend = CpuBackend;
    let a = synth_matrix(3, 5, 1);
    let b = synth_matrix(5, 7, 2);
    let c = backend.matmul(&a, &b);
    assert_eq!(c.shape(), &[3, 7]);
}

#[test]
fn matmul_transb_output_shape() {
    let backend = CpuBackend;
    let a = synth_matrix(3, 5, 1);
    let b = synth_matrix(7, 5, 2); // [n, k]
    let c = backend.matmul_transb(&a, &b);
    assert_eq!(c.shape(), &[3, 7]);
}

#[test]
fn matmul_single_element() {
    let backend = CpuBackend;
    let a = Array2::from_elem((1, 1), 3.0f32);
    let b = Array2::from_elem((1, 1), 4.0f32);
    let c = backend.matmul(&a, &b);
    assert!((c[[0, 0]] - 12.0).abs() < 1e-6);
}

// ── Correctness tests ──

#[test]
fn matmul_matches_reference_small() {
    let backend = CpuBackend;
    let a = synth_matrix(4, 6, 10);
    let b = synth_matrix(6, 8, 20);
    let got = backend.matmul(&a, &b);
    let want = reference_matmul(&a, &b);
    assert!(
        max_abs_diff(&got, &want) < 1e-5,
        "matmul mismatch: max diff = {}",
        max_abs_diff(&got, &want)
    );
}

#[test]
fn matmul_transb_matches_reference_small() {
    let backend = CpuBackend;
    let a = synth_matrix(4, 6, 30);
    let b = synth_matrix(8, 6, 40); // [n, k]
    let got = backend.matmul_transb(&a, &b);
    let want = reference_matmul_transb(&a, &b);
    assert!(
        max_abs_diff(&got, &want) < 1e-5,
        "matmul_transb mismatch: max diff = {}",
        max_abs_diff(&got, &want)
    );
}

#[test]
fn matmul_matches_reference_transformer_scale() {
    // Realistic attention projection: [seq_len, hidden] x [hidden, head_dim]
    let backend = CpuBackend;
    let a = synth_matrix(6, 256, 100);
    let b = synth_matrix(256, 64, 200);
    let got = backend.matmul(&a, &b);
    let want = reference_matmul(&a, &b);
    assert!(
        max_abs_diff(&got, &want) < 1e-3,
        "transformer-scale mismatch: max diff = {}",
        max_abs_diff(&got, &want)
    );
}

#[test]
fn matmul_transb_matches_reference_transformer_scale() {
    // Realistic Q @ K^T: [seq, head_dim] x [seq, head_dim] → [seq, seq]
    let backend = CpuBackend;
    let a = synth_matrix(6, 64, 300);
    let b = synth_matrix(6, 64, 400);
    let got = backend.matmul_transb(&a, &b);
    let want = reference_matmul_transb(&a, &b);
    assert!(
        max_abs_diff(&got, &want) < 1e-4,
        "QK^T-scale mismatch: max diff = {}",
        max_abs_diff(&got, &want)
    );
}

// ── Identity / zeros tests ──

#[test]
fn matmul_identity() {
    let backend = CpuBackend;
    let a = synth_matrix(4, 4, 50);
    let eye = Array2::eye(4);
    let got = backend.matmul(&a, &eye);
    assert!(max_abs_diff(&got, &a) < 1e-6);
}

#[test]
fn matmul_zeros() {
    let backend = CpuBackend;
    let a = synth_matrix(3, 5, 60);
    let z = Array2::zeros((5, 7));
    let got = backend.matmul(&a, &z);
    assert!(got.iter().all(|&v| v.abs() < 1e-10));
}

// ── Batch tests ──

#[test]
fn matmul_batch_matches_serial() {
    let backend = CpuBackend;

    let ops: Vec<MatMulOp> = (0..4)
        .map(|i| MatMulOp {
            a: synth_matrix(6, 64, 100 + i),
            b: synth_matrix(64, 64, 200 + i),
            transpose_b: false,
        })
        .collect();

    let batch_results = backend.matmul_batch(&ops);
    for (i, op) in ops.iter().enumerate() {
        let serial = backend.matmul(&op.a, &op.b);
        assert!(
            max_abs_diff(&batch_results[i], &serial) < 1e-6,
            "batch[{i}] differs from serial"
        );
    }
}

#[test]
fn matmul_batch_transb_matches_serial() {
    let backend = CpuBackend;

    let ops: Vec<MatMulOp> = (0..4)
        .map(|i| MatMulOp {
            a: synth_matrix(6, 64, 500 + i),
            b: synth_matrix(6, 64, 600 + i),
            transpose_b: true,
        })
        .collect();

    let batch_results = backend.matmul_batch(&ops);
    for (i, op) in ops.iter().enumerate() {
        let serial = backend.matmul_transb(&op.a, &op.b);
        assert!(
            max_abs_diff(&batch_results[i], &serial) < 1e-6,
            "batch_transb[{i}] differs from serial"
        );
    }
}

#[test]
fn matmul_batch_empty() {
    let backend = CpuBackend;
    let results = backend.matmul_batch(&[]);
    assert!(results.is_empty());
}

#[test]
fn matmul_batch_mixed_transpose() {
    let backend = CpuBackend;

    let ops = vec![
        MatMulOp {
            a: synth_matrix(3, 8, 700),
            b: synth_matrix(8, 5, 701),
            transpose_b: false,
        },
        MatMulOp {
            a: synth_matrix(3, 8, 702),
            b: synth_matrix(5, 8, 703),
            transpose_b: true,
        },
    ];

    let results = backend.matmul_batch(&ops);
    assert_eq!(results[0].shape(), &[3, 5]);
    assert_eq!(results[1].shape(), &[3, 5]);

    let want0 = backend.matmul(&ops[0].a, &ops[0].b);
    let want1 = backend.matmul_transb(&ops[1].a, &ops[1].b);
    assert!(max_abs_diff(&results[0], &want0) < 1e-6);
    assert!(max_abs_diff(&results[1], &want1) < 1e-6);
}

// ── Non-square tests ──

#[test]
fn matmul_tall_skinny() {
    let backend = CpuBackend;
    let a = synth_matrix(100, 4, 800);
    let b = synth_matrix(4, 3, 801);
    let got = backend.matmul(&a, &b);
    let want = reference_matmul(&a, &b);
    assert_eq!(got.shape(), &[100, 3]);
    assert!(max_abs_diff(&got, &want) < 1e-5);
}

#[test]
fn matmul_wide_flat() {
    let backend = CpuBackend;
    let a = synth_matrix(2, 100, 900);
    let b = synth_matrix(100, 200, 901);
    let got = backend.matmul(&a, &b);
    let want = reference_matmul(&a, &b);
    assert_eq!(got.shape(), &[2, 200]);
    assert!(max_abs_diff(&got, &want) < 1e-3);
}

// ── Backend trait tests ──

#[test]
fn cpu_backend_name() {
    let backend = CpuBackend;
    assert!(backend.name().contains("cpu"));
}

#[test]
fn default_backend_returns_valid_name() {
    let backend = super::default_backend();
    assert!(!backend.name().is_empty());
}

#[test]
fn default_backend_matmul_works() {
    let backend = super::default_backend();
    let a = synth_matrix(4, 8, 1000);
    let b = synth_matrix(8, 6, 1001);
    let c = backend.matmul(&a, &b);
    assert_eq!(c.shape(), &[4, 6]);
    let want = reference_matmul(&a, &b);
    assert!(max_abs_diff(&c, &want) < 1e-5);
}

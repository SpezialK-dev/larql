# larql-inference

Inference engine for transformer models. Forward pass, BLAS-fused attention, hardware-accelerated matmul backends, and pluggable FFN routing.

## Overview

This crate runs transformer forward passes with Apple Accelerate (AMX) and optional Metal GPU acceleration. It uses `larql-vindex` for gate KNN (sparse feature selection) and `larql-models` for weight loading and architecture definitions.

```rust
use larql_inference::InferenceModel;

// Load a model
let model = InferenceModel::load("google/gemma-3-4b-it")?;

// Run inference
let result = larql_inference::predict(
    model.weights(), model.tokenizer(), &token_ids, 5,
);
println!("Top prediction: {} ({:.1}%)", result.predictions[0].0, result.predictions[0].1 * 100.0);
```

## Key Components

| Module | Purpose |
|--------|---------|
| `backend/` | MatMulBackend trait: CPU (Accelerate BLAS) and Metal GPU with auto-calibration |
| `attention.rs` | BLAS-fused GQA attention with online softmax (no [seq,seq] allocation) |
| `forward.rs` | Forward pass: `predict()`, `predict_with_ffn()`, `forward_to_layer()` |
| `ffn/` | FFN evaluation: dense, sparse, highway, cached, experimental |
| `residual.rs` | RMS norm, layer norm |
| `trace/` | Residual stream decomposition and tiered storage |
| `vindex/walk_ffn.rs` | WalkFfn: sparse FFN using vindex gate KNN |
| `capture.rs` | Residual stream vector capture for probing |
| `walker/` | Weight-level graph walkers (no forward pass) |
| `model.rs` | Model loading (re-exports from larql-models) |

## Matmul Backend

All large matrix multiplications dispatch through the `MatMulBackend` trait:

```rust
use larql_inference::backend::{default_backend, MatMulBackend};

let backend = default_backend();  // Auto-selects CPU or Metal, calibrates
println!("Using: {}", backend.name());

let c = backend.matmul_transb(&input, &weights);
```

**CPU backend** (default): ndarray + `cblas_sgemm` via Apple Accelerate. AMX hardware at ~2-4 TFLOPS f32.

**Metal GPU backend** (`--features metal`): Tiled 32x32 compute shaders with buffer cache and auto-calibrated FLOP threshold. Weight matrices are uploaded to GPU once and reused across all calls.

```bash
# Build with Metal support
cargo build --release -p larql-inference --features metal
```

See [docs/inference-engine.md](../../docs/inference-engine.md) for architecture details and benchmarks.

## BLAS-Fused Attention

The attention kernel uses BLAS `gemv` inside an online-softmax loop. For each query position:

1. `scores = K[0..=qi] @ Q[qi]` (BLAS gemv, AMX-accelerated)
2. Scale + optional softcap + two-pass softmax (f64 accumulation)
3. `output = V[0..=qi]^T @ softmax_scores` (BLAS gemv)

Never allocates the `[seq, seq]` attention matrix. At Gemma-3's head_dim=256, **1.6x faster** than the materialized path. Supports GQA, softcap (Gemma2), attention weight capture.

## WalkFfn

The WalkFfn replaces the dense FFN with a sparse version that uses the vindex gate KNN:

1. Gate KNN selects top-8092 features (from gate_vectors.bin)
2. Only selected features go through up/down projections
3. Result is identical to dense FFN (97.91% on France->Paris)
4. Enables interpretable inference (see which features fired)

## Examples

```bash
# Fused attention demo (correctness, GQA, softcap, capture)
cargo run --release -p larql-inference --example attention_demo

# Fused attention benchmark (fused vs materialized, seq scaling)
cargo run --release -p larql-inference --example bench_attention

# Backend demo (routing, buffer cache, calibration)
cargo run --release -p larql-inference --example backend_demo --features metal

# Backend benchmark (CPU vs Metal at transformer scale)
cargo run --release -p larql-inference --example bench_backend --features metal

# Full inference benchmark (needs model weights)
cargo run --release -p larql-inference --example bench_inference

# End-to-end inference demo (needs model weights)
cargo run --release -p larql-inference --example inference_demo

# Clustering and pair matching demos
cargo run -p larql-inference --example clustering_demo
cargo run -p larql-inference --example pair_matching_demo
```

## Tests

```bash
# All tests (109 total)
cargo test -p larql-inference

# With Metal GPU tests (+6 Metal-specific)
cargo test -p larql-inference --features metal

# Individual test suites
cargo test -p larql-inference --test test_fused_attention   # 18 tests
cargo test -p larql-inference --test test_backend           # 13+6 tests
cargo test -p larql-inference --test test_modules           # 15 tests
cargo test -p larql-inference --test test_trace             # 14 tests
cargo test -p larql-inference --test test_walkers           # 12 tests
cargo test -p larql-inference --test test_walker_utils      # 10 tests
```

| Area | Tests | Coverage |
|------|-------|----------|
| Backend (unit + integration) | 34 | Shape, correctness, batch, Metal vs CPU, factory |
| Fused attention | 18 | GQA, softcap, capture, reference agreement, edge cases |
| FFN | 9 | SiLU, GELU, dense, highway, multi-position |
| Attention/residual | 10 | RoPE, GQA, RMS norm, per-head norm |
| Trace stores | 14 | Write/read, tiers, boundaries, additive property |
| Walkers | 12 | Weight/attention walkers, vector extractor |
| Utils | 10 | Top-k, rounding, entity sorting |

## Crate Dependencies

```
larql-models      ModelWeights, architecture traits, quant
    |
larql-vindex      VectorIndex, gate KNN, patches
    |
larql-inference   Forward pass, attention, backends, WalkFfn
```

## License

Apache-2.0

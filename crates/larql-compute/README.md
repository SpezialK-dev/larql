# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

## Backends

| Backend | Feature flag | f32 matmul | Q4 fused ops | Multi-layer pipeline |
|---------|-------------|------------|--------------|---------------------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled compute shaders | Simdgroup Q4×Q8 (v4 kernel) | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance (M3 Max, Gemma 3 4B)

```
Operation                      CPU         Metal       Winner
─────────────────────────────  ──────────  ──────────  ──────
f32 matmul [6,2560²]           1.03ms      0.70ms     Metal
Q4 matvec (v4) [10240,2560]    0.96ms      0.24ms     Metal (4×)
Q4 sparse matvec (K=400)       —           0.25ms     Metal
Q4 pair_batch (6 pos)          11.4ms      1.39ms     Metal (8×)
Q8 matvec (V projection)       —           ~0.25ms    Metal
21-layer Q4 FFN                60ms        8.5ms      Metal (7×)
Full layer (attn+FFN, seq=1)   —           1.7ms      Metal
Full pipeline (21L, all Q4)    —           10.4ms     Metal (96 tok/s)
KV-cached attention (T=10)     —           1.1ms      Metal (21 layers)
```

## Quick start

```rust
use larql_compute::{ComputeBackend, default_backend, cpu_backend};

// Auto-detect best backend (Metal if available, else CPU)
let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// Force CPU only (no GPU, no calibration overhead)
let cpu = cpu_backend();

// f32 matmul
let c = backend.matmul_transb(a.view(), b.view());

// Q4 fused operations
if backend.has_q4() {
    let scores = backend.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden);
}
```

## Architecture

Every shader and every operation lives in its own file with its own tests.

```
src/
  lib.rs                    — crate root, exports, factory functions
  backend.rs                — ComputeBackend trait + helper functions

  cpu/
    mod.rs                  — CpuBackend struct + trait impl
    ops/
      f32_matmul.rs         — BLAS sgemm/sgemm_transb       (3 tests)
      q4_matvec.rs          — C kernel Q4×Q8 matvec          (2 tests)
      q4_vecmat.rs          — C kernel Q4 vecmat             (2 tests)
      q4_common.rs          — Q8 quantize, C FFI decls       (2 tests)
      q8_matvec.rs          — Q8 matvec + weight quantizer   (2 tests)
      geglu.rs              — SiLU gate activation            (3 tests)
      attention.rs          — Causal attention (fused QKV)    (3 tests)

  metal/                    (feature-gated: --features metal)
    mod.rs                  — MetalBackend struct + trait impl
    shaders/                — 18 Metal Shading Language kernels:
      common.rs             — f16 decode, metal_stdlib header
      sgemm.rs              — f32 tiled matmul C=A×B
      sgemm_transb.rs       — f32 tiled matmul C=A×B^T
      q4_matvec.rs          — Q4×Q8 simdgroup (v1, original)
      q4_matvec_v2.rs       — Q4×f32, 4 rows per thread
      q4_matvec_v3.rs       — Q4×f32, 8 rows unrolled
      q4_matvec_v4.rs       — Q4×Q8 uint32 wide loads (production, 0.24ms)
      q4_matvec_v5.rs       — Q4×Q8, 256 rows/TG, no simd_sum
      q4_vecmat.rs          — Q4 scatter-accumulate
      q4_f32_matvec.rs      — Q4×f32 for transposed down
      q4_sparse_matvec.rs   — Sparse Q4 matvec by index (walk architecture)
      q8_matvec.rs          — Q8×Q8 matvec (V projection)
      geglu.rs              — Element-wise SiLU gate
      quantize_q8.rs        — f32→Q8 quantization (layer chaining)
      residual_inject.rs    — Buffer copy, residual add, RMS norm
      causal_attention.rs   — Basic causal attention (seq≤64)
      kv_attention.rs       — KV-cached attention + cache append
    ops/                    — GPU dispatch modules:
      q4_common.rs          — Q4Pipelines struct + quantize_to_q8
      q4_matvec.rs          — Single Q4 matvec dispatch
      q4_vecmat.rs          — Single Q4 vecmat dispatch
      q4_f32_matvec.rs      — Single Q4×f32 matvec dispatch
      q4_batched.rs         — pair_batch + multi_layer_ffn
      full_layer.rs         — Attention + FFN in one cmd buffer
      full_pipeline.rs      — 21 layers, all Q4, one submission
      kv_cache.rs           — KV cache struct + append+attend
    buffers.rs              — GPU buffer cache (zero-copy mmap)
    calibrate.rs            — CPU vs GPU auto-calibration
    f32_ops.rs              — f32 dispatch with GPU/CPU routing

  csrc/
    q4_dot.c                — C kernel: ARM vdotq_s32 + scalar fallback
```

## Tests

```bash
# CPU tests only (17 tests)
cargo test -p larql-compute

# CPU + Metal tests (44 tests)
cargo test -p larql-compute --features metal
```

Test coverage:
- f32 matmul: CPU vs ndarray reference, identity, shapes
- Q4 matvec: CPU kernel, Metal v4, zero input, small matrix, Metal vs CPU
- Q4 vecmat: CPU kernel, Metal, zero activation
- Q4 sparse: Metal sparse matches dense at selected indices
- Q8 matvec: CPU kernel, Q8 vs f32 cosine > 0.999, Metal nonzero
- GEGLU: SiLU basic, Metal vs CPU cross-validation
- Residual: Metal add correctness
- Attention: single token, causal mask, output shape
- Batch: Metal pair_batch matches individual calls
- Multi-layer: 21-layer pipeline produces output
- Shader compilation: all 19 kernel functions exist
- Buffer cache: pointer reuse verified
- Trait dispatch: Metal implements ComputeBackend correctly

## Benchmarks

```bash
# Every operation, CPU + Metal side by side
cargo run --release -p larql-compute --features metal --example bench_shaders

# Kernel variant comparison (v1-v5 + sparse)
cargo run --release -p larql-compute --features metal --example bench_kernel_variants

# Multi-layer pipeline, mixed backend, generation simulation
cargo run --release -p larql-compute --features metal --example bench_pipeline

# Full 21-layer pipeline (all Q4, one submission)
cargo run --release -p larql-compute --features metal --example bench_full_pipeline

# Token generation with KV cache
cargo run --release -p larql-compute --features metal --example bench_kv_cache

# All operations at representative sizes
cargo run --release -p larql-compute --features metal --example bench_full

# Backend auto-detection demo
cargo run --release -p larql-compute --features metal --example demo
```

## Design principles

1. **One file per operation** — every shader and dispatch function lives in its own file with its own tests.
2. **Trait-based dispatch** — callers use `ComputeBackend` exclusively. Implementation is invisible.
3. **Zero-copy for mmap** — weight buffers use `newBufferWithBytesNoCopy` on Apple Silicon unified memory.
4. **Cached vs transient** — weight buffers cached by pointer. Input/output allocated fresh each call.
5. **Feature-gated** — Metal with `--features metal`. CPU always available. CUDA planned.
6. **Auto-calibration** — Metal benchmarks CPU vs GPU at startup for optimal routing.
7. **Batch API** — multi-layer pipeline encodes all operations in one command buffer.
8. **Mixed precision** — Q4 for Q/K/O projections + FFN, Q8 for V projection (validated by cosine similarity testing).

## Adding a new backend

1. Create `src/newbackend/mod.rs`
2. Implement `ComputeBackend` trait
3. Add feature flag to `Cargo.toml`
4. Add to `default_backend()` factory with priority
5. Add tests in `tests/test_newbackend.rs`
6. Add to `bench_shaders.rs` for side-by-side comparison

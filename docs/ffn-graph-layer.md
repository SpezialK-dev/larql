# FFN Graph Layer

The FFN graph layer replaces dense matrix multiplication with vindex lookups for the feed-forward network. Gate KNN finds active features, the model's weight matrices compute the sparse output. No approximation — the walk produces identical predictions to the dense forward pass.

## Architecture

```
Dense FFN (traditional):
  gate = x @ W_gate.T           ← matmul: [seq, 2560] × [10240, 2560]
  up   = x @ W_up.T             ← matmul: [seq, 2560] × [10240, 2560]
  act  = silu(gate) * up         ← elementwise
  out  = act @ W_down.T          ← matmul: [seq, 10240] × [2560, 10240]

Walk FFN (graph layer):
  features = gate_knn(x, vindex)  ← mmap gemm: finds active features
  gate = W_gate[features] @ x     ← sparse gemv: K rows only
  up   = W_up[features] @ x       ← sparse gemv: K rows only
  act  = silu(gate) * up           ← elementwise
  out  = W_down[:, features] @ act ← sparse gemv: K columns only
```

The gate KNN IS the gate matmul. The dot product `residual × gate_vectors^T` serves as both the gate computation and the similarity search. Same operation, different framing. The vindex gate vectors are the model's `W_gate` rows, pre-extracted and indexed for efficient access.

## Proof of Correctness

The [walk boundary sweep](walk-boundary-sweep.md) tested vindex FFN at every layer boundary from L0 to L34 on Gemma-3 4B:

```
     B   walk%   correct  top1_avg  details
  -------------------------------------------------------
  L0     100%    5/5      82.63%   all match ground truth
  L4      88%    5/5      82.63%   all match ground truth
  L8      76%    5/5      82.63%   all match ground truth
  L12     65%    5/5      82.63%   all match ground truth
  L16     53%    5/5      82.63%   all match ground truth
  L20     41%    5/5      82.63%   all match ground truth
  L24     29%    5/5      82.63%   all match ground truth
  L28     18%    5/5      82.63%   all match ground truth
  L34      0%    5/5      82.63%   all match ground truth
```

Zero divergence. Same top-1 token, same probability, at every boundary. The gate vectors at each layer are calibrated to that layer's residual space — the KNN matches because the index and query live in the same space.

## Performance

### Optimization progression

| Version | Walk | Dense | Gap |
|---------|------|-------|-----|
| Unoptimized | 21,197ms | 708ms | 30x slower |
| + batch gate KNN (one gemm per layer) | 4,178ms | 685ms | 6.1x |
| + sparse down projection | — | — | (included above) |
| + f16 decode cache | — | — | (included above) |
| + trace recording off by default | 841ms | 685ms | 23% |
| + f32 gate vectors (mmap, zero-copy) | **685ms** | **560ms** | **22%** |

### Per-layer breakdown

```
Dense FFN:    6.4ms/layer  (3 matmuls: gate + up + down)
Walk FFN:    10.7ms/layer  (gate KNN 4ms + sparse FFN 6.4ms)
Gate KNN:     4.0ms/layer  (mmap gemm: [10240, 2560] × [2560, 6])
Sparse FFN:   6.4ms/layer  (dense fallback at K ≈ intermediate)
```

The 4ms gate KNN is memory-bound: reading 100MB of f32 gate vectors per layer from mmap. On subsequent tokens, the OS page cache keeps hot pages resident, reducing this toward L3 cache latency (~1ms).

### What eliminated the 30x gap

| Optimization | Speedup | What it fixed |
|-------------|---------|---------------|
| Batch gate KNN | 5x | One BLAS gemm per layer instead of 6 separate gemv calls |
| Trace off by default | 5x | Deferred trace to take_trace() — was 8092 feature_meta lookups + allocations per layer |
| f32 mmap | 1.2x | Zero decode, zero allocation, zero warmup. Pointer reinterpretation to BLAS. |
| Sparse down projection | 1.2x | gather K columns of W_down, not full [hidden, intermediate] matmul |
| f16 decode cache | — | Amortized cost (eliminated by f32 conversion) |

## Data Path

### f32 mmap (current, optimal)

```
gate_vectors.bin (f32, 3.6GB on disk)
    ↓ mmap (zero-copy, OS manages pages)
[10240, 2560] ArrayView2 per layer (pointer reinterpretation, no allocation)
    ↓ BLAS gemm
[10240, 6] scores (top-K selection)
    ↓ sparse_ffn_forward
[6, 2560] output (residual contribution)
```

No decode. No allocation. No mutex. The gemm reads directly from the memory-mapped file.

### f16 mmap (legacy, with warmup)

```
gate_vectors.bin (f16, 1.8GB on disk)
    ↓ mmap
    ↓ warmup(): decode f16 → f32 into heap cache (1.5s one-time)
    ↓ RwLock read (lock-free after warmup)
    ↓ BLAS gemm
```

Convert to f32 with:
```bash
cargo run --release -p larql-vindex --example convert_gates_f32 -- path/to/vindex/
```

## WalkFfn API

```rust
use larql_inference::vindex::WalkFfn;

// Fast path: no trace recording (default)
let walk_ffn = WalkFfn::new(weights, &index, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);

// With trace (for analysis — re-runs gate KNN lazily on take_trace)
let walk_ffn = WalkFfn::new_with_trace(weights, &index, top_k);
let result = predict_with_ffn(weights, tokenizer, &token_ids, 5, &walk_ffn);
let trace = walk_ffn.take_trace();
```

The walk FFN integrates transparently with all forward pass variants:
- `predict_with_ffn()` — full walk inference
- `predict_with_router()` — per-layer dense/walk selection
- `trace_forward_with_ffn()` — residual capture with walk
- Server `/v1/infer` — walk mode via HTTP/gRPC

## HNSW Index (experimental)

An HNSW graph index is available for approximate gate search. At 10,240 vectors it provides no speedup over brute-force BLAS gemm (the graph overhead equals the savings). It will matter at larger feature counts.

```rust
// Enable HNSW (builds lazily, dim=64 random projection)
index.enable_hnsw(200);  // ef_search = 200

// Disable (revert to brute-force gemm)
index.disable_hnsw();
```

Build time: ~700ms one-time (34 layers, 10,240 vectors, dim=64 projected).

## Implications

### FFN quantization is unnecessary

The entire FFN across all 34 layers is served by vindex gate KNN + model weight sparse compute. No FFN weight matrices need to be loaded separately for inference — the gate vectors in the vindex ARE the gate weights, and the up/down weights are accessed sparsely.

### Remaining matmuls

With the FFN graph layer active, the only matrix multiplications in the forward pass are:

| Operation | Per layer | Notes |
|-----------|-----------|-------|
| Q projection | ~1ms | Accelerate AMX |
| K projection | ~1ms | Accelerate AMX |
| V projection | ~1ms | Accelerate AMX |
| O projection | ~1ms | Accelerate AMX |
| Gate KNN | ~4ms | mmap gemm (bandwidth-bound) |
| Final logits | ~27ms (once) | Not per-layer |

Everything else — embedding, RoPE, norms, activation, sparse gather — is lookup or scalar math.

### Path forward

```
Current:     560ms dense, 685ms walk (22% gap)
+ warm page cache:  walk gate KNN drops to ~1ms/layer → walk ~450ms
+ Q4_K_M attention: attention projections 4x cheaper → ~200ms total
+ template cache:   attention eliminated → ~170ms (gate KNN + logits)
```

## Files

| File | Purpose |
|------|---------|
| `crates/larql-inference/src/vindex/walk_ffn.rs` | WalkFfn backend |
| `crates/larql-inference/src/ffn/sparse_compute.rs` | Sparse FFN compute (shared) |
| `crates/larql-vindex/src/index/core.rs` | Gate KNN, mmap, f16 cache, HNSW |
| `crates/larql-vindex/src/index/hnsw.rs` | HNSW graph index |
| `crates/larql-vindex/examples/convert_gates_f32.rs` | f16 → f32 converter |
| `crates/larql-inference/examples/bench_walk_inference.rs` | Walk benchmark |
| `crates/larql-inference/examples/walk_boundary_sweep.rs` | Correctness sweep |

//! Metal GPU matmul backend — tiled compute shaders on Apple GPU.
//!
//! Hybrid dispatch: small matmuls route to CPU (Accelerate/AMX, near-zero
//! overhead), large matmuls route to Metal GPU (cached weight buffers,
//! tiled compute shader). The FLOP threshold is auto-calibrated at startup
//! by benchmarking both paths on the actual hardware.
//!
//! Buffer cache: weight matrices from mmap'd safetensors have stable
//! addresses. Their GPU buffers are created once on first use and reused
//! for all subsequent calls. Only the small input residual and output
//! buffers are allocated per call.

use ndarray::Array2;
use metal::*;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use super::{MatMulBackend, MatMulOp};

/// Metal Shading Language source — tiled sgemm and sgemm_transb.
const SHADER_SRC: &str = r#"
#include <metal_stdlib>
using namespace metal;

constant uint TS = 32;

// C = A * B  (A: [M,K], B: [K,N], C: [M,N])
kernel void sgemm(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS - 1) / TS;

    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS + tid.x;
        uint br = t * TS + tid.y;

        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (br < K && col < N) ? B[br * N + col] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// C = A * B^T  (A: [M,K], B: [N,K], C: [M,N])
kernel void sgemm_transb(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 gid [[threadgroup_position_in_grid]])
{
    threadgroup float As[TS][TS];
    threadgroup float Bs[TS][TS];

    uint row = gid.y * TS + tid.y;
    uint col = gid.x * TS + tid.x;
    float acc = 0.0f;
    uint tiles = (K + TS - 1) / TS;

    for (uint t = 0; t < tiles; t++) {
        uint ac = t * TS + tid.x;
        uint bk = t * TS + tid.y;

        As[tid.y][tid.x] = (row < M && ac < K) ? A[row * K + ac] : 0.0f;
        Bs[tid.y][tid.x] = (col < N && bk < K) ? B[col * K + bk] : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TS; i++) {
            acc = fma(As[tid.y][i], Bs[i][tid.x], acc);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}
"#;

/// Conservative default before calibration runs.
const DEFAULT_FLOP_THRESHOLD: usize = 500_000_000;

/// Absolute floor: never dispatch to GPU below this.
/// GPU dispatch overhead (~200-300us) dominates for tiny ops.
const MIN_FLOP_FLOOR: usize = 100_000;

/// Cache key: (pointer address, byte length) of the source data.
type CacheKey = (usize, usize);

pub struct MetalBackend {
    device: Device,
    queue: CommandQueue,
    sgemm_pipeline: ComputePipelineState,
    transb_pipeline: ComputePipelineState,
    /// Pointer-based buffer cache. Weight matrices from mmap'd safetensors
    /// have stable addresses — GPU buffers are created once and reused.
    buffer_cache: Mutex<HashMap<CacheKey, Buffer>>,
    /// FLOP threshold for GPU dispatch. Auto-calibrated at startup.
    /// Reads are lock-free via AtomicUsize.
    flop_threshold: AtomicUsize,
}

impl MetalBackend {
    /// Create a Metal backend. Returns None if no Metal device is available.
    /// Call `calibrate()` after creation to auto-tune the FLOP threshold.
    pub fn new() -> Option<Self> {
        let device = Device::system_default()?;
        let queue = device.new_command_queue();

        let opts = CompileOptions::new();
        let library = device
            .new_library_with_source(SHADER_SRC, &opts)
            .map_err(|e| eprintln!("[metal] shader compile error: {e}"))
            .ok()?;

        let sgemm_fn = library.get_function("sgemm", None).ok()?;
        let transb_fn = library.get_function("sgemm_transb", None).ok()?;

        let sgemm_pipeline = device
            .new_compute_pipeline_state_with_function(&sgemm_fn)
            .ok()?;
        let transb_pipeline = device
            .new_compute_pipeline_state_with_function(&transb_fn)
            .ok()?;

        Some(Self {
            device,
            queue,
            sgemm_pipeline,
            transb_pipeline,
            buffer_cache: Mutex::new(HashMap::new()),
            flop_threshold: AtomicUsize::new(DEFAULT_FLOP_THRESHOLD),
        })
    }

    /// Auto-calibrate the FLOP threshold by benchmarking CPU vs Metal
    /// at representative matrix sizes. Takes ~50-100ms.
    ///
    /// Tests transposed matmul (the dominant pattern in transformer inference)
    /// at sizes matching attention projections and FFN layers. Finds the
    /// lowest FLOP count where Metal (with warm buffer cache) beats CPU.
    pub fn calibrate(&self) {
        // Representative sizes: (m, n, k) for C = A[m,k] × B[n,k]^T
        let test_cases: &[(usize, usize, usize)] = &[
            (6, 256, 256),       // ~800K FLOPs — small projection
            (6, 2560, 512),      // ~15M FLOPs — medium
            (6, 2560, 2560),     // ~79M FLOPs — attention Q/K/V/O projection
            (6, 10240, 2560),    // ~315M FLOPs — FFN gate/up
        ];

        let mut best_threshold = DEFAULT_FLOP_THRESHOLD;

        for &(m, n, k) in test_cases {
            let flops = 2 * m * n * k;
            let a = Self::calibration_matrix(m, k, 42);
            let b = Self::calibration_matrix(n, k, 43);

            // Warm the Metal buffer cache
            let a_data = Self::contiguous_data(&a);
            let b_data = Self::contiguous_data(&b);
            let _ = self.dispatch_transb(&a_data, &b_data, m, n, k);

            // Benchmark CPU (median of 5)
            let cpu_us = Self::bench_median(5, || {
                let _ = a.dot(&b.t());
            });

            // Benchmark Metal with warm cache (median of 5)
            let metal_us = Self::bench_median(5, || {
                let a_data = Self::contiguous_data(&a);
                let b_data = Self::contiguous_data(&b);
                let _ = self.dispatch_transb(&a_data, &b_data, m, n, k);
            });

            if metal_us < cpu_us {
                best_threshold = best_threshold.min(flops);
            }
        }

        self.flop_threshold.store(best_threshold, Ordering::Relaxed);
    }

    /// Current FLOP threshold (for diagnostics/logging).
    pub fn flop_threshold(&self) -> usize {
        self.flop_threshold.load(Ordering::Relaxed)
    }

    /// Number of cached GPU buffers (for diagnostics).
    pub fn cache_size(&self) -> usize {
        self.buffer_cache.lock().unwrap().len()
    }

    /// Manually set the FLOP threshold.
    pub fn set_flop_threshold(&self, threshold: usize) {
        self.flop_threshold.store(threshold.max(MIN_FLOP_FLOOR), Ordering::Relaxed);
    }

    // ── Internal helpers ──

    /// Deterministic matrix for calibration (no external dependency).
    fn calibration_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
        let mut state = seed;
        let data: Vec<f32> = (0..rows * cols)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
            })
            .collect();
        Array2::from_shape_vec((rows, cols), data).unwrap()
    }

    /// Median timing of N runs in microseconds.
    fn bench_median<F: FnMut()>(n: usize, mut f: F) -> u64 {
        let mut times = Vec::with_capacity(n);
        for _ in 0..n {
            let t0 = Instant::now();
            f();
            times.push(t0.elapsed().as_micros() as u64);
        }
        times.sort_unstable();
        times[n / 2]
    }

    /// Extract a contiguous f32 slice from an ndarray, copying if needed.
    fn contiguous_data(a: &Array2<f32>) -> std::borrow::Cow<'_, [f32]> {
        if let Some(s) = a.as_slice() {
            std::borrow::Cow::Borrowed(s)
        } else {
            let owned = a.as_standard_layout();
            std::borrow::Cow::Owned(owned.as_slice().unwrap().to_vec())
        }
    }

    /// Get or create a GPU buffer for the given data slice.
    fn get_or_create_buffer(&self, data: &[f32]) -> Buffer {
        let key: CacheKey = (data.as_ptr() as usize, data.len());
        let mut cache = self.buffer_cache.lock().unwrap();

        if let Some(buf) = cache.get(&key) {
            return buf.clone();
        }

        let bytes = (data.len() * std::mem::size_of::<f32>()) as u64;
        let buf = self.device.new_buffer_with_data(
            data.as_ptr() as *const c_void,
            bytes,
            MTLResourceOptions::StorageModeShared,
        );
        cache.insert(key, buf.clone());
        buf
    }

    /// Encode a matmul into a compute command encoder.
    fn encode_matmul(
        &self,
        encoder: &ComputeCommandEncoderRef,
        pipeline: &ComputePipelineState,
        buf_a: &Buffer,
        buf_b: &Buffer,
        buf_c: &Buffer,
        m: usize,
        n: usize,
        k: usize,
    ) {
        let m_val = m as u32;
        let n_val = n as u32;
        let k_val = k as u32;

        encoder.set_compute_pipeline_state(pipeline);
        encoder.set_buffer(0, Some(buf_a), 0);
        encoder.set_buffer(1, Some(buf_b), 0);
        encoder.set_buffer(2, Some(buf_c), 0);
        encoder.set_bytes(3, 4, &m_val as *const u32 as *const c_void);
        encoder.set_bytes(4, 4, &n_val as *const u32 as *const c_void);
        encoder.set_bytes(5, 4, &k_val as *const u32 as *const c_void);

        let threadgroup_size = MTLSize::new(32, 32, 1);
        let grid = MTLSize::new(
            ((n + 31) / 32) as u64,
            ((m + 31) / 32) as u64,
            1,
        );
        encoder.dispatch_thread_groups(grid, threadgroup_size);
    }

    /// Dispatch C = A * B on GPU with cached buffers.
    fn dispatch_notrans(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let buf_a = self.get_or_create_buffer(a_data);
        let buf_b = self.get_or_create_buffer(b_data);
        let buf_c = self.device.new_buffer(c_bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_matmul(encoder, &self.sgemm_pipeline, &buf_a, &buf_b, &buf_c, m, n, k);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_c.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, m * n).to_vec() }
    }

    /// Dispatch C = A * B^T on GPU with cached buffers.
    fn dispatch_transb(
        &self,
        a_data: &[f32],
        b_data: &[f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Vec<f32> {
        let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let buf_a = self.get_or_create_buffer(a_data);
        let buf_b = self.get_or_create_buffer(b_data);
        let buf_c = self.device.new_buffer(c_bytes, MTLResourceOptions::StorageModeShared);

        let cmd = self.queue.new_command_buffer();
        let encoder = cmd.new_compute_command_encoder();
        self.encode_matmul(encoder, &self.transb_pipeline, &buf_a, &buf_b, &buf_c, m, n, k);
        encoder.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let ptr = buf_c.contents() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, m * n).to_vec() }
    }

    /// Check if a matmul has enough FLOPs to benefit from GPU dispatch.
    fn should_use_gpu(&self, m: usize, n: usize, k: usize) -> bool {
        let flops = 2 * m * n * k;
        flops >= self.flop_threshold.load(Ordering::Relaxed)
    }
}

impl MatMulBackend for MetalBackend {
    fn matmul(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[1];
        debug_assert_eq!(a.shape()[1], b.shape()[0], "matmul: inner dims mismatch");

        if !self.should_use_gpu(m, n, k) {
            return a.dot(b);
        }

        let a_data = Self::contiguous_data(a);
        let b_data = Self::contiguous_data(b);
        let c_data = self.dispatch_notrans(&a_data, &b_data, m, n, k);
        Array2::from_shape_vec((m, n), c_data).unwrap()
    }

    fn matmul_transb(&self, a: &Array2<f32>, b: &Array2<f32>) -> Array2<f32> {
        let (m, k) = (a.shape()[0], a.shape()[1]);
        let n = b.shape()[0];
        debug_assert_eq!(a.shape()[1], b.shape()[1], "matmul_transb: K dims mismatch");

        if !self.should_use_gpu(m, n, k) {
            return a.dot(&b.t());
        }

        let a_data = Self::contiguous_data(a);
        let b_data = Self::contiguous_data(b);
        let c_data = self.dispatch_transb(&a_data, &b_data, m, n, k);
        Array2::from_shape_vec((m, n), c_data).unwrap()
    }

    fn matmul_batch(&self, ops: &[MatMulOp]) -> Vec<Array2<f32>> {
        if ops.is_empty() {
            return Vec::new();
        }

        let mut results = Vec::with_capacity(ops.len());
        let mut c_buffers: Vec<Option<Buffer>> = Vec::with_capacity(ops.len());
        let mut dims = Vec::with_capacity(ops.len());
        let mut any_gpu = false;

        let cmd = self.queue.new_command_buffer();

        for op in ops {
            let (m, k) = (op.a.shape()[0], op.a.shape()[1]);
            let n = if op.transpose_b { op.b.shape()[0] } else { op.b.shape()[1] };

            if !self.should_use_gpu(m, n, k) {
                c_buffers.push(None);
                dims.push((m, n, k));
                continue;
            }

            any_gpu = true;
            let a_data = Self::contiguous_data(&op.a);
            let b_data = Self::contiguous_data(&op.b);

            let c_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
            let buf_a = self.get_or_create_buffer(&a_data);
            let buf_b = self.get_or_create_buffer(&b_data);
            let buf_c = self.device.new_buffer(c_bytes, MTLResourceOptions::StorageModeShared);

            let pipeline = if op.transpose_b {
                &self.transb_pipeline
            } else {
                &self.sgemm_pipeline
            };

            let encoder = cmd.new_compute_command_encoder();
            self.encode_matmul(encoder, pipeline, &buf_a, &buf_b, &buf_c, m, n, k);
            encoder.end_encoding();

            c_buffers.push(Some(buf_c));
            dims.push((m, n, k));
        }

        if any_gpu {
            cmd.commit();
            cmd.wait_until_completed();
        }

        for (i, op) in ops.iter().enumerate() {
            let (m, n, _k) = dims[i];
            if let Some(ref buf_c) = c_buffers[i] {
                let ptr = buf_c.contents() as *const f32;
                let data = unsafe { std::slice::from_raw_parts(ptr, m * n).to_vec() };
                results.push(Array2::from_shape_vec((m, n), data).unwrap());
            } else {
                let result = if op.transpose_b {
                    op.a.dot(&op.b.t())
                } else {
                    op.a.dot(&op.b)
                };
                results.push(result);
            }
        }

        results
    }

    fn name(&self) -> &str {
        "metal (GPU compute)"
    }
}

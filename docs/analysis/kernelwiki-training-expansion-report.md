# KernelWiki Training Library Expansion Report: DeepSpeed

**Library**: Microsoft/DeepSpeed
**GitHub URL**: https://github.com/microsoft/DeepSpeed
**Local Source Path**: `/root/DeepSpeed`
**Analysis Date**: 2026-05-30
**Library Type**: **Full-Stack Training Framework** (all six dimensions apply equally)

---

## Dimension 1: Compute Kernels

### Kernel File Census

| Category | Directory | .cu Files | .cuh Files | Triton (.py) | C++ Bindings (.cpp) | Total |
|---|---|---|---|---|---|---|
| Fused Optimizers | `csrc/adam/` | 1 | 1 | 0 | 2 | 4 |
| Fused Optimizers | `csrc/lamb/` | 1 | 0 | 0 | 1 | 2 |
| Fused Optimizers | `csrc/lion/` | 1 | 1 | 0 | 2 | 4 |
| Fused Optimizers | `csrc/adagrad/` | 0 | 0 | 0 | 1 (CPU-only) | 1 |
| Transformer (training) | `csrc/transformer/` | 7 | 0 | 0 | 1 | 8 |
| Transformer (inference v1) | `csrc/transformer/inference/csrc/` | 9 | 0 | 0 | 1 | 10 |
| Inference v2 core_ops | `deepspeed/inference/v2/kernels/core_ops/` | 5 | 7 | 0 | 1 | 13 |
| Inference v2 ragged_ops | `deepspeed/inference/v2/kernels/ragged_ops/` | 5 | 5 | 0 | 1 | 11 |
| Inference v2 cutlass_ops | `deepspeed/inference/v2/kernels/cutlass_ops/` | 2 | 0 | 0 | 1 | 3 |
| Inference v2 ragged | `deepspeed/inference/v2/ragged/csrc/` | 1 | 0 | 0 | 1 | 2 |
| Quantization | `csrc/quantization/` | 5 | 0 | 0 | 1 | 6 |
| FP Quantizer | `csrc/fp_quantizer/` | 1 | 0 | 1 (Triton) | 1 | 3 |
| Random LTD | `csrc/random_ltd/` | 3 | 0 | 0 | 1 | 4 |
| Spatial | `csrc/spatial/csrc/` | 1 | 0 | 0 | 1 | 2 |
| DeepSpeed4Science | `csrc/deepspeed4science/evoformer_attn/` | 2 | 0 | 0 | 1 | 3 |
| DeepCompile | `csrc/compile/` | 0 | 0 | 0 | 1 (TORCH_LIBRARY) | 1 |
| Sparse Attention (Triton) | `deepspeed/ops/sparse_attention/` | 0 | 0 | 2 | 0 | 2 |
| Triton Inference | `deepspeed/ops/transformer/inference/triton/` | 0 | 0 | 7 | 0 | 7 |
| CPU Optimizers | `csrc/cpu/adam/`, `csrc/cpu/lion/` | 0 | 0 | 0 | 2 | 2 |
| Async I/O | `csrc/aio/` | 0 | 0 | 0 | 1 | 1 |
| **TOTALS** | | **46** | **15** | **10** | **29** | **88** |

**Summary**: 61 CUDA files (.cu + .cuh), 10 Triton kernel files (containing 17 `@triton.jit` kernels), 29 C++ binding entry points.

### Training-Specific Kernels

| Kernel | File Path | Proposed Tag | Description |
|---|---|---|---|
| AdamFunctor | `csrc/adam/multi_tensor_adam.cu:34` | `fused-adam` | Fused multi-tensor Adam/AdamW optimizer. Updates params, first moment, second moment in a single kernel launch. |
| lamb_cuda_kernel_part1/2/3 | `csrc/lamb/fused_lamb_cuda_kernel.cu:195,241,261` | `fused-lamb` | Three-phase fused LAMB optimizer: compute m,v updates + L2 norms, reduce norms, apply trust-ratio-clipped update. |
| LionFunctor | `csrc/lion/multi_tensor_lion.cu:29` | `fused-lion` | Fused multi-tensor Lion optimizer. Uses sign of interpolated momentum for update direction. |
| Adagrad_Optimizer::Step_1 | `csrc/adagrad/cpu_adagrad.cpp:17` | `cpu-adagrad` | CPU Adagrad optimizer with AVX-512/AVX-256 SIMD vectorization. |
| cpu_adam::ds_adam_step | `csrc/adam/cpu_adam.cpp` | `cpu-adam` | CPU Adam with SIMD (AVX-512/256). Used for ZeRO-Offload CPU parameter updates. |
| cpu_lion::ds_lion_step | `csrc/cpu/lion/fused_lion.cpp` | `cpu-lion` | CPU Lion optimizer with SIMD vectorization. |
| d_gelu_func | `csrc/transformer/gelu_kernels.cu:181` | `activation-backward` | Backward pass for GELU activation. Training-only. |
| softmax_backward_kernel | `csrc/transformer/softmax_kernels.cu:443` | `attention-backward` | Backward pass for attention softmax. Multiple template specializations. Training-only. |
| LayerNormBackward1 | `csrc/transformer/normalize_kernels.cu:628` | `layernorm-backward` | Computes gamma/beta gradients for LayerNorm backward. Uses tiled transpose with shared memory. |
| LayerNormBackward2 | `csrc/transformer/normalize_kernels.cu:761` | `layernorm-backward` | Computes input gradient for LayerNorm backward. Supports invertible and non-invertible modes. |
| LayerNormBackward1_fused_add | `csrc/transformer/normalize_kernels.cu:1377` | `layernorm-backward` | LayerNorm backward fusing two gradient streams from skip connections. Training-only. |
| LayerNormBackward2_fused_add | `csrc/transformer/normalize_kernels.cu:1499` | `layernorm-backward` | Input gradient for LayerNorm fused with residual gradient addition. Training-only. |
| dropout_kernel | `csrc/transformer/dropout_kernels.cu:10` | `dropout` | Forward dropout with Philox PRNG mask generation. Multiple overloads: basic, with bias, with bias+residual. Training-only. |
| dropout_kernel_bwd | `csrc/transformer/dropout_kernels.cu:159` | `dropout-backward` | Backward dropout applying saved mask with inverse scaling. Training-only. |
| BertTransformerLayer | `csrc/transformer/ds_transformer_cuda.cpp:54` | `fused-transformer-layer` | Complete fused BERT transformer layer with forward AND backward passes. |
| gather_tokens_impl | `csrc/random_ltd/gather_scatter.cu:16` | `random-ltd` | Token gather for Random-LTD (Layer Token Dropping). Training-only. |
| scatter_tokens_impl | `csrc/random_ltd/gather_scatter.cu` | `random-ltd` | Token scatter (inverse of gather) for Random-LTD backward. Training-only. |
| scan_sort | `csrc/random_ltd/token_sort.cu:28` | `random-ltd` | Scan-based parallel sort for Random-LTD token indices. Training-only. |
| slice_gpt_mask_impl | `csrc/random_ltd/slice_attn_masks.cu:12` | `random-ltd` | Slices attention masks for truncated sequence length. Training-only. |
| fake_quantize_kernel | `csrc/quantization/fake_quantizer.cu:12` | `fake-quantize` | Simulated quantization for quantization-aware training (QAT). Training-only. |
| attention_back_impl_template | `csrc/deepspeed4science/evoformer_attn/attention_back.cu:23` | `evoformer-attention` | Backward pass for Evoformer attention (AlphaFold2). Training-only. |
| dc::reduce_grad | `csrc/compile/init.cpp:25` | `deepcompile-op` | DeepCompile gradient reduction for ZeRO 1/2/3. Registered via TORCH_LIBRARY. |
| dc::end_backward | `csrc/compile/init.cpp:32` | `deepcompile-op` | DeepCompile backward pass completion synchronization. Training-only. |

### Inference-Shared Kernels (with training behavior differences)

| Kernel | File Path | Training Behavior Difference |
|---|---|---|
| gelu_kernel / fused_bias_gelu | `csrc/transformer/gelu_kernels.cu:43,99` | Training saves activations for backward (d_gelu). Inference v1 has separate copy at `csrc/transformer/inference/csrc/gelu.cu:36`. |
| attn_softmax | `csrc/transformer/softmax_kernels.cu:28` | Training additionally invokes `softmax_backward_kernel*`. Inference v1 has its own at `csrc/transformer/inference/csrc/softmax.cu`. |
| fused_bias_residual_layer_norm | `csrc/transformer/normalize_kernels.cu:21` | Training stores mean/variance for backward (`if (training) means[row] = mean`). Inference skips these stores. |
| Transpose_Kernel / transform_0213 | `csrc/transformer/transform_kernels.cu:12,60` | QKV reshape/transpose used in both. No behavioral difference. |
| cublas_gemm_ex | `csrc/transformer/cublas_wrappers.cu:24,252` | cuBLAS GEMM wrappers used for both forward+backward (training) and forward-only (inference). |
| cached_quantization | `csrc/quantization/quantize.cu:23` | INT4/INT8 quantization. Used in training (communication compression) and inference (weight quantization). |
| apply_quantization (FP quantizer) | `csrc/fp_quantizer/fp_quantize_impl.cu:68` | Training uses stochastic rounding; inference uses nearest rounding. |
| attention_impl_template (Evoformer) | `csrc/deepspeed4science/evoformer_attn/attention_cu.cu:19` | Forward Evoformer attention used in both training and inference. |

### Fused Kernels

| Kernel | Operations Fused | Estimated Memory Savings |
|---|---|---|
| FusedAdam (`multi_tensor_adam.cu`) | Gradient read + 1st moment + 2nd moment + bias correction + param update across multiple tensors | Eliminates 4 intermediate tensors per param group; ~20-30% bandwidth savings vs unfused |
| FusedLAMB (`fused_lamb_cuda_kernel.cu`) | Adam-like m/v update + L2 norm computation + trust ratio + param update (3-phase) | Eliminates 2 intermediate norm tensors; ~25% bandwidth savings |
| FusedLion (`multi_tensor_lion.cu`) | Momentum interpolation + sign extraction + weight decay + param update | Eliminates sign intermediate; ~15% bandwidth savings |
| fused_bias_gelu (`gelu_kernels.cu:99`) | Bias addition + GELU activation in single pass | Saves 1 intermediate tensor; ~50% bandwidth savings |
| fused_bias_residual_layer_norm (`normalize_kernels.cu:21`) | Residual addition + mean + variance + normalize + affine transform | Saves 3 intermediate tensors; ~60% memory savings |
| LayerNormBackward2_fused_add (`normalize_kernels.cu:1499`) | LayerNorm input-gradient + residual gradient add from skip connection | Saves 1 intermediate tensor, eliminates 1 kernel launch |
| attn_softmax (`softmax_kernels.cu:28`) | Attention mask addition + max-subtract + exp + sum + normalize | Saves 3 intermediate tensors; ~70% memory savings |
| dropout_kernel (residual variant, `dropout_kernels.cu:649`) | Bias add + dropout + mask generation + residual add | Saves 2 intermediate tensors |
| dequant_reduce (`quant_reduce.cu:21`) | Dequantize multiple INT8 tensors + sum-reduce + re-quantize | Avoids materializing N full-precision tensors during allreduce |
| matmul_kernel_fp8_bf16 (Triton, `fp8_gemm_triton.py:19`) | FP8 weight dequantization + GEMM | Avoids materializing full dequantized weight matrix |

### Kernel Dependency Graph

DeepSpeed writes its own CUDA kernels for activations, normalization, dropout, optimizers, quantization, and Random-LTD. For matrix multiplication (GEMM), it exclusively relies on upstream libraries:

| Provider Library | Kernel Types Provided | Import/Include Evidence |
|---|---|---|
| cuBLAS (NVIDIA) | GEMM, strided batched GEMM | `csrc/transformer/cublas_wrappers.cu` -- `cublasGemmEx`, `cublasGemmStridedBatchedEx` |
| CUTLASS (NVIDIA) | Mixed-precision GEMM (FP8/FP4), MoE GEMM | `deepspeed/inference/v2/kernels/cutlass_ops/` |
| cuRAND (NVIDIA) | Philox4_32_10 PRNG | `csrc/transformer/dropout_kernels.cu`, `csrc/fp_quantizer/fp_quantize_impl.cu` |
| Triton (OpenAI) | Block-sparse attention, inference matmul, FP8 dequant GEMM | `deepspeed/ops/sparse_attention/`, `deepspeed/ops/transformer/inference/triton/`, `deepspeed/ops/fp_quantizer/fp8_gemm_triton.py` |
| NVIDIA/apex | Multi-tensor apply infrastructure | `csrc/adam/multi_tensor_adam.cu` -- adapted from apex commit `a109f85` |
| xFormers/CUTLASS | Memory-efficient attention patterns | `csrc/deepspeed4science/evoformer_attn/` -- `kernel_forward.h`, `kernel_backward.h` |

### Proposed New kernel_types

| Tag | Representative File | Description |
|---|---|---|
| `fused-adam` | `csrc/adam/multi_tensor_adam.cu` | Fused multi-tensor Adam/AdamW GPU optimizer kernel |
| `fused-lamb` | `csrc/lamb/fused_lamb_cuda_kernel.cu` | Fused three-phase LAMB optimizer with trust-ratio |
| `fused-lion` | `csrc/lion/multi_tensor_lion.cu` | Fused multi-tensor Lion sign-based optimizer |
| `cpu-adam` | `csrc/adam/cpu_adam.cpp` | CPU Adam with AVX SIMD for ZeRO-Offload |
| `cpu-adagrad` | `csrc/adagrad/cpu_adagrad.cpp` | CPU Adagrad with AVX SIMD |
| `cpu-lion` | `csrc/cpu/lion/fused_lion.cpp` | CPU Lion with AVX SIMD |
| `activation-backward` | `csrc/transformer/gelu_kernels.cu:181` | GELU/activation backward pass derivative kernel |
| `attention-backward` | `csrc/transformer/softmax_kernels.cu:443` | Softmax backward pass for attention |
| `layernorm-backward` | `csrc/transformer/normalize_kernels.cu:628` | LayerNorm backward (gamma/beta grad + input grad) |
| `dropout` | `csrc/transformer/dropout_kernels.cu:10` | Dropout with Philox PRNG mask generation (fwd+bwd) |
| `fused-transformer-layer` | `csrc/transformer/ds_transformer_cuda.cpp:54` | Complete fused BERT transformer layer with fwd+bwd |
| `random-ltd` | `csrc/random_ltd/gather_scatter.cu` | Random Layer Token Dropping: gather, scatter, sort, mask slice |
| `fake-quantize` | `csrc/quantization/fake_quantizer.cu:12` | Simulated quantization for QAT |
| `quant-reduce` | `csrc/quantization/quant_reduce.cu:21` | Fused dequantize + multi-tensor reduce + requantize |
| `fp-quantize` | `csrc/fp_quantizer/fp_quantize_impl.cu:68` | FP8/FP6/FP4 floating-point quantization with stochastic rounding |
| `evoformer-attention` | `csrc/deepspeed4science/evoformer_attn/attention_cu.cu` | Evoformer attention for AlphaFold2 (fwd+bwd) |
| `deepcompile-op` | `csrc/compile/init.cpp` | DeepCompile graph-level ops for ZeRO compiled graphs |

---

## Dimension 2: Communication Kernels and Strategies

### Collective Operations

| Operation | API Function | Backend | Async Support | Primary Use Case |
|---|---|---|---|---|
| AllReduce | `dist.all_reduce()` | NCCL/RCCL/CCL/HCCL/Gloo | Yes | ZeRO-1 gradient averaging, norm computation |
| AllReduce (Coalesced) | `dist.all_reduce_coalesced()` | NCCL | Yes | Multi-tensor gradient reduction batching |
| AllReduce (Inference) | `dist.inference_all_reduce()` | NCCL/custom op | No | Inference-time TP allreduce |
| ReduceScatter | `dist.reduce_scatter_fn()` | NCCL | Yes | ZeRO-2/3 gradient partitioning |
| ReduceScatter (Coalesced) | `reduce_scatter_coalesced()` | NCCL | No | ZeRO-3 batched gradient reduce-scatter |
| AllGather | `dist.all_gather()` | NCCL | Yes | Parameter reconstruction |
| AllGather Into Tensor | `dist.all_gather_into_tensor()` | NCCL/SDMA(mori) | Yes | ZeRO-3 parameter prefetch |
| AllGather (Coalesced) | `all_gather_coalesced()` | NCCL | Yes | Batched parameter gathering |
| AllToAll | `dist.all_to_all_single()` | NCCL | Yes | Sequence parallelism, ZeRO++ quantized gradients |
| Reduce | `dist.reduce()` | NCCL | Yes | Targeted gradient reduction to specific rank |
| Broadcast | `dist.broadcast()` | NCCL | Yes | Parameter initialization, PP p2p fallback |
| Send/Recv | `dist.send()` / `dist.recv()` | NCCL | No | Pipeline parallel activation/gradient transfer |
| ISend/IRecv | `dist.isend()` / `dist.irecv()` | NCCL | Yes (inherent) | 1-bit compressed allreduce gather phase |
| Barrier | `dist.barrier()` | NCCL | Yes | Synchronization points |
| Monitored Barrier | `dist.monitored_barrier()` | NCCL | No | Debugging hangs with timeout |

**Source files**: `deepspeed/comm/comm.py` (API layer), `deepspeed/comm/torch.py` (TorchBackend implementation)

### Communication Backend Architecture

```
Backend (base class, deepspeed/comm/backend.py)
  |-- TorchBackend (default, wraps torch.distributed)
  |     |-- CCLBackend (Intel oneCCL, extends TorchBackend)
  |     |-- mori SDMA backend (plugged into TorchBackend.all_gather_into_tensor)
  |-- NcclBackend (runtime/comm/nccl.py -- 1-bit compressed allreduce, uses cupy)
  |-- CompressedBackend (runtime/comm/compressed.py -- packbits-based)
  |-- HcclBackend (runtime/comm/hccl.py -- Huawei Ascend)
```

Supported named backends: `nccl`, `ccl`, `mpi`, `gloo`, `sccl`, `hccl`.

### Communication-Compute Overlap Patterns

| Pattern | Mechanism | Where Used | Config Parameter |
|---|---|---|---|
| Gradient Reduce + Backward Overlap | Separate `reduction_stream` runs allreduce/reduce-scatter concurrently with backward compute | ZeRO-1/2 (`stage_1_and_2.py:516,1231-1237`) | `overlap_comm: true` |
| AllGather Prefetch + Compute Overlap | `PartitionedParameterCoordinator` kicks off allgather for next N modules while current module executes | ZeRO-3 (`partitioned_param_coordinator.py:427-461`) | `prefetch_bucket_size` |
| Domino Async TP AllReduce | `dist.all_reduce(grad_input, async_op=True)` during backward; handle stored for later wait | Domino (`domino/async_linear.py:35`) | N/A (structural) |
| ZeRO-3 Reduce+Partition Stream | `reduce_and_partition_stream` runs gradient copy+reduce while backward continues | ZeRO-3 (`parameter_offload.py:172`) | `overlap_comm: true` |
| Double-Buffered IPG Buckets | When `overlap_comm=True` + `contiguous_gradients`, ZeRO-1/2 uses two buffers alternating between fill and reduce | ZeRO-1/2 (`stage_1_and_2.py:1099-1101`) | `overlap_comm + contiguous_gradients` |
| SDMA Resource Overlap (AMD) | AllGather routed through SDMA copy engines (hardware DMA), leaving CUs free for compute | ZeRO-3 allgather (`comm/mori.py`) | `DS_SDMA_ALLGATHER=1` env var |
| DeepCompile Double-Buffered Reduce | `DoubleBufferedReduceBucket` swaps current/shadow buffers; copy and reduce-scatter on separate streams | DeepCompile (`csrc/includes/deepcompile.h:179-250`) | DeepCompile config |
| Pipeline 1F1B Schedule | Interleaved forward/backward micro-batches; send/recv overlap with compute of other micro-batches | Pipeline engine (`pipe/schedule.py:189-298`) | Pipeline schedule |
| NVMe Swap + Compute Overlap | `AsyncPartitionedParameterSwapper` prefetches params from NVMe while compute runs | ZeRO-Infinity | `aio.overlap_events` |
| Coalesced Grad Reduction | `engine.coalesce_grad_reduction()` defers all gradient reduction across multiple backward passes | ZeRO-1/2/3 (`engine.py:2569-2663`) | API context manager |

### ZeRO Stage Communication Patterns

#### ZeRO-1 (Optimizer State Partitioning)

| Phase | Collective | Data |
|---|---|---|
| Backward (gradient sync) | AllReduce (default) or ReduceScatter + AllGather | Full gradients per bucket |
| Step (weight sync) | AllGather | Updated fp16 weight partitions |

#### ZeRO-2 (Gradient + Optimizer State Partitioning)

| Phase | Collective | Data |
|---|---|---|
| Backward (gradient partition) | ReduceScatter (default) | Gradient buckets -> each rank gets its partition |
| Step (weight sync) | AllGather | Updated fp16 weight partitions |

#### ZeRO-3 (Full Partitioning: Parameters + Gradients + Optimizer States)

| Phase | Collective | Data |
|---|---|---|
| Forward (param gather) | AllGather (async, prefetch) | Partitioned parameters reconstructed per-module |
| Backward (param gather) | AllGather (async) | Partitioned parameters for backward |
| Backward (gradient reduce) | ReduceScatter (coalesced) | Gradient buckets -> each rank gets partition |
| Backward (ZeRO++ quantized) | All-to-All (INT4, 2-stage hierarchical) | Quantized gradients: intra-node a2a + inter-node a2a |

### Advanced Communication Features Checklist

| Feature | Status | Evidence |
|---|---|---|
| NCCL/RCCL integration | Production | Via `torch.distributed` |
| Intel oneCCL | Production | `CCLBackend` in `deepspeed/comm/ccl.py` |
| HCCL (Huawei Ascend) | Experimental | `deepspeed/runtime/comm/hccl.py` |
| Shared-memory intra-node allreduce | Production | `ShareMemCommBuilder` in `csrc/cpu/comm/shm.h` |
| SDMA AllGather (AMD MI300) | Production (opt-in) | `mori` backend via `DS_SDMA_ALLGATHER=1` |
| Symmetric Memory (torch) | Experimental | `enable_symm_mem_for_group` in DeepCompile (`compile/config.py:33`) |
| NCCL direct calls (DeepCompile) | Production | `ncclComm_t` in `deepcompile.h` |
| Communication profiling | Production | `CommsLogger` with per-op timing, throughput, busbw |
| Communication kill switches | Production | Per-op `DS_COMM_*_OFF` flags |
| torch.compile compatibility | Production | `@disable_compiler_collective` decorator |
| Device Mesh | Production (torch>=2.2) | `init_device_mesh` for multi-dim parallelism |
| Quantized weight communication (ZeRO++) | Production | `zero_quantized_weights`: quantize params for allgather |
| Quantized gradient communication (ZeRO++) | Production | `zero_quantized_gradients`: INT4 hierarchical all-to-all |
| LoCo error compensation | Production | `zeropp_loco_param`: error-feedback quantized gradients |
| HPZ (Hierarchical Param Partition) | Production | `zero_hpz_partition_size` |
| Coalesced gradient reduction | Production | `engine.coalesce_grad_reduction()` |
| Zero-SM / Copy Engine communication | Not implemented | No `ZERO_CTA` / `copy_engine` patterns |
| MSCCL / NVSHMEM | Not integrated | No references found |
| GPU-initiated communication (GIN) | Not implemented | No device-side patterns |
| Async Tensor Parallel (Domino) | Production | `DominoAsyncColumnParallelLinear` |

### Proposed New Communication kernel_types

| Tag | Description |
|---|---|
| `allreduce` | NCCL AllReduce gradient synchronization via torch.distributed |
| `reduce-scatter` | NCCL ReduceScatter for ZeRO-2/3 gradient partitioning |
| `allgather` | NCCL AllGather for parameter reconstruction |
| `all-to-all` | NCCL All-to-All for sequence parallelism and ZeRO++ |
| `p2p-send-recv` | Point-to-point Send/Recv for pipeline parallel |
| `quantized-allreduce` | 1-bit compressed AllReduce with error feedback |
| `quantized-reduce-scatter` | INT4 hierarchical All-to-All gradient reduction (ZeRO++) |
| `sdma-allgather` | AMD SDMA copy-engine-based AllGather (mori) |

### Proposed New Communication techniques

| Tag | Evidence | Description |
|---|---|---|
| `compute-comm-overlap` | `stage_1_and_2.py` overlap_comm | Backward ReduceScatter overlapped with gradient computation |
| `double-buffered-ipg` | `stage_1_and_2.py:1099` | Double-buffered Independent Partition Gradient buckets |
| `trace-based-prefetch` | `partitioned_param_coordinator.py` | Record-replay parameter access order for async allgather |
| `gradient-bucketing` | `stage_1_and_2.py` reduce_bucket_size | Batching gradients into configurable buckets for collective ops |
| `hierarchical-quantized-reduction` | `coalesced_collectives.py` | 2-stage intra-node/inter-node INT4 quantized all-to-all |
| `loco-error-feedback` | `coalesced_collectives.py` | Low-Communication quantized gradients with EMA error compensation |
| `domino-async-tp` | `domino/async_linear.py` | Dual micro-batch interleaved async TP AllReduce overlap |
| `sdma-offload` | `comm/mori.py` | SDMA copy engine communication offload for zero-CU data transfer |

---

## Dimension 3: Parallelism Strategies

### Supported Parallelism Dimensions

| Dimension | Abbrev. | Supported | Key Implementation Files | Communication Pattern Triggered |
|---|---|---|---|---|
| Data Parallelism (ZeRO) | DP/ZeRO | Stages 0-3 | `runtime/zero/stage_1_and_2.py`, `runtime/zero/stage3.py` | AllReduce (Stage 0-1), ReduceScatter + AllGather (Stage 2-3), All-to-All (ZeRO++) |
| Pipeline Parallelism | PP | Yes | `runtime/pipe/engine.py`, `runtime/pipe/schedule.py`, `runtime/pipe/topology.py` | P2P Send/Recv (activations fwd, gradients bwd) |
| Tensor Parallelism | TP | Yes (training + inference) | `runtime/tensor_parallel/tp_manager.py`, `module_inject/auto_tp.py`, `runtime/domino/` | AllReduce (row-parallel), identity/scatter (column-parallel) |
| Sequence Parallelism | SP | Yes (Ulysses + FPDT) | `sequence/layer.py`, `sequence/fpdt_layer.py`, `sequence/auto_sp.py` | All-to-All (head/seq redistribution) |
| Expert Parallelism | EP | Yes | `moe/layer.py`, `moe/sharded_moe.py`, `moe/mappings.py` | All-to-All (token dispatch + combine) |
| Context Parallelism | CP | Not natively | N/A | N/A (SP serves this role in DeepSpeed) |
| MiCS | MiCS | Yes | `runtime/zero/mics.py`, `runtime/zero/mics_utils.py` | Hierarchical AllGather (intra-node + inter-node) |

### ZeRO Stages Detail

| Aspect | ZeRO-0 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|---|---|---|---|---|
| Optimizer states | Replicated | Partitioned | Partitioned | Partitioned |
| Gradients | Replicated | Replicated | Partitioned | Partitioned |
| Parameters | Replicated | Replicated | Replicated | Partitioned (on-demand gather) |
| Primary comm (fwd) | None | None | None | AllGather per layer |
| Primary comm (bwd) | AllReduce | AllReduce | ReduceScatter | ReduceScatter |
| Primary comm (step) | None | AllGather (opt states) | AllGather (params) | None (re-gathered on next fwd) |
| CPU/NVMe offload | No | Optimizer states | Optimizer states | Params + optimizer + NVMe |
| PP compatible | Yes | Yes | No | No |

**Key ZeRO-3 design decisions**:
- **Sub-group tiling** (`sub_group_size` default 1e9): Processes optimizer states in chunks, enabling trillion-parameter models
- **Trace-based prefetching**: Records parameter access order during first iteration, uses it for async allgather overlap
- **Param persistence threshold** (default 1e5): Small parameters stay un-partitioned to avoid latency-dominated communication

**ZeRO++ enhancements**:
- `zero_quantized_weights`: INT4/INT8 quantized AllGather (~4x bandwidth reduction)
- `zero_quantized_gradients`: Hierarchical INT4 All-to-All gradient reduction
- `zero_hpz_partition_size`: Hierarchical Parameter ZeRO (secondary partition group)
- `zeropp_loco_param`: LoCo error-feedback quantized gradients (EMA error buffers, periodic reset)

### Pipeline Scheduling Strategies

| Schedule | Strategy | Buffers | Bubble Rate | Source |
|---|---|---|---|---|
| TrainSchedule | Synchronous 1F1B | min(stages - stage_id, micro_batches) | ~P/M | `pipe/schedule.py:189-298` |
| InferenceSchedule | Fill-drain (forward-only) | 2 | N/A | `pipe/schedule.py` |
| DataParallelSchedule | Sequential fwd-bwd (no pipeline) | 1 | N/A | `pipe/schedule.py` |

Pipeline instructions: `SendActivation`, `RecvActivation`, `SendGrad`, `RecvGrad`, `ReduceGrads`, `ReduceTiedGrads`, `OptimizerStep`.

**Composition constraint** (`pipe/engine.py:76-77`): ZeRO-2 and ZeRO-3 are explicitly incompatible with PP. Only ZeRO-0 and ZeRO-1 work with pipeline parallelism.

### Tensor Parallelism

**Training TP** (`runtime/tensor_parallel/tp_manager.py`):
- `TpTrainingManager` initializes TP groups via `_init_tp_mesh_device()` creating a `(data_parallel, tensor_parallel)` mesh
- `AutoTP.tp_parser()` auto-identifies linear layers: `LinearAllreduce` (row-parallel), `LinearLayer` (column-parallel), `GateUpPack_LinearLayer` (fused gate-up), `fused_LinearLayer` (fused QKV)
- Preset models: llama, bloom, chatglm, mixtral, deepseek_v2, qwen2, phi3

### Domino: Asynchronous Intra-Layer Communication-Computation Overlap

**Location**: `runtime/domino/async_linear.py`, `runtime/domino/transformer.py`

Domino overlaps TP communication with computation by processing two micro-batches simultaneously:
1. `DominoAsyncColumnParallelLinear`: Launches async AllReduce on `grad_input` immediately in backward
2. `DominoTransformerLayer.forward(hidden_states0, hidden_states1)`: Interleaves MB0 attention -> async AllReduce -> MB1 attention -> wait MB0 -> MB0 MLP -> wait MB1 -> MB1 MLP

### Sequence Parallelism

**DeepSpeed-Ulysses** (`sequence/layer.py`): `DistributedAttention` uses 4 All-to-All operations per attention layer (3 for Q/K/V scatter, 1 for output gather). Transposes from sequence-partitioned `[s/P, h]` to head-partitioned `[s, h/P]`.

**FPDT** (`sequence/fpdt_layer.py`): Chunked attention with Flash Attention for extremely long sequences. All-to-All per chunk with online softmax accumulation across chunks. Optional CPU offload of KV between chunks.

**AutoSP** (`sequence/auto_sp.py`): One-call SP wrapping for multimodal models (LLaVA, InternVL, Qwen2-VL).

### MoE Expert Parallelism

- `num_experts` partitioned across `ep_size` ranks
- Gate selects top-k experts (k=1 or k=2) with Jitter/RSample/None routing
- Communication: 2x All-to-All per layer (dispatch + combine)
- Supports combined EP+TP via `enable_expert_tensor_parallelism`

### ZenFlow: Selective Gradient Update Optimization

Selects top-k important gradient columns (`topk_ratio`, default 10%) for frequent updates; accumulates unimportant gradients for delayed update at `update_interval`. Reduces effective optimizer computation and CPU-GPU communication for ZeRO-Offload.

### Composable Multi-Dimensional Parallelism

| Combination | Supported | Notes |
|---|---|---|
| ZeRO-0/1 + PP | Yes | Only ZeRO stages below `gradients` |
| ZeRO-2/3 + PP | No | Explicitly asserted incompatible |
| ZeRO + TP | Yes | Via `mpu` integration |
| ZeRO + EP | Yes | Separate process groups for expert vs. non-expert params |
| ZeRO + SP (Ulysses) | Yes | `seq_data_parallel_group` combines SP and DP |
| PP + TP | Yes | Via `PipeModelDataParallelTopology` 3D grid |
| TP + EP | Yes | `enable_expert_tensor_parallelism` flag |
| TP + Domino | Native | Domino is specifically designed for TP overlap |
| ZeRO + ZenFlow | Yes | ZenFlow extends ZeRO-1/2/3 optimizers |
| ZeRO-3 + DeepCompile | Yes | Compiled AllGather scheduling |

### Communication Pattern Summary per Dimension

| Parallelism | Forward Comm | Backward Comm | Optimizer Step Comm |
|---|---|---|---|
| ZeRO-0 | None | AllReduce (gradients) | None |
| ZeRO-1 | None | AllReduce (gradients) | AllGather (updated params) |
| ZeRO-2 | None | ReduceScatter (gradients) | AllGather (updated params) |
| ZeRO-3 | AllGather (params, per-layer) | ReduceScatter (gradients) + Release (params) | None (re-gathered on next fwd) |
| PP | P2P Send/Recv (activations) | P2P Send/Recv (gradients) | AllReduce (DP within stage) |
| TP | None (column) / AllReduce (row) | AllReduce (column) / None (row) | None |
| Domino TP | Async AllReduce (interleaved 2 MBs) | Async AllReduce (interleaved) | None |
| SP (Ulysses) | 3x All-to-All (Q,K,V) + 1x All-to-All (output) | Reverse All-to-All | None |
| EP (MoE) | 2x All-to-All (dispatch + combine) | 2x All-to-All (reverse) | AllReduce (expert DP group) |
| MiCS | Hierarchical AllGather (intra + inter node) | ReduceScatter (shard group) | Hierarchical AllGather |

### Proposed New Parallelism techniques

| Tag | Evidence | Description |
|---|---|---|
| `zero-stage-1` | `stage_1_and_2.py` | Optimizer state partitioning across DP ranks |
| `zero-stage-2` | `stage_1_and_2.py` | Gradient + optimizer state partitioning |
| `zero-stage-3` | `stage3.py` | Full parameter + gradient + optimizer partitioning with on-demand gather |
| `zero-plus-plus` | `config.py`, `coalesced_collectives.py` | Quantized hierarchical communication for ZeRO-3 |
| `pipeline-1f1b` | `pipe/schedule.py` | Synchronous 1-forward-1-backward interleaved pipeline |
| `auto-tp` | `module_inject/auto_tp.py` | Automatic tensor parallelism via model structure analysis |
| `domino-async-tp` | `domino/async_linear.py` | Dual micro-batch async TP overlap |
| `ulysses-sp` | `sequence/layer.py` | All-to-All based head/sequence redistribution |
| `fpdt-sp` | `sequence/fpdt_layer.py` | Chunked Flash Attention with tiled sequence distribution |
| `expert-parallel` | `moe/sharded_moe.py` | MoE token dispatch via All-to-All |
| `mics` | `runtime/zero/mics.py` | Minimum Communication Sharding with node-aware groups |
| `zenflow-selective` | `runtime/zenflow/` | Top-k selective gradient update optimization |
| `deepcompile-z3` | `compile/` | Compiled ZeRO-3 with optimized AllGather scheduling |
| `3d-topology` | `pipe/topology.py` | pipe x data x model process topology |

---

## Dimension 4: Memory Management

### Memory Component Analysis

| Component | Data Type | Bytes/Param | Sharding (ZeRO-3) | Communication Triggered |
|---|---|---|---|---|
| Parameters (FP16/BF16) | float16/bfloat16 | 2 | Partitioned | AllGather before each layer's forward |
| Parameters (FP32 master) | float32 | 4 | Partitioned | None (local optimizer step) |
| Gradients (FP16/BF16) | float16/bfloat16 | 2 | Partitioned | ReduceScatter after each layer's backward |
| Gradients (FP32 accum) | float32 | 4 | Local | None |
| Adam Momentum | float32 | 4 | Partitioned | None |
| Adam Variance | float32 | 4 | Partitioned | None |
| Activations | varies | varies | Selectively recomputed | None (local) |
| Communication Buffers | varies | configurable | N/A | N/A |

**Total per-parameter cost (Adam, mixed precision, no ZeRO):** ~20 bytes/param (2 param_fp16 + 4 master_fp32 + 2 grad_fp16 + 4 grad_fp32_accum + 4 momentum + 4 variance).

### ZeRO Stages Memory Optimization

| ZeRO Stage | What is Partitioned | Per-GPU Memory (N GPUs, Adam) |
|---|---|---|
| Stage 0 (Disabled) | Nothing | 18P bytes |
| Stage 1 (optimizer_states) | Optimizer states | 2P + 2P + 4P + 8P/N bytes |
| Stage 2 (gradients) | Optimizer states + Gradients | 2P + 2P/N + 4P/N + 8P/N bytes |
| Stage 3 (weights) | Everything | (2+4+8)P/N + largest_layer bytes |

### Activation Checkpointing Strategies

| Strategy | Config Flag | Memory Saved | Recompute Cost | Description |
|---|---|---|---|---|
| No checkpointing | (default) | None | None | All activations retained |
| Full recompute | `checkpoint()` by user | ~O(sqrt(L)) | One extra fwd per checkpoint segment | Standard gradient checkpointing via `CheckpointFunction` |
| Partitioned activations | `partition_activations=True` | Activations/MP_size | AllGather during backward | Activations partitioned across model-parallel GPUs |
| CPU checkpointing | `cpu_checkpointing=True` | GPU memory freed | CPU-to-GPU transfer during backward | Activations moved to CPU pinned memory |
| Contiguous checkpointing | `contiguous_memory_optimization=True` | Reduces fragmentation | Requires `partition_activations` | Pre-allocated contiguous buffers |
| Non-reentrant | Automatic (torch >= 2.0) | Same as full recompute | Same | Via `torch.autograd.graph.saved_tensors_hooks` |

### CPU Offload (ZeRO-Offload)

- **Optimizer offload (Stages 1/2/3)**: Optimizer states stored in CPU memory. `DeepSpeedCPUAdam` performs optimizer step on CPU with AVX SIMD. Gradients async-copied via `async_inplace_copy_grad_to_fp32_buffer_from_gpu()`.
- **Parameter offload (Stage 3 only)**: Model parameters on CPU; allgathered to GPU per-module. `max_in_cpu` controls overflow to NVMe.
- **Partial offload (`offload_ratio`)**: Hybrid GPU/CPU placement at sub-group granularity. Sub-groups mapped to CPU/GPU via `subgroup_to_device`.
- **Round-robin gradients**: Fine-grained gradient partitioning for parallel CPU memory copies across ranks.

### NVMe Offload (ZeRO-Infinity)

- **AIO engine** (`deepspeed/runtime/swap_tensor/`): Uses Linux AIO (`libaio`) or GPUDirect Storage (GDS) for direct NVMe-to-GPU transfers
- **Parameter swapping**: `AsyncPartitionedParameterSwapper` with pool of pinned buffers for async swap-in
- **Optimizer state swapping**: `PipelinedOptimizerSwapper` overlaps read of next tile with computation of current tile
- **Config**: `block_size` (1MB), `queue_depth` (8), `overlap_events` (True), `use_gds` (False)

### SuperOffload

Designed for NVIDIA Grace Hopper superchips with high CPU-GPU bandwidth:
- Runs CPU Adam optimizer in a **separate process** via `torch.multiprocessing`
- CPU affinity explicitly partitioned: `cpuadam_cores_perc` (80%) to optimizer, rest to PyTorch
- Async pipeline: fires CPU Adam step during `partition_grads()` in backward, polls results asynchronously
- Supports overflow rollback via `_handle_overflow_rollback()`

### Gradient Compression (ZeRO++)

- `all_to_all_quant_reduce()`: 2-stage INT4 hierarchical reduction (intra-node a2a + inter-node a2a), ~4x bandwidth reduction
- `all_to_all_loco_quant_reduce()`: Error-feedback variant with EMA error buffers (`err_beta=0.8`, `reset_T=1024`)
- Falls back to `reduce_scatter_coalesced` for 1D or non-divisible tensors

### Memory Estimation (Reference: 7B model, FP16+Adam, 8 GPUs)

| Configuration | Per-GPU Memory |
|---|---|
| ZeRO-0 (no sharding) | ~130 GB |
| ZeRO-1 (8 GPUs) | ~28 GB |
| ZeRO-2 (8 GPUs) | ~18 GB |
| ZeRO-3 (8 GPUs) | ~17 GB |
| ZeRO-3 + CPU offload (all) | ~0.5 GB (largest layer) |

### Proposed New Memory-related techniques

| Tag | Description | Source |
|---|---|---|
| `zero-offload-cpu` | CPU offloading for optimizer states and parameters | `offload_config.py` |
| `zero-infinity-nvme` | NVMe offloading with async I/O for parameters and optimizer states | `swap_tensor/` |
| `superoffload-async-cpu` | Separate-process CPU Adam for superchips | `superoffload_utils.py` |
| `pipelined-nvme-io` | Overlapped read/write for optimizer state NVMe tiles | `pipelined_optimizer_swapper.py` |
| `activation-checkpoint` | Gradient checkpointing (reentrant and non-reentrant) | `checkpointing.py` |
| `activation-partition` | Activation partitioning across model-parallel GPUs | `checkpointing.py` |
| `activation-cpu-offload` | Activation offloading to CPU pinned memory | `checkpointing.py` |
| `contiguous-grad-buffer` | Pre-allocated buffer to avoid gradient fragmentation | `contiguous_gradients=True` |
| `random-ltd-token-drop` | Layer token dropping for activation memory reduction | `random_ltd/dropping_utils.py` |
| `gds-direct-storage` | GPUDirect Storage for NVMe-to-GPU DMA | `aio_config.py` |
| `defrag-contiguous-alloc` | Contiguous memory allocator with defragmentation | `contiguous_memory_allocator.py` |

---

## Dimension 5: Precision Management

### Supported Precision Modes

| Mode | Config Key | Master Weights | Loss Scaling | Source File(s) |
|---|---|---|---|---|
| FP32 (baseline) | None (default) | N/A (native FP32) | None | `engine.py` |
| FP16 mixed precision | `"fp16": {"enabled": true}` | FP32 master weights | Dynamic or static | `runtime/fp16/fused_optimizer.py` |
| BF16 mixed precision | `"bf16": {"enabled": true}` | FP32 partitioned | None (not needed) | `runtime/bf16_optimizer.py` |
| FP16 + ZeRO (1/2/3) | `"fp16" + "zero_optimization"` | FP32 partitioned | Dynamic loss scaling | `runtime/zero/stage_1_and_2.py`, `stage3.py` |
| BF16 + ZeRO 1 | `"bf16" + "zero_optimization"` | FP32 partitioned | None | `runtime/bf16_optimizer.py` |
| AMP (NVIDIA Apex) | `"amp": {"enabled": true}` | Managed by Apex | Managed by Apex | `engine.py:1629-1633` |

### FP16 Dynamic Loss Scaling

| Parameter | Config Key | Default | Description |
|---|---|---|---|
| Initial scale | `initial_scale_power` | 16 (2^16 = 65536) | Starting loss scale as power of 2 |
| Scale window | `loss_scale_window` | 1000 | Consecutive stable iterations before scale-up |
| Hysteresis | `hysteresis` | 2 | Overflow must recur before scale-down |
| Min loss scale | `min_loss_scale` | 1 | Floor for dynamic scale |
| Static scale | `loss_scale` | 0 (means dynamic) | Non-zero enables static scaling |

Loss scaling is **only** applied for `torch.float16`. For BF16, scale is set to 1.0 because BF16's dynamic range matches FP32.

### BF16 Training

- `BF16_Optimizer` extends `ZeROOptimizer`, maintains `fp32_groups_flat_partition` as FP32 master weight partition
- Configurable `bf16_master_weights_and_grads` and `bf16_optimizer_states` for reduced memory
- `immediate_grad_update` mode: backward hooks for inline HP gradient accumulation
- No loss scaling whatsoever

### FP8 Support Assessment

**Status: Partial -- weight quantization only, no native FP8 training loop.**

| Feature | Status |
|---|---|
| FP8 training config (`"fp8": {"enabled": true}`) | Not implemented |
| FP8 format enums (E4M3/E5M2) | Not present |
| Per-tensor FP8 scaling (amax history) | Not implemented |
| FP8 weight quantization | Implemented (FP_Quantize class) |
| FP8 GEMM (weight-only dequant) | Implemented (Triton `fp8_gemm_triton.py`) |
| MXFP8 / Blackwell block scaling | Not implemented |
| FP8 gradient communication | Not implemented |

### FP Quantizer (`csrc/fp_quantizer/`)

Custom floating-point quantization engine for weight compression:

| Precision | Bits | Mantissa | Exponent | Dynamic Range |
|---|---|---|---|---|
| FP8 (E4M3-like) | 8 | 3 | 4 | 480.0 |
| FP8 (E5M2-like) | 8 | 2 | 5 | 114688.0 |
| FP12 | 12 | 4 | 4 | 510.0 |
| FP6 | 6 | 2 | 3 | 28.0 |
| FP4 | 4 | 1 | 2 | 6.0 |

Features: Group quantization (`group_size=512`), stochastic rounding, selective dequantization.

### Quantization Support (INT8/INT4)

CUDA kernels in `csrc/quantization/`:
- `fake_quantize` / `sr_fake_quantize`: Simulated quantization for QAT (symmetric/asymmetric)
- `quantize`: Real INT8/INT4 quantization with scale+offset
- `dequantize`: INT8/INT4 dequantization
- `swizzled_quant`: Swizzled layout for ZeRO++ all-to-all
- `quant_reduce`: Fused dequantize + reduce for quantized communication

### Precision per Training Component

| Component | Forward | Backward | Optimizer Step | Communication |
|---|---|---|---|---|
| FP16 mode | FP16 | FP16 (grads) | FP32 master | Configurable (FP16/BF16/FP32) |
| BF16 mode | BF16 | BF16 grads, FP32 accum | FP32 (or BF16 opt-in) | Configurable |
| ZeRO++ (qgZ) | Same as ZeRO-3 | INT4 quantized a2a + dequant | FP32 | INT4 (swizzled) |

### Compression/Quantization in Training

**Weight Quantization (QAT)**: Symmetric/asymmetric, configurable `start_bits` -> `target_bits` with `quantization_period` for gradual precision reduction. STE training via `quantize_weight_in_forward`.

**1-Bit Communication** (`runtime/fp16/onebit/`):
- `OnebitAdam`: 1-bit compressed Adam with warmup phase (`freeze_step` uncompressed, then 1-bit)
- `OnebitLamb`: Same for LAMB optimizer
- Error feedback via cupy `packbits`

**Pruning methods**: Sparse (L1/TopK/SNIP-Momentum), Row, Head, Channel, Layer reduction.

### FP8 Communication Integration Checklist

- [ ] FP8 forward pass (native): Not implemented
- [ ] FP8 backward pass: Not implemented
- [ ] FP8 weight storage during training: Partial (`deepspeed/linear/quantization.py`)
- [ ] FP8 gradient communication: Not implemented (limited to FP16/BF16/FP32)
- [ ] FP8 AllReduce/ReduceScatter: Not implemented
- [ ] Per-tensor FP8 scaling (amax): Not implemented
- [ ] MXFP8 / Blackwell block scaling: Not implemented
- [ ] FP8 GEMM (weight-only dequant): Implemented (`fp8_gemm_triton.py`)

### Proposed New Precision techniques

| Tag | Description |
|---|---|
| `dynamic-loss-scaling` | FP16 dynamic loss scaling with hysteresis and scale window |
| `bf16-mixed-precision` | BF16 mixed precision training with FP32 master weights |
| `fp-quantize` | FP8/FP6/FP4 floating-point quantization with stochastic rounding |
| `fake-quantize-qat` | Simulated quantization for quantization-aware training |
| `onebit-compressed-comm` | 1-bit sign compression with error feedback for Adam/LAMB |
| `quantized-weight-comm` | INT4/INT8 quantized AllGather for weight communication |
| `quantized-gradient-comm` | INT4 hierarchical All-to-All for gradient communication |
| `gradient-predivide` | Pre-dividing gradients before allreduce for overflow prevention |

---

## Dimension 6: Profiling and Observability

### Built-in Profiling Capabilities

| Capability | Module | Description |
|---|---|---|
| FLOPs Profiler | `profiling/flops_profiler/profiler.py` | Theoretical FLOPs, MACs, params, latency estimation per module via monkey-patching |
| Wall Clock Breakdown | `utils/timer.py` | `SynchronizedWallClockTimer` with CUDA events for fwd/bwd/step phases |
| Communication Logger | `utils/comms_logging.py` | Per-collective latency, algbw, busbw with straggler analysis |
| Compile Graph Profiler | `compile/profilers/graph_profile.py` | Per-FX-node device_time, wall_time, memory tracking |
| Compile Comm Profiler | `compile/profilers/comm_profile.py` | AllGather latency predictor via scipy interpolation |
| Partitioned Param Profiler | `runtime/zero/partitioned_param_profiler.py` | Gather/release event counts and timing for ZeRO-3 |
| Throughput Timer | `utils/timer.py` | Running average samples/sec with memory monitoring |
| NVTX Annotations | `utils/nvtx.py` | `@instrument_w_nvtx` decorator with "DeepSpeed" domain markers |
| Memory Usage Reporter | `runtime/utils.py:815` | GPU allocated/cached/peak and CPU virtual memory |
| MoE Forward Breakdown | `runtime/engine.py:2408` | Gate time, all-to-all time within forward pass |
| Autotuner | `autotuning/autotuner.py` | Grid/random/XGBoost-based search over ZeRO stages, batch sizes, knobs |
| Monitoring Backends | `monitor/` | TensorBoard, WandB, CSV, Comet dispatch |

### FLOPs Profiler Details

The `FlopsProfiler` instruments models via:
1. Monkey-patching `torch.nn.functional` operations (linear, conv, activations, norms, attention)
2. Monkey-patching `torch.Tensor` methods (matmul, mm, bmm, addmm, einsum)
3. Forward hooks on `nn.Module` subclasses (RNN, LSTM, GRU)

Metrics: FLOPs per GPU, MACs, parameters, forward latency, FLOPS throughput. Supports `recompute_fwd_factor` for activation recomputation. MoE-aware parameter counting.

Config: `"flops_profiler": {"enabled": true, "profile_step": N, "module_depth": M, "detailed": true}`

### Monitoring Backends

| Backend | Class | Output |
|---|---|---|
| TensorBoard | `TensorBoardMonitor` | Event files via `SummaryWriter` |
| Weights & Biases | `WandbMonitor` | WandB cloud logs |
| CSV | `csvMonitor` | Per-metric CSV files |
| Comet ML | `CometMonitor` | Comet experiment metrics |

`MonitorMaster` coordinates all backends, writing from rank 0 only.

### Wall Clock Breakdown

Timer categories (via `EngineTimers`):

| Timer | Scope |
|---|---|
| `fwd_microstep` / `fwd` | Forward pass (micro / global) |
| `bwd_microstep` / `bwd` | Full backward (micro / global) |
| `bwd_inner_microstep` / `bwd_inner` | Backward computation only |
| `bwd_allreduce_microstep` / `bwd_allreduce` | Gradient allreduce |
| `step_microstep` / `step` | Optimizer step |

Activated by `"wall_clock_breakdown": true`.

### Communication Profiling (CommsLogger)

`timed_op` decorator wraps every distributed collective. Records per-operation, per-message-size statistics:
- Latency, algorithmic bandwidth (algbw), bus bandwidth (busbw)
- Cross-rank straggler analysis via `all_reduce(MIN)` on latency vectors
- Config: `"comms_logger": {"enabled": true, "prof_all": true}`

### Autotuning System

Searches over ZeRO stages (0-3), micro batch sizes, and per-stage ZeRO knobs:
- Stage 1: `reduce_bucket_size`, `allgather_bucket_size`
- Stage 2: `overlap_comm`, `reduce_scatter`, bucket sizes, `contiguous_gradients`
- Stage 3: All of Stage 2 plus `allgather_partitions`

Tuner strategies: `GridSearchTuner`, `RandomTuner`, `ModelBasedTuner` (XGBoost cost model).
Metrics: `throughput` (default), `latency`, `flops`, `forward`, `step`.

### NVTX Integration

`@instrument_w_nvtx` applied to: `DeepSpeedEngine.forward()`, `backward()`, `allreduce_gradients()`, ZeRO-3 parameter operations (all_gather, partition, wait), MiCS, SuperOffload. Auto-disabled under `torch.compile`.

### Flight Recorder

**Not implemented.** No `FlightRecorder` class, no structured failure diagnosis logging system, no crash dump mechanism.

### Recommended Profiling Dimensions for KernelWiki

| Dimension | DeepSpeed Coverage | Gaps |
|---|---|---|
| Compute Throughput | FLOPs Profiler (theoretical) | No hardware counter integration, no roofline model |
| Communication | CommsLogger + comm_profile | No automatic overlap measurement |
| Memory | see_memory_usage + MemoryProfilingInterpreter | No memory timeline export |
| Parameter Movement | PartitionedParameterProfiler | ZeRO-3 only, no bandwidth utilization |
| End-to-End Throughput | ThroughputTimer + Autotuner | No tokens/sec, no MFU calculation |
| GPU Kernel-Level | NVTX annotations | Function-level only, not kernel-level |
| Failure Diagnosis | None | No flight recorder |

### Proposed New Profiling techniques

| Tag | Description |
|---|---|
| `flops-profiler` | Theoretical FLOPs estimation via tensor shape analysis |
| `wallclock-breakdown` | Per-phase training step timing (fwd/bwd/step) |
| `comm-profiler` | Per-collective latency/bandwidth/straggler profiling |
| `autotuning` | Automated ZeRO stage and batch size search |
| `nvtx-annotation` | NVTX range markers for nsys integration |
| `compile-profiler` | Per-FX-node timing and memory for compiled graphs |

---

## Synthesis: Expansion Decision Summary

### S.1 Library Classification

| Property | Value |
|---|---|
| Library | DeepSpeed |
| GitHub URL | microsoft/DeepSpeed |
| Type | **full-stack** (owns compute kernels, communication orchestration, parallelism strategies, memory management, precision, and profiling) |
| Contains CUDA Kernels | Yes (61 .cu/.cuh files, 10 Triton files) |
| Primary Knowledge Dimensions | Dim 3 (Parallelism -- ZeRO is the flagship), Dim 4 (Memory -- ZeRO-Offload/Infinity), Dim 2 (Communication -- gradient bucketing, quantized comm) |
| Recommended KernelWiki Priority | **P0** (core training framework, foundational to the distributed training ecosystem) |

### S.2 Proposed Tags (for controlled vocabulary YAML)

```yaml
kernel_types:
  # New from DeepSpeed -- Compute
  - fused-adam           # Fused multi-tensor Adam/AdamW GPU optimizer
  - fused-lamb           # Fused three-phase LAMB optimizer with trust-ratio
  - fused-lion           # Fused multi-tensor Lion sign-based optimizer
  - cpu-adam             # CPU Adam with AVX SIMD for ZeRO-Offload
  - cpu-adagrad          # CPU Adagrad with AVX SIMD
  - cpu-lion             # CPU Lion with AVX SIMD
  - activation-backward  # Training activation backward pass kernel (dgelu, dsilu)
  - attention-backward   # Softmax backward pass for attention
  - layernorm-backward   # LayerNorm backward (gamma/beta grad + input grad)
  - dropout              # Dropout with Philox PRNG mask generation
  - fused-transformer-layer  # Complete fused transformer layer with fwd+bwd
  - random-ltd           # Random Layer Token Dropping gather/scatter/sort
  - fake-quantize        # Simulated quantization for QAT
  - quant-reduce         # Fused dequantize + multi-tensor reduce + requantize
  - fp-quantize          # FP8/FP6/FP4 floating-point quantization
  - deepcompile-op       # DeepCompile graph-level ops for compiled ZeRO
  - evoformer-attention  # Evoformer attention for AlphaFold2 (fwd+bwd)
  # New from DeepSpeed -- Communication
  - quantized-allreduce  # 1-bit compressed AllReduce with error feedback
  - quantized-reduce-scatter  # INT4 hierarchical All-to-All gradient reduction
  - sdma-allgather       # AMD SDMA copy-engine-based AllGather

techniques:
  # New from DeepSpeed -- Communication
  - compute-comm-overlap        # Backward reduce overlapped with gradient computation
  - double-buffered-ipg         # Double-buffered Independent Partition Gradient buckets
  - trace-based-prefetch        # Record-replay parameter access for async allgather
  - gradient-bucketing          # Batching gradients into configurable buckets
  - hierarchical-quantized-reduction  # 2-stage intra/inter-node INT4 all-to-all
  - loco-error-feedback         # Error-compensated quantized gradients with EMA
  - domino-async-tp             # Dual micro-batch interleaved async TP overlap
  # New from DeepSpeed -- Parallelism
  - zero-stage-1                # Optimizer state partitioning
  - zero-stage-2                # Gradient + optimizer state partitioning
  - zero-stage-3                # Full partitioning with on-demand parameter gather
  - zero-plus-plus              # Quantized hierarchical communication for ZeRO-3
  - pipeline-1f1b               # Synchronous 1F1B interleaved pipeline schedule
  - auto-tp                     # Automatic tensor parallelism via model analysis
  - ulysses-sp                  # All-to-All based sequence parallelism
  - fpdt-sp                     # Chunked Flash Attention tiled sequence distribution
  - expert-parallel             # MoE token dispatch via All-to-All
  - mics                        # Minimum Communication Sharding
  - zenflow-selective            # Top-k selective gradient update
  - deepcompile-z3              # Compiled ZeRO-3 with AllGather scheduling
  # New from DeepSpeed -- Memory
  - zero-offload-cpu            # CPU offloading for optimizer and parameters
  - zero-infinity-nvme          # NVMe offloading with async I/O
  - superoffload-async-cpu      # Separate-process CPU Adam for superchips
  - pipelined-nvme-io           # Overlapped read/write for NVMe tiles
  - activation-checkpoint       # Gradient checkpointing (reentrant/non-reentrant)
  - activation-partition        # Activation partitioning across MP GPUs
  - activation-cpu-offload      # Activation offloading to CPU pinned memory
  - random-ltd-token-drop       # Layer token dropping for activation memory reduction
  # New from DeepSpeed -- Precision
  - dynamic-loss-scaling        # FP16 dynamic loss scaling with hysteresis
  - bf16-mixed-precision        # BF16 training with FP32 master weights
  - onebit-compressed-comm      # 1-bit sign compression with error feedback
  - quantized-weight-comm       # INT4/INT8 quantized AllGather for weights
  - quantized-gradient-comm     # INT4 hierarchical All-to-All for gradients
  - gradient-predivide          # Pre-dividing gradients before allreduce
  # New from DeepSpeed -- Profiling
  - flops-profiler              # Theoretical FLOPs estimation via shape analysis
  - wallclock-breakdown         # Per-phase training step timing
  - comm-profiler               # Per-collective latency/bandwidth profiling
  - autotuning                  # Automated ZeRO stage and batch size search

hardware_features:
  # New from DeepSpeed
  - sdma-copy-engine   # AMD MI300 SDMA hardware DMA for AllGather (mori backend)
  - gds                # GPUDirect Storage for NVMe-to-GPU DMA (ZeRO-Infinity)
  - avx-512            # AVX-512 SIMD for CPU optimizer kernels

source_categories:
  - training-framework  # Full-stack LLM training framework (already exists or add)
```

### S.3 Wiki Page Topics

| # | Wiki Subdirectory | Proposed Page ID | Title | Source Evidence | Related Existing Pages |
|---|---|---|---|---|---|
| 1 | training/ | training-zero-optimizer | ZeRO: Zero Redundancy Optimizer | `runtime/zero/stage_1_and_2.py`, `stage3.py` | (none) |
| 2 | training/ | training-fused-optimizer | Fused GPU Optimizer Kernels (Adam/LAMB/Lion) | `csrc/adam/`, `csrc/lamb/`, `csrc/lion/` | (none) |
| 3 | training/ | training-activation-checkpoint | Activation Checkpointing Strategies | `runtime/activation_checkpointing/checkpointing.py` | (none) |
| 4 | training/ | training-mixed-precision | Mixed Precision Training (FP16/BF16) | `runtime/fp16/`, `runtime/bf16_optimizer.py` | (none) |
| 5 | training/ | training-quantized-comm | Quantized Communication (1-bit, ZeRO++) | `runtime/fp16/onebit/`, `comm/coalesced_collectives.py` | (none) |
| 6 | training/ | training-cpu-offload | CPU/NVMe Offloading (ZeRO-Offload/Infinity) | `runtime/swap_tensor/`, `offload_config.py` | (none) |
| 7 | communication/ | comm-gradient-bucketing | Gradient Bucketing and Communication Scheduling | `runtime/zero/stage_1_and_2.py` IPG buckets | (none) |
| 8 | communication/ | comm-domino-async-tp | Domino Async Tensor Parallel Overlap | `runtime/domino/async_linear.py` | (none) |
| 9 | communication/ | comm-sdma-allgather | AMD SDMA Copy Engine AllGather | `comm/mori.py` | (none) |
| 10 | parallelism/ | parallel-pipeline-1f1b | Pipeline Parallelism 1F1B Schedule | `runtime/pipe/schedule.py`, `engine.py` | (none) |
| 11 | parallelism/ | parallel-ulysses-sp | DeepSpeed-Ulysses Sequence Parallelism | `sequence/layer.py` | (none) |
| 12 | parallelism/ | parallel-moe-expert | MoE Expert Parallelism with All-to-All | `moe/sharded_moe.py` | (none) |
| 13 | parallelism/ | parallel-deepcompile | DeepCompile: torch.compile + ZeRO-3 | `compile/` | (none) |
| 14 | parallelism/ | parallel-zenflow | ZenFlow Selective Gradient Update | `runtime/zenflow/` | (none) |

### S.4 Repository Mappings (slug -> org/repo)

```python
# For the PR candidate search script
"deepspeed": "microsoft/DeepSpeed",

# For the PR page generation script
"deepspeed": "microsoft/DeepSpeed",
```

### S.5 Keyword-to-Tag Mappings (for automated PR tagger)

```python
# keyword -> kernel_type tag
"fused_adam": "fused-adam",
"fusedadam": "fused-adam",
"cpu_adam": "cpu-adam",
"fused_lamb": "fused-lamb",
"fused_lion": "fused-lion",
"cpu_lion": "cpu-lion",
"cpu_adagrad": "cpu-adagrad",
"dgelu": "activation-backward",
"dsilu": "activation-backward",
"dswiglu": "activation-backward",
"softmax_backward": "attention-backward",
"layernorm_backward": "layernorm-backward",
"dropout_kernel": "dropout",
"random_ltd": "random-ltd",
"layer_token_drop": "random-ltd",
"fake_quantize": "fake-quantize",
"quant_reduce": "quant-reduce",
"fp_quantize": "fp-quantize",
"fp8_gemm": "fp-quantize",
"evoformer": "evoformer-attention",
"deepcompile": "deepcompile-op",
"onebit": "quantized-allreduce",
"1bit_adam": "quantized-allreduce",
"sdma_allgather": "sdma-allgather",

# keyword -> technique tag
"zero_stage": "zero-stage-3",
"zero_optimization": "zero-stage-3",
"zero_offload": "zero-offload-cpu",
"zero_infinity": "zero-infinity-nvme",
"zero_quantized": "zero-plus-plus",
"zeropp": "zero-plus-plus",
"hpz": "zero-plus-plus",
"loco": "loco-error-feedback",
"overlap_comm": "compute-comm-overlap",
"backward_prefetch": "trace-based-prefetch",
"reduce_bucket": "gradient-bucketing",
"allgather_bucket": "gradient-bucketing",
"domino": "domino-async-tp",
"pipeline_parallel": "pipeline-1f1b",
"pipe_schedule": "pipeline-1f1b",
"auto_tp": "auto-tp",
"tensor_parallel": "auto-tp",
"ulysses": "ulysses-sp",
"distributed_attention": "ulysses-sp",
"fpdt": "fpdt-sp",
"expert_parallel": "expert-parallel",
"moe_layer": "expert-parallel",
"mics": "mics",
"zenflow": "zenflow-selective",
"activation_checkpoint": "activation-checkpoint",
"checkpoint_activations": "activation-checkpoint",
"cpu_checkpointing": "activation-cpu-offload",
"partition_activations": "activation-partition",
"superoffload": "superoffload-async-cpu",
"dynamic_loss_scale": "dynamic-loss-scaling",
"loss_scale_window": "dynamic-loss-scaling",
"bf16_optimizer": "bf16-mixed-precision",
"flops_profiler": "flops-profiler",
"wall_clock_breakdown": "wallclock-breakdown",
"comms_logger": "comm-profiler",
"autotuning": "autotuning",

# keyword -> hardware_feature tag
"sdma": "sdma-copy-engine",
"mori": "sdma-copy-engine",
"use_gds": "gds",
"gpu_direct": "gds",
"avx512": "avx-512",
"avx_512": "avx-512",
```

### S.6 PR Search Keywords (for candidate ledger)

```yaml
keywords_used:
  - zero_optimization
  - zero_stage
  - zero_offload
  - zero_infinity
  - zeropp
  - zero_quantized
  - loco
  - hpz
  - fused_adam
  - fused_lamb
  - fused_lion
  - cpu_adam
  - cpu_adagrad
  - deepcompile
  - domino
  - pipeline_parallel
  - tensor_parallel
  - auto_tp
  - ulysses
  - distributed_attention
  - fpdt
  - sequence_parallel
  - expert_parallel
  - moe
  - mics
  - zenflow
  - activation_checkpoint
  - superoffload
  - random_ltd
  - fp_quantize
  - fp8_gemm
  - onebit
  - 1bit_adam
  - dynamic_loss_scale
  - bf16
  - mixed_precision
  - flops_profiler
  - comms_logger
  - autotuning
  - overlap_comm
  - reduce_bucket
  - gradient_bucketing
  - sdma
  - mori
  - nvme
  - gds
  - swap_tensor
  - fake_quantize
  - quant_reduce
  - evoformer
```

### S.7 Inclusion Policy Lane

```yaml
deepspeed-training-framework:
  description: |
    Full-stack training framework covering compute kernels, ZeRO optimizer,
    multi-dimensional parallelism, memory management, precision, and profiling.
    Capture PRs touching kernel code, ZeRO runtime, parallelism modules,
    communication optimizations, and precision management.
  capture_criteria:
    - changed_paths_match:
        - "csrc/**/*.cu"
        - "csrc/**/*.cuh"
        - "csrc/**/*.cpp"
        - "csrc/**/*.h"
        - "deepspeed/ops/**/*.py"
        - "deepspeed/runtime/zero/**/*.py"
        - "deepspeed/runtime/pipe/**/*.py"
        - "deepspeed/runtime/tensor_parallel/**/*.py"
        - "deepspeed/runtime/domino/**/*.py"
        - "deepspeed/runtime/zenflow/**/*.py"
        - "deepspeed/runtime/fp16/**/*.py"
        - "deepspeed/runtime/activation_checkpointing/**/*.py"
        - "deepspeed/runtime/swap_tensor/**/*.py"
        - "deepspeed/runtime/superoffload/**/*.py"
        - "deepspeed/runtime/comm/**/*.py"
        - "deepspeed/compile/**/*.py"
        - "deepspeed/comm/**/*.py"
        - "deepspeed/sequence/**/*.py"
        - "deepspeed/moe/**/*.py"
        - "deepspeed/linear/**/*.py"
        - "deepspeed/profiling/**/*.py"
        - "deepspeed/compression/**/*.py"
        - "op_builder/**/*.py"
    - title_contains_any:
        - zero
        - fused
        - kernel
        - backward
        - optimizer
        - pipeline
        - tensor_parallel
        - sequence_parallel
        - expert_parallel
        - moe
        - domino
        - zenflow
        - deepcompile
        - fp8
        - fp16
        - bf16
        - quantiz
        - checkpoint
        - offload
        - nvme
        - overlap
        - allreduce
        - reduce_scatter
        - allgather
        - all_to_all
        - comm
        - profiler
        - autotuning
        - sdma
        - mics
        - random_ltd
        - onebit
        - 1bit
        - superoffload
  skip_criteria:
    - changed_paths_match_only:
        - "docs/**"
        - "blogs/**"
        - "examples/**"
        - "docker/**"
        - ".github/**"
        - "*.md"
        - "*.rst"
    - pure_config_only: true
```

### S.8 Schema Extensions (if any)

New optional frontmatter fields for Wiki pages from DeepSpeed:

- `scope: training` -- distinguishes training-specific pages from inference
- `zero_stage: [0, 1, 2, 3]` -- applicable ZeRO stages
- `parallelism_dimension: [dp, tp, pp, sp, ep]` -- which parallelism dimension
- `communication_pattern: [allreduce, reduce-scatter, allgather, all-to-all, p2p]` -- primary collective
- `offload_target: [gpu, cpu, nvme]` -- where data resides
- `precision_mode: [fp32, fp16, bf16, fp8, int8, int4]` -- numerical precision

### S.9 Hardware Features Relevant to DeepSpeed's Training Workloads

| Hardware Feature | Inference Relevance | Training Relevance | Specific Impact on DeepSpeed |
|---|---|---|---|
| NVLink 5 (1.8 TB/s) | Partial | Core | Doubles gradient AllReduce/ReduceScatter bandwidth for ZeRO |
| NVSwitch 4 (NVL72, 130 TB/s) | Partial | Core | Enables ZeRO-3 AllGather at scale with minimal latency |
| NVLink-SHARP FP8 | No | Core | Would 4x reduce ZeRO gradient ReduceScatter bandwidth (not yet integrated) |
| Symmetric Memory (9x latency) | Partial | Core | DeepCompile has `enable_symm_mem_for_group` but not yet used for ZeRO-3 AllGather |
| Copy Engine (zero-SM) | Partial | Core | AMD SDMA AllGather implemented via mori; NVIDIA CE not yet integrated |
| MXFP8 hardware (Blackwell) | Yes | Core | Not yet supported; would enable hardware-native FP8 training |
| 192 MB L2 Cache | Beneficial | Beneficial | Improves activation recomputation and fused optimizer kernel performance |
| 192 GB HBM3e @ 8 TB/s | Core | Core | Larger models fit per-GPU, reduces ZeRO-3 AllGather frequency |
| GPUDirect RDMA | Important | Important | Multi-node gradient sync bypasses CPU copy |
| GPUDirect Storage | Partial | Core | Integrated via `use_gds` for ZeRO-Infinity NVMe offload |
| AVX-512 | No | Core | CPU optimizer kernels (ZeRO-Offload Adam/Lion/Adagrad) |
| AMD SDMA | No | Core | MI300 copy-engine AllGather via mori backend |

### S.10 Upstream/Downstream Dependencies to Also Track

| Slug | GitHub URL | Relationship | Justification |
|---|---|---|---|
| `pytorch` | pytorch/pytorch | runtime-dependency | DeepSpeed builds on torch.distributed, torch.autograd, torch.compile |
| `nccl` | NVIDIA/nccl | communication-backend | All GPU communication goes through NCCL via torch.distributed |
| `apex` | NVIDIA/apex | kernel-provider | Multi-tensor apply infrastructure adapted from apex |
| `triton` | triton-lang/triton | kernel-provider | Sparse attention, inference, FP8 dequant GEMM Triton kernels |
| `cutlass` | NVIDIA/cutlass | kernel-provider | Mixed-precision GEMM and MoE GEMM for inference v2 |
| `transformer-engine` | NVIDIA/TransformerEngine | potential-integration | FP8 training recipes DeepSpeed currently lacks |
| `torchtitan` | pytorch/torchtitan | related-framework | PyTorch-native training framework, alternative approach to same problems |
| `megatron-lm` | NVIDIA/Megatron-LM | related-framework | Megatron-style TP/PP that DeepSpeed's TP design references |

---

### Library Type Adaptation Rationale

DeepSpeed is classified as a **full-stack training framework**. All six dimensions receive full analysis because:

1. **Dimension 1 (Compute Kernels)**: DeepSpeed writes 61+ CUDA kernels for optimizers, activations, normalization, dropout, quantization, and Random-LTD. It relies on cuBLAS/CUTLASS only for GEMM.
2. **Dimension 2 (Communication)**: Extensive communication orchestration with 16+ collective operations, multiple backends, gradient bucketing, quantized communication (ZeRO++), and custom SDMA paths.
3. **Dimension 3 (Parallelism)**: The flagship dimension -- ZeRO stages 0-3, pipeline parallelism, tensor parallelism (with Domino async overlap), sequence parallelism (Ulysses + FPDT), MoE expert parallelism, MiCS, ZenFlow.
4. **Dimension 4 (Memory)**: The other flagship -- ZeRO-Offload (CPU), ZeRO-Infinity (NVMe), SuperOffload, activation checkpointing variants, gradient compression, Random-LTD.
5. **Dimension 5 (Precision)**: FP16/BF16 mixed precision with dynamic loss scaling, FP8 weight quantization (partial), INT4/INT8 QAT, 1-bit compressed communication. Notable gap: no native FP8 training loop.
6. **Dimension 6 (Profiling)**: FLOPs profiler, wall clock breakdown, communication logger with straggler analysis, autotuning, NVTX annotations, compile-time profilers. Notable gap: no flight recorder.

No dimensions were skipped or given light treatment.

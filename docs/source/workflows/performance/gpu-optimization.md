# GPU Optimization Guide

## Overview

AIQToolkit leverages NVIDIA GPUs to achieve exceptional performance through CUDA kernels, tensor cores, and optimized memory management. This guide covers GPU optimization techniques for maximum performance.

## GPU Architecture Optimization

### CUDA Kernel Development

```python
# src/aiq/cuda_kernels/similarity_kernels.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// Optimized cosine similarity kernel using tensor cores
__global__ void tensor_core_similarity_kernel(
    const half* __restrict__ embeddings,
    const half* __restrict__ query,
    float* __restrict__ results,
    int num_embeddings,
    int embedding_dim
) {
    // Tensor Core dimensions (16x16x16)
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Warp-level matrix multiply-accumulate
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    // Global thread indices
    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int warpN = (blockIdx.y * blockDim.y + threadIdx.y);
    
    // Bounds checking
    if (warpM >= num_embeddings || warpN >= 1) return;
    
    // Compute dot product using tensor cores
    for (int k = 0; k < embedding_dim; k += WMMA_K) {
        // Load embedding fragment
        wmma::load_matrix_sync(
            a_frag,
            embeddings + warpM * WMMA_M * embedding_dim + k,
            embedding_dim
        );
        
        // Load query fragment
        wmma::load_matrix_sync(
            b_frag,
            query + k,
            embedding_dim
        );
        
        // Perform matrix multiply-accumulate
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    
    // Store results
    wmma::store_matrix_sync(
        results + warpM * WMMA_M,
        c_frag,
        num_embeddings,
        wmma::mem_row_major
    );
}

// Optimized attention kernel
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const int seq_len,
    const int d_model,
    const int num_heads
) {
    extern __shared__ float shared_mem[];
    
    const int head_dim = d_model / num_heads;
    const int tid = threadIdx.x;
    const int head_id = blockIdx.x;
    
    // Shared memory pointers
    float* s_q = shared_mem;
    float* s_k = s_q + head_dim;
    float* s_v = s_k + head_dim;
    float* s_scores = s_v + head_dim;
    
    // Load query for this head
    if (tid < head_dim) {
        s_q[tid] = Q[head_id * head_dim + tid];
    }
    __syncthreads();
    
    // Compute attention scores
    for (int pos = 0; pos < seq_len; pos++) {
        // Load key
        if (tid < head_dim) {
            s_k[tid] = K[pos * d_model + head_id * head_dim + tid];
        }
        __syncthreads();
        
        // Compute dot product
        float score = 0.0f;
        for (int i = 0; i < head_dim; i++) {
            score += s_q[i] * s_k[i];
        }
        score /= sqrtf(float(head_dim));
        
        // Apply softmax (simplified)
        s_scores[pos] = expf(score);
    }
    __syncthreads();
    
    // Normalize scores
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        sum += s_scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        s_scores[i] /= sum;
    }
    __syncthreads();
    
    // Compute weighted sum of values
    if (tid < head_dim) {
        float out = 0.0f;
        for (int pos = 0; pos < seq_len; pos++) {
            float v = V[pos * d_model + head_id * head_dim + tid];
            out += s_scores[pos] * v;
        }
        output[head_id * head_dim + tid] = out;
    }
}
```

### Mixed Precision Training

```python
# src/aiq/optimization/mixed_precision.py

import torch
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Any

class MixedPrecisionOptimizer:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.model = model
        self.config = config
        self.scaler = GradScaler()
        
        # Convert model to mixed precision
        self._convert_to_mixed_precision()
        
    def _convert_to_mixed_precision(self):
        """Convert model layers to mixed precision"""
        
        for module in self.model.modules():
            # Convert linear layers
            if isinstance(module, torch.nn.Linear):
                # Keep compute in FP16, accumulate in FP32
                module.half()
                
                # Use tensor cores for matmul
                if module.in_features % 8 == 0 and module.out_features % 8 == 0:
                    module.use_tensor_cores = True
            
            # Convert attention layers
            elif isinstance(module, torch.nn.MultiheadAttention):
                module.half()
                
                # Enable flash attention if available
                if hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
                    module.use_flash_attention = True
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step with mixed precision"""
        
        self.model.train()
        
        # Forward pass with autocast
        with autocast():
            outputs = self.model(**batch)
            loss = outputs['loss']
        
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Optimizer step
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Clear gradients
        self.optimizer.zero_grad()
        
        return {
            'loss': loss.item(),
            'scale': self.scaler.get_scale()
        }
```

### CUDA Graphs

```python
# src/aiq/optimization/cuda_graphs.py

import torch
from typing import List, Dict, Any

class CUDAGraphOptimizer:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def capture_graph(self, sample_input: torch.Tensor):
        """Capture CUDA graph for static computation"""
        
        # Warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        
        with torch.cuda.stream(s):
            for _ in range(3):
                _ = self.model(sample_input)
        
        torch.cuda.current_stream().wait_stream(s)
        
        # Create static input/output
        self.static_input = sample_input.clone()
        self.graph = torch.cuda.CUDAGraph()
        
        # Record graph
        with torch.cuda.graph(self.graph):
            self.static_output = self.model(self.static_input)
    
    def execute(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Execute using CUDA graph"""
        
        if self.graph is None:
            self.capture_graph(input_tensor)
        
        # Copy input to static buffer
        self.static_input.copy_(input_tensor)
        
        # Replay graph
        self.graph.replay()
        
        # Return copy of output
        return self.static_output.clone()
```

## Memory Optimization

### GPU Memory Management

```python
# src/aiq/optimization/gpu_memory.py

import torch
import GPUtil
from typing import Dict, Any, List

class GPUMemoryManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.memory_pool = {}
        self.pinned_memory = {}
        
    def allocate_tensor(
        self,
        shape: tuple,
        dtype: torch.dtype = torch.float32,
        device: str = 'cuda:0'
    ) -> torch.Tensor:
        """Allocate tensor with memory pooling"""
        
        key = (shape, dtype, device)
        
        # Check pool first
        if key in self.memory_pool and self.memory_pool[key]:
            tensor = self.memory_pool[key].pop()
            tensor.zero_()
            return tensor
        
        # Allocate new tensor
        return torch.empty(shape, dtype=dtype, device=device)
    
    def free_tensor(self, tensor: torch.Tensor):
        """Return tensor to pool"""
        
        key = (tensor.shape, tensor.dtype, str(tensor.device))
        
        if key not in self.memory_pool:
            self.memory_pool[key] = []
        
        self.memory_pool[key].append(tensor)
    
    def pin_memory(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pin tensor memory for faster CPU-GPU transfer"""
        
        if not tensor.is_pinned():
            pinned = torch.empty_like(tensor, pin_memory=True)
            pinned.copy_(tensor)
            return pinned
        
        return tensor
    
    def optimize_memory_usage(self):
        """Optimize GPU memory usage"""
        
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        
        # Synchronize all streams
        torch.cuda.synchronize()
        
        # Get memory info
        for gpu in GPUtil.getGPUs():
            print(f"GPU {gpu.id}: {gpu.memoryUsed}MB / {gpu.memoryTotal}MB")
            
            # Free memory if usage is high
            if gpu.memoryUtil > 0.9:
                self._emergency_cleanup(gpu.id)
    
    def _emergency_cleanup(self, gpu_id: int):
        """Emergency memory cleanup"""
        
        # Clear memory pools
        self.memory_pool.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache
        torch.cuda.empty_cache()
```

### Gradient Checkpointing

```python
# src/aiq/optimization/gradient_checkpointing.py

import torch
from torch.utils.checkpoint import checkpoint
from typing import List, Any

class GradientCheckpointing:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.checkpointed_modules = []
        
    def enable_checkpointing(self, modules: List[str]):
        """Enable gradient checkpointing for specified modules"""
        
        for name, module in self.model.named_modules():
            if any(target in name for target in modules):
                # Wrap module with checkpointing
                self._wrap_module(module)
                self.checkpointed_modules.append(name)
    
    def _wrap_module(self, module: torch.nn.Module):
        """Wrap module with gradient checkpointing"""
        
        original_forward = module.forward
        
        def checkpointed_forward(*args, **kwargs):
            return checkpoint(original_forward, *args, **kwargs)
        
        module.forward = checkpointed_forward
```

## Multi-GPU Optimization

### Distributed Training

```python
# src/aiq/optimization/distributed_training.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, Any

class DistributedTrainer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.world_size = torch.cuda.device_count()
        
    def setup_distributed(self, rank: int):
        """Setup distributed training"""
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=rank
        )
        
        # Set device
        torch.cuda.set_device(rank)
        
    def create_model(self, model: torch.nn.Module, rank: int) -> DDP:
        """Create distributed model"""
        
        # Move model to GPU
        model = model.to(f'cuda:{rank}')
        
        # Wrap with DDP
        ddp_model = DDP(
            model,
            device_ids=[rank],
            output_device=rank,
            find_unused_parameters=self.config.get('find_unused_parameters', False)
        )
        
        return ddp_model
    
    def train_epoch(
        self,
        model: DDP,
        dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        rank: int
    ) -> Dict[str, float]:
        """Train one epoch with distributed model"""
        
        model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            # Move batch to device
            batch = {k: v.to(f'cuda:{rank}') for k, v in batch.items()}
            
            # Forward pass
            outputs = model(**batch)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient synchronization happens automatically
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
        
        # Synchronize metrics across GPUs
        avg_loss = self._sync_metrics({'loss': total_loss / len(dataloader)})
        
        return avg_loss
    
    def _sync_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """Synchronize metrics across all GPUs"""
        
        synced_metrics = {}
        
        for key, value in metrics.items():
            tensor = torch.tensor(value).cuda()
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)
            synced_metrics[key] = tensor.item()
        
        return synced_metrics
```

### Pipeline Parallelism

```python
# src/aiq/optimization/pipeline_parallel.py

import torch
from torch.distributed.pipeline.sync import Pipe
from typing import List, Any

class PipelineParallel:
    def __init__(self, model: torch.nn.Module, config: Dict[str, Any]):
        self.config = config
        self.num_gpus = torch.cuda.device_count()
        self.model = self._split_model(model)
        
    def _split_model(self, model: torch.nn.Module) -> Pipe:
        """Split model across GPUs"""
        
        # Analyze model structure
        layers = list(model.children())
        
        # Calculate split points
        split_size = len(layers) // self.num_gpus
        splits = []
        
        for i in range(self.num_gpus):
            start = i * split_size
            end = start + split_size if i < self.num_gpus - 1 else len(layers)
            
            # Create sequential module for this split
            split_module = torch.nn.Sequential(*layers[start:end])
            splits.append(split_module.to(f'cuda:{i}'))
        
        # Create pipeline
        return Pipe(
            torch.nn.Sequential(*splits),
            balance=[split_size] * (self.num_gpus - 1) + [len(layers) - split_size * (self.num_gpus - 1)],
            devices=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'][:self.num_gpus],
            chunks=self.config.get('micro_batch_size', 8)
        )
    
    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Forward pass through pipeline"""
        
        return self.model(input_tensor)
```

## Kernel Fusion

### Custom Fused Kernels

```python
# src/aiq/cuda_kernels/fused_operations.cu

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// Fused LayerNorm + GeLU kernel
__global__ void fused_layernorm_gelu_kernel(
    const float* __restrict__ input,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ output,
    const int batch_size,
    const int hidden_size,
    const float epsilon
) {
    extern __shared__ float shared_mem[];
    
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const int idx = bid * hidden_size + tid;
    
    // Shared memory for reduction
    float* s_mean = shared_mem;
    float* s_var = s_mean + blockDim.x;
    
    // Compute mean
    float local_sum = 0.0f;
    if (tid < hidden_size) {
        local_sum = input[idx];
    }
    
    // Warp-level reduction for mean
    __shared__ float warp_sums[32];
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    float warp_sum = warp_reduce_sum(local_sum);
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            total_sum += warp_sums[i];
        }
        s_mean[0] = total_sum / hidden_size;
    }
    __syncthreads();
    
    float mean = s_mean[0];
    
    // Compute variance
    float local_var = 0.0f;
    if (tid < hidden_size) {
        float diff = input[idx] - mean;
        local_var = diff * diff;
    }
    
    // Similar reduction for variance
    float warp_var = warp_reduce_sum(local_var);
    if (lane_id == 0) {
        warp_sums[warp_id] = warp_var;
    }
    __syncthreads();
    
    if (tid == 0) {
        float total_var = 0.0f;
        for (int i = 0; i < (blockDim.x + 31) / 32; i++) {
            total_var += warp_sums[i];
        }
        s_var[0] = total_var / hidden_size;
    }
    __syncthreads();
    
    float var = s_var[0];
    float inv_std = rsqrtf(var + epsilon);
    
    // Apply LayerNorm and GeLU
    if (tid < hidden_size) {
        float normalized = (input[idx] - mean) * inv_std;
        float scaled = normalized * gamma[tid] + beta[tid];
        
        // GeLU activation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const float c0 = 0.5f;
        const float c1 = 0.7978845608f; // sqrt(2/pi)
        const float c2 = 0.044715f;
        
        float x = scaled;
        float x3 = x * x * x;
        float tanh_arg = c1 * (x + c2 * x3);
        float tanh_val = tanhf(tanh_arg);
        
        output[idx] = c0 * x * (1.0f + tanh_val);
    }
}

// Fused attention + dropout kernel
__global__ void fused_attention_dropout_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    const float* __restrict__ dropout_mask,
    const int seq_len,
    const int d_model,
    const int num_heads,
    const float dropout_rate
) {
    extern __shared__ float shared_mem[];
    
    const int head_dim = d_model / num_heads;
    const int tid = threadIdx.x;
    const int head_id = blockIdx.x;
    const int seq_id = blockIdx.y;
    
    // Shared memory layout
    float* s_q = shared_mem;
    float* s_k = s_q + head_dim;
    float* s_scores = s_k + head_dim;
    
    // Load query
    if (tid < head_dim) {
        s_q[tid] = Q[seq_id * d_model + head_id * head_dim + tid];
    }
    __syncthreads();
    
    // Compute attention scores with masking
    for (int i = tid; i < seq_len; i += blockDim.x) {
        // Load key
        if (tid < head_dim) {
            s_k[tid] = K[i * d_model + head_id * head_dim + tid];
        }
        __syncthreads();
        
        // Compute dot product
        float score = 0.0f;
        for (int j = 0; j < head_dim; j++) {
            score += s_q[j] * s_k[j];
        }
        
        // Scale
        score /= sqrtf(float(head_dim));
        
        // Apply attention mask if needed
        if (i > seq_id) {
            score = -INFINITY;
        }
        
        s_scores[i] = score;
    }
    __syncthreads();
    
    // Softmax
    float max_score = -INFINITY;
    for (int i = 0; i < seq_len; i++) {
        max_score = fmaxf(max_score, s_scores[i]);
    }
    
    float sum = 0.0f;
    for (int i = 0; i < seq_len; i++) {
        s_scores[i] = expf(s_scores[i] - max_score);
        sum += s_scores[i];
    }
    
    for (int i = 0; i < seq_len; i++) {
        s_scores[i] /= sum;
        
        // Apply dropout
        if (dropout_mask[seq_id * seq_len + i] < dropout_rate) {
            s_scores[i] = 0.0f;
        } else {
            s_scores[i] /= (1.0f - dropout_rate);
        }
    }
    __syncthreads();
    
    // Compute weighted sum of values
    if (tid < head_dim) {
        float out = 0.0f;
        for (int i = 0; i < seq_len; i++) {
            float v = V[i * d_model + head_id * head_dim + tid];
            out += s_scores[i] * v;
        }
        output[seq_id * d_model + head_id * head_dim + tid] = out;
    }
}
```

## Profiling and Debugging

### GPU Profiling

```python
# src/aiq/profiling/gpu_profiler.py

import torch
import torch.profiler
from typing import Dict, Any, List

class GPUProfiler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.profiler = None
        
    def start_profiling(self):
        """Start GPU profiling"""
        
        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=1,
                warmup=1,
                active=3,
                repeat=2
            ),
            on_trace_ready=self._trace_handler,
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        self.profiler.start()
    
    def step(self):
        """Profiler step"""
        
        if self.profiler:
            self.profiler.step()
    
    def stop_profiling(self):
        """Stop profiling and generate report"""
        
        if self.profiler:
            self.profiler.stop()
            
            # Generate reports
            self._generate_reports()
    
    def _trace_handler(self, prof):
        """Handle profiler traces"""
        
        # Export Chrome trace
        prof.export_chrome_trace(f"trace_{prof.step_num}.json")
        
        # Print summary
        print(prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=10
        ))
    
    def _generate_reports(self):
        """Generate profiling reports"""
        
        # Generate kernel analysis
        kernel_stats = self._analyze_kernels()
        
        # Generate memory analysis
        memory_stats = self._analyze_memory()
        
        # Generate optimization suggestions
        suggestions = self._generate_suggestions(kernel_stats, memory_stats)
        
        # Save report
        with open('gpu_profile_report.txt', 'w') as f:
            f.write("GPU Profiling Report\n")
            f.write("===================\n\n")
            
            f.write("Kernel Statistics:\n")
            for stat in kernel_stats:
                f.write(f"  {stat}\n")
            
            f.write("\nMemory Statistics:\n")
            for stat in memory_stats:
                f.write(f"  {stat}\n")
            
            f.write("\nOptimization Suggestions:\n")
            for suggestion in suggestions:
                f.write(f"  - {suggestion}\n")
    
    def _analyze_kernels(self) -> List[str]:
        """Analyze CUDA kernels"""
        
        stats = []
        
        # Analyze kernel launch configuration
        stats.append("Kernel launch analysis:")
        stats.append("  - Check block/grid dimensions for optimization")
        stats.append("  - Look for kernels with low occupancy")
        
        return stats
    
    def _analyze_memory(self) -> List[str]:
        """Analyze GPU memory usage"""
        
        stats = []
        
        # Memory transfer analysis
        stats.append("Memory transfer analysis:")
        stats.append("  - Minimize CPU-GPU transfers")
        stats.append("  - Use pinned memory for frequent transfers")
        stats.append("  - Batch small transfers")
        
        return stats
```

### CUDA Error Checking

```python
# src/aiq/cuda/error_checking.py

import torch
import functools
from typing import Any, Callable

def cuda_error_check(func: Callable) -> Callable:
    """Decorator for CUDA error checking"""
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        torch.cuda.synchronize()
        
        try:
            result = func(*args, **kwargs)
            torch.cuda.synchronize()
            
            # Check for CUDA errors
            if torch.cuda.is_available():
                error = torch.cuda.get_last_error()
                if error != torch.cuda.CudaError.success:
                    raise RuntimeError(f"CUDA error: {error}")
            
            return result
            
        except Exception as e:
            # Get more detailed CUDA error info
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                error_name = torch.cuda.get_error_name(torch.cuda.get_last_error())
                error_string = torch.cuda.get_error_string(torch.cuda.get_last_error())
                
                raise RuntimeError(
                    f"CUDA operation failed: {error_name} - {error_string}\n"
                    f"Original error: {str(e)}"
                )
            else:
                raise
    
    return wrapper
```

## Best Practices

### 1. Memory Optimization

```python
# Best practice: Efficient memory usage
def optimize_gpu_memory():
    # Use gradient checkpointing for large models
    model.gradient_checkpointing_enable()
    
    # Clear cache regularly
    torch.cuda.empty_cache()
    
    # Use mixed precision
    with torch.cuda.amp.autocast():
        output = model(input)
    
    # Reuse buffers
    buffer = torch.empty(size, device='cuda')
    for batch in dataloader:
        buffer.copy_(batch)
        process(buffer)
```

### 2. Kernel Optimization

```python
# Best practice: Optimize kernel launches
def optimize_kernels():
    # Fuse operations
    output = torch.nn.functional.gelu(
        torch.nn.functional.layer_norm(input, normalized_shape)
    )
    
    # Use optimal block sizes
    BLOCK_SIZE = 256  # Multiple of 32 for warp efficiency
    
    # Minimize global memory access
    # Use shared memory for frequently accessed data
```

### 3. Multi-GPU Scaling

```python
# Best practice: Efficient multi-GPU usage
def scale_across_gpus():
    # Use DDP for data parallelism
    model = DistributedDataParallel(model)
    
    # Overlap computation and communication
    with torch.cuda.stream(compute_stream):
        output = model(input)
    
    with torch.cuda.stream(comm_stream):
        dist.all_reduce(gradients)
```

## Performance Monitoring

### Real-time GPU Monitoring

```python
# src/aiq/monitoring/gpu_monitor.py

import GPUtil
import psutil
import time
from typing import Dict, List

class GPUMonitor:
    def __init__(self):
        self.gpus = GPUtil.getGPUs()
        self.history = []
        
    def get_gpu_stats(self) -> List[Dict[str, Any]]:
        """Get current GPU statistics"""
        
        stats = []
        
        for gpu in self.gpus:
            stats.append({
                'id': gpu.id,
                'name': gpu.name,
                'temperature': gpu.temperature,
                'utilization': gpu.load * 100,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'memory_util': gpu.memoryUtil * 100,
                'power_draw': gpu.powerDraw,
                'power_limit': gpu.powerLimit
            })
        
        return stats
    
    def monitor_continuous(self, interval: float = 1.0):
        """Continuously monitor GPU stats"""
        
        while True:
            stats = self.get_gpu_stats()
            self.history.append({
                'timestamp': time.time(),
                'stats': stats
            })
            
            # Keep only last hour of data
            cutoff = time.time() - 3600
            self.history = [
                h for h in self.history 
                if h['timestamp'] > cutoff
            ]
            
            time.sleep(interval)
```

### Optimization Dashboard

```yaml
# monitoring/gpu_dashboard.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: gpu-dashboard
data:
  dashboard.json: |
    {
      "title": "GPU Performance Dashboard",
      "panels": [
        {
          "title": "GPU Utilization",
          "type": "graph",
          "targets": [
            {
              "expr": "gpu_utilization_percent{device=~\"$device\"}"
            }
          ]
        },
        {
          "title": "Memory Usage",
          "type": "graph",
          "targets": [
            {
              "expr": "gpu_memory_used_bytes{device=~\"$device\"} / gpu_memory_total_bytes{device=~\"$device\"} * 100"
            }
          ]
        },
        {
          "title": "Kernel Performance",
          "type": "table",
          "targets": [
            {
              "expr": "topk(10, gpu_kernel_duration_ms)"
            }
          ]
        }
      ]
    }
```

## Troubleshooting

### Common GPU Issues

1. **Out of Memory (OOM)**
   ```python
   try:
       output = model(input)
   except torch.cuda.OutOfMemoryError:
       torch.cuda.empty_cache()
       # Reduce batch size
       output = model(input[:batch_size//2])
   ```

2. **Kernel Launch Failures**
   ```python
   # Check kernel launch parameters
   max_threads = torch.cuda.get_device_properties(0).maxThreadsPerBlock
   block_size = min(desired_block_size, max_threads)
   ```

3. **Performance Degradation**
   ```python
   # Monitor temperature and throttling
   gpu = GPUtil.getGPUs()[0]
   if gpu.temperature > 80:
       print(f"GPU throttling: {gpu.temperature}°C")
   ```

### Debug Commands

```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA errors
export CUDA_LAUNCH_BLOCKING=1
python script.py

# Profile CUDA kernels
nsys profile --stats=true python script.py

# Analyze with Nsight Compute
ncu --export profile python script.py
```

## Next Steps

- Review [Performance Benchmarks](benchmarks.md)
- See [Distributed Training Guide](distributed-training.md)
- Check [Memory Optimization Guide](memory-optimization.md)
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Any
import logging
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

class GPUArchitecture(str):
    """GPU architecture identifiers"""
    VOLTA = "volta"
    TURING = "turing"
    AMPERE = "ampere"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"
    CPU = "cpu"

@dataclass
class OptimizationConfig:
    """Configuration for tensor optimization"""
    use_tf32: bool = True
    use_fp16: bool = True
    use_fp8: bool = False
    use_int8: bool = False
    mixed_precision: bool = True
    channel_last: bool = True
    cudnn_benchmark: bool = True

@dataclass
class ResourceRequirements:
    """Predicted resource requirements"""
    memory_mb: float
    compute_flops: float
    bandwidth_gbps: float
    recommended_batch_size: int
    recommended_gpu_count: int
    estimated_latency_ms: float

class TensorCoreOptimizer:
    """
    Optimizes models for NVIDIA Tensor Core utilization
    """
    
    TENSOR_CORE_SHAPES = {
        # M, N, K dimensions for optimal Tensor Core usage
        "fp16": [(16, 16, 16), (32, 8, 16), (8, 32, 16)],
        "tf32": [(16, 16, 8)],
        "fp8": [(16, 16, 16)],  # Hopper+
        "int8": [(16, 16, 32), (32, 16, 32), (16, 32, 32)]
    }
    
    def __init__(self, target_gpu: Optional[str] = None):
        self.target_gpu = target_gpu or self._detect_gpu()
        self.tensor_core_compatible = self._check_tensor_core_support()
        self.optimization_config = self._get_default_config()
        
        logger.info(f"Initialized TensorCoreOptimizer for {self.target_gpu}")
    
    def _detect_gpu(self) -> str:
        """Detect GPU architecture"""
        if not torch.cuda.is_available():
            return GPUArchitecture.CPU
        
        capability = torch.cuda.get_device_capability()
        major, minor = capability
        
        if major >= 9:  # Compute capability 9.0+
            return GPUArchitecture.HOPPER  # H100, H200
        elif major == 8:
            if minor >= 6:
                return GPUArchitecture.AMPERE  # A100, A30
            else:
                return GPUArchitecture.AMPERE  # A40, RTX 30 series
        elif major == 7:
            if minor >= 5:
                return GPUArchitecture.TURING  # RTX 20 series
            else:
                return GPUArchitecture.VOLTA  # V100
        else:
            return GPUArchitecture.CPU
    
    def _check_tensor_core_support(self) -> bool:
        """Check if GPU supports Tensor Cores"""
        return self.target_gpu in [
            GPUArchitecture.VOLTA,
            GPUArchitecture.TURING,
            GPUArchitecture.AMPERE,
            GPUArchitecture.HOPPER,
            GPUArchitecture.BLACKWELL
        ]
    
    def _get_default_config(self) -> OptimizationConfig:
        """Get default optimization config for architecture"""
        configs = {
            GPUArchitecture.HOPPER: OptimizationConfig(
                use_tf32=True,
                use_fp16=True,
                use_fp8=True,  # Hopper supports FP8
                use_int8=True,
                mixed_precision=True,
                channel_last=True,
                cudnn_benchmark=True
            ),
            GPUArchitecture.AMPERE: OptimizationConfig(
                use_tf32=True,
                use_fp16=True,
                use_fp8=False,
                use_int8=True,
                mixed_precision=True,
                channel_last=True,
                cudnn_benchmark=True
            ),
            GPUArchitecture.TURING: OptimizationConfig(
                use_tf32=False,  # No TF32 on Turing
                use_fp16=True,
                use_fp8=False,
                use_int8=True,
                mixed_precision=True,
                channel_last=True,
                cudnn_benchmark=True
            ),
            GPUArchitecture.VOLTA: OptimizationConfig(
                use_tf32=False,
                use_fp16=True,
                use_fp8=False,
                use_int8=False,  # Limited INT8 on Volta
                mixed_precision=True,
                channel_last=False,  # Less benefit on Volta
                cudnn_benchmark=True
            ),
            GPUArchitecture.CPU: OptimizationConfig(
                use_tf32=False,
                use_fp16=False,
                use_fp8=False,
                use_int8=False,
                mixed_precision=False,
                channel_last=False,
                cudnn_benchmark=False
            )
        }
        
        return configs.get(self.target_gpu, configs[GPUArchitecture.CPU])
    
    def optimize_model(self, model: nn.Module) -> nn.Module:
        """
        Optimize model for Tensor Core acceleration
        
        Args:
            model: PyTorch model to optimize
        
        Returns:
            Optimized model
        """
        if not self.tensor_core_compatible:
            logger.warning("No Tensor Core support, returning original model")
            return model
        
        # Apply global optimizations
        self._apply_global_optimizations()
        
        # Optimize model based on architecture
        if self.target_gpu == GPUArchitecture.HOPPER:
            model = self._optimize_for_hopper(model)
        elif self.target_gpu == GPUArchitecture.AMPERE:
            model = self._optimize_for_ampere(model)
        else:
            model = self._optimize_generic(model)
        
        # Optimize layer dimensions for Tensor Cores
        model = self._optimize_layer_dimensions(model)
        
        # Apply memory format optimizations
        if self.optimization_config.channel_last:
            model = model.to(memory_format=torch.channels_last)
        
        logger.info(f"Model optimized for {self.target_gpu}")
        return model
    
    def _apply_global_optimizations(self):
        """Apply global PyTorch optimizations"""
        if self.optimization_config.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        if self.optimization_config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        
        # Enable cuDNN deterministic mode for reproducibility
        torch.backends.cudnn.deterministic = False
    
    def _optimize_for_hopper(self, model: nn.Module) -> nn.Module:
        """Optimize specifically for Hopper architecture"""
        # Enable FP8 if available
        if self.optimization_config.use_fp8:
            try:
                # Apply FP8 conversion for compatible layers
                for name, module in model.named_modules():
                    if isinstance(module, (nn.Linear, nn.Conv2d)):
                        # Placeholder for FP8 conversion
                        # In production, use NVIDIA Transformer Engine
                        logger.debug(f"Would convert {name} to FP8")
            except Exception as e:
                logger.warning(f"FP8 optimization failed: {e}")
        
        return model
    
    def _optimize_for_ampere(self, model: nn.Module) -> nn.Module:
        """Optimize specifically for Ampere architecture"""
        # Ampere-specific optimizations
        if self.optimization_config.use_fp16:
            model = model.half()
        
        return model
    
    def _optimize_generic(self, model: nn.Module) -> nn.Module:
        """Generic Tensor Core optimizations"""
        if self.optimization_config.use_fp16:
            model = model.half()
        
        return model
    
    def _optimize_layer_dimensions(self, model: nn.Module) -> nn.Module:
        """Optimize layer dimensions for Tensor Core usage"""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Ensure dimensions are multiples of Tensor Core tiles
                in_features = module.in_features
                out_features = module.out_features
                
                optimal_in = self._round_to_tensor_core_dims(in_features)
                optimal_out = self._round_to_tensor_core_dims(out_features)
                
                if optimal_in != in_features or optimal_out != out_features:
                    logger.debug(
                        f"Layer {name}: Suggest resizing from "
                        f"({in_features}, {out_features}) to "
                        f"({optimal_in}, {optimal_out}) for Tensor Cores"
                    )
        
        return model
    
    def _round_to_tensor_core_dims(self, dim: int, dtype: str = "fp16") -> int:
        """Round dimension to nearest Tensor Core-friendly size"""
        tile_sizes = [shape[0] for shape in self.TENSOR_CORE_SHAPES.get(dtype, [])]
        if not tile_sizes:
            return dim
        
        # Round to nearest multiple of smallest tile size
        min_tile = min(tile_sizes)
        return ((dim + min_tile - 1) // min_tile) * min_tile
    
    def profile_tensor_core_usage(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Profile Tensor Core usage for a model
        
        Args:
            model: Model to profile
            input_shape: Input tensor shape
        
        Returns:
            Dictionary with profiling metrics
        """
        if not torch.cuda.is_available():
            return {"tensor_core_eligible": 0.0, "tensor_core_actual": 0.0}
        
        # Create dummy input
        dummy_input = torch.randn(input_shape, device='cuda')
        if self.optimization_config.use_fp16:
            dummy_input = dummy_input.half()
        
        # Profile with PyTorch profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True
        ) as prof:
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Analyze profiler results
        tensor_core_ops = 0
        total_ops = 0
        
        for event in prof.key_averages():
            if event.device_type == torch.profiler.DeviceType.CUDA:
                total_ops += 1
                # Check if operation is Tensor Core eligible
                if self._is_tensor_core_op(event):
                    tensor_core_ops += 1
        
        return {
            "tensor_core_eligible": tensor_core_ops / max(total_ops, 1),
            "total_cuda_ops": total_ops,
            "tensor_core_ops": tensor_core_ops
        }
    
    def _is_tensor_core_op(self, event) -> bool:
        """Check if a profiler event is a Tensor Core operation"""
        tensor_core_keywords = [
            "gemm", "conv", "tensor", "wmma",  # Common Tensor Core kernels
            "cutlass", "cublas"  # Libraries that use Tensor Cores
        ]
        
        event_name = event.key.lower()
        return any(keyword in event_name for keyword in tensor_core_keywords)

class ResourcePredictor:
    """
    Predicts resource requirements for workloads
    """
    def __init__(self):
        self.device_properties = {}
        self._initialize_device_properties()
    
    def _initialize_device_properties(self):
        """Initialize GPU device properties"""
        if not torch.cuda.is_available():
            return
        
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            self.device_properties[i] = {
                "name": prop.name,
                "compute_capability": (prop.major, prop.minor),
                "total_memory": prop.total_memory,
                "multiprocessor_count": prop.multi_processor_count,
                "max_threads_per_multiprocessor": prop.max_threads_per_multi_processor,
                "memory_bandwidth": self._estimate_bandwidth(prop.name)
            }
    
    def _estimate_bandwidth(self, gpu_name: str) -> float:
        """Estimate memory bandwidth in GB/s based on GPU name"""
        bandwidth_map = {
            "V100": 900,
            "A100": 1555,
            "A30": 933,
            "A40": 696,
            "H100": 2039,
            "H200": 4900,
            "RTX 3090": 936,
            "RTX 4090": 1008,
        }
        
        for key, value in bandwidth_map.items():
            if key in gpu_name:
                return value
        
        return 500  # Default estimate
    
    def predict_requirements(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        target_batch_size: int
    ) -> ResourceRequirements:
        """
        Predict resource requirements for a model and workload
        
        Args:
            model: PyTorch model
            input_shape: Input tensor shape (without batch dimension)
            target_batch_size: Desired batch size
        
        Returns:
            ResourceRequirements with predictions
        """
        # Calculate model parameters
        param_count = sum(p.numel() for p in model.parameters())
        param_memory = param_count * 4  # Assuming FP32
        
        # Estimate activations memory
        # This is a simplified estimate
        activation_memory = self._estimate_activation_memory(model, input_shape)
        
        # Total memory per sample
        memory_per_sample = (param_memory + activation_memory) / (1024 ** 2)  # MB
        total_memory = memory_per_sample * target_batch_size
        
        # Estimate compute requirements (FLOPs)
        flops = self._estimate_flops(model, input_shape)
        total_flops = flops * target_batch_size
        
        # Estimate bandwidth requirements
        bandwidth_gbps = self._estimate_bandwidth_requirements(
            model, input_shape, target_batch_size
        )
        
        # Recommend GPU count based on memory
        if torch.cuda.is_available() and 0 in self.device_properties:
            gpu_memory = self.device_properties[0]["total_memory"] / (1024 ** 3)  # GB
            recommended_gpus = max(1, int(np.ceil(total_memory / (gpu_memory * 0.8))))
        else:
            recommended_gpus = 1
        
        # Estimate latency (very rough estimate)
        if torch.cuda.is_available() and 0 in self.device_properties:
            compute_throughput = self.device_properties[0]["multiprocessor_count"] * 1e12  # TFLOPs
            estimated_latency = (total_flops / compute_throughput) * 1000  # ms
        else:
            estimated_latency = 100  # Default estimate
        
        return ResourceRequirements(
            memory_mb=total_memory,
            compute_flops=total_flops,
            bandwidth_gbps=bandwidth_gbps,
            recommended_batch_size=self._optimize_batch_size(total_memory, gpu_memory),
            recommended_gpu_count=recommended_gpus,
            estimated_latency_ms=estimated_latency
        )
    
    def _estimate_activation_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate activation memory requirements"""
        # Simplified estimation - in production, use more sophisticated methods
        # like torch.jit.script or custom hooks
        total_activations = np.prod(input_shape)
        
        # Rough estimate: 2x input size per layer on average
        num_layers = len(list(model.modules()))
        return total_activations * 4 * num_layers * 2  # bytes
    
    def _estimate_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> int:
        """Estimate FLOPs for model inference"""
        # Simplified estimation
        total_params = sum(p.numel() for p in model.parameters())
        # Rough estimate: 2 FLOPs per parameter (multiply-add)
        return total_params * 2
    
    def _estimate_bandwidth_requirements(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        batch_size: int
    ) -> float:
        """Estimate memory bandwidth requirements in GB/s"""
        # Estimate based on parameter size and activation size
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        activation_size = np.prod(input_shape) * 4 * batch_size  # Assuming FP32
        
        # Assume we need to move this data multiple times during inference
        data_movement = (param_size + activation_size) * 10  # Factor of 10 for multiple accesses
        
        # Convert to GB/s assuming 10ms inference time
        return (data_movement / (1024 ** 3)) / 0.01
    
    def _optimize_batch_size(self, required_memory: float, available_memory: float) -> int:
        """Optimize batch size based on available memory"""
        if required_memory <= available_memory * 0.8:  # Leave 20% headroom
            return int(available_memory * 0.8 / (required_memory / 32))  # Scale from base batch size of 32
        else:
            return max(1, int(available_memory * 0.8 / required_memory))
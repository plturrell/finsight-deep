# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-GPU Manager for AIQToolkit
Provides efficient model and data distribution across multiple GPUs
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Tuple
import logging
from dataclasses import dataclass
import psutil
import pynvml

logger = logging.getLogger(__name__)


@dataclass
class GPUInfo:
    """Information about a GPU device"""
    device_id: int
    name: str
    memory_total: int  # bytes
    memory_free: int   # bytes
    utilization: float # percentage
    temperature: float # celsius
    power_draw: float  # watts
    
    @property
    def memory_used(self) -> int:
        return self.memory_total - self.memory_free
    
    @property
    def memory_usage_percent(self) -> float:
        return (self.memory_used / self.memory_total) * 100 if self.memory_total > 0 else 0


class MultiGPUManager:
    """
    Manages multiple GPUs for distributed processing in AIQToolkit
    """
    
    def __init__(self):
        self.device_count = torch.cuda.device_count()
        self.devices = list(range(self.device_count))
        self._init_pynvml()
        
        logger.info(f"Initialized MultiGPUManager with {self.device_count} GPUs")
    
    def get_device_info(self) -> List[Dict[str, Any]]:
        """Get information about available GPU devices"""
        device_info = []
        if torch.cuda.is_available():
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                info = {
                    "device_id": i,
                    "name": props.name,
                    "total_memory": props.total_memory,
                    "compute_capability": f"{props.major}.{props.minor}",
                    "multi_processor_count": props.multi_processor_count
                }
                device_info.append(info)
        return device_info
    
    def _init_pynvml(self):
        """Initialize NVIDIA Management Library"""
        try:
            pynvml.nvmlInit()
            self.nvml_available = True
        except Exception as e:
            logger.warning(f"Failed to initialize NVML: {e}")
            self.nvml_available = False
    
    def get_gpu_info(self, device_id: Optional[int] = None) -> List[GPUInfo]:
        """
        Get information about GPU(s)
        
        Args:
            device_id: Specific GPU device ID, or None for all GPUs
            
        Returns:
            List of GPUInfo objects
        """
        if device_id is not None:
            devices = [device_id]
        else:
            devices = self.devices
            
        gpu_infos = []
        
        for device in devices:
            if device >= self.device_count:
                continue
                
            # Get basic PyTorch info
            props = torch.cuda.get_device_properties(device)
            
            # Get NVML info if available
            if self.nvml_available:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # Convert to watts
                    
                    gpu_info = GPUInfo(
                        device_id=device,
                        name=props.name,
                        memory_total=mem_info.total,
                        memory_free=mem_info.free,
                        utilization=util_info.gpu,
                        temperature=temp,
                        power_draw=power
                    )
                except Exception as e:
                    logger.warning(f"Failed to get NVML info for GPU {device}: {e}")
                    gpu_info = self._get_basic_gpu_info(device, props)
            else:
                gpu_info = self._get_basic_gpu_info(device, props)
                
            gpu_infos.append(gpu_info)
            
        return gpu_infos
    
    def _get_basic_gpu_info(self, device: int, props) -> GPUInfo:
        """Get basic GPU info without NVML"""
        return GPUInfo(
            device_id=device,
            name=props.name,
            memory_total=props.total_memory,
            memory_free=props.total_memory - torch.cuda.memory_allocated(device),
            utilization=0.0,
            temperature=0.0,
            power_draw=0.0
        )
    
    def select_best_gpu(self, min_memory_mb: int = 1024) -> int:
        """
        Select the best GPU based on available memory
        
        Args:
            min_memory_mb: Minimum required memory in MB
            
        Returns:
            Device ID of best GPU
        """
        gpu_infos = self.get_gpu_info()
        
        # Filter GPUs with enough memory
        suitable_gpus = [
            gpu for gpu in gpu_infos 
            if gpu.memory_free >= min_memory_mb * 1024 * 1024
        ]
        
        if not suitable_gpus:
            raise RuntimeError(f"No GPU with at least {min_memory_mb}MB free memory")
        
        # Sort by free memory and utilization
        suitable_gpus.sort(
            key=lambda gpu: (gpu.memory_free, -gpu.utilization),
            reverse=True
        )
        
        selected = suitable_gpus[0]
        logger.info(f"Selected GPU {selected.device_id} ({selected.name}) with "
                   f"{selected.memory_free / (1024**3):.1f}GB free memory")
        
        return selected.device_id
    
    def distribute_model(self, model: nn.Module, devices: Optional[List[int]] = None) -> nn.Module:
        """
        Distribute model across multiple GPUs
        
        Args:
            model: PyTorch model to distribute
            devices: List of device IDs to use, or None for all
            
        Returns:
            Distributed model
        """
        if self.device_count == 0:
            logger.info("No GPUs available, using CPU")
            return model
            
        if self.device_count == 1:
            logger.info("Single GPU available, using standard model")
            return model.cuda()
        
        if devices is None:
            devices = self.devices
            
        if len(devices) == 1:
            logger.info(f"Using single GPU: {devices[0]}")
            return model.to(f'cuda:{devices[0]}')
        
        logger.info(f"Distributing model across GPUs: {devices}")
        
        # Use DataParallel for multiple GPUs
        model = nn.DataParallel(model, device_ids=devices)
        model = model.to(f'cuda:{devices[0]}')
        
        return model
    
    def distribute_data(self, data: torch.Tensor, batch_size: int, devices: Optional[List[int]] = None) -> List[torch.Tensor]:
        """
        Distribute data across multiple GPUs
        
        Args:
            data: Input tensor
            batch_size: Batch size per GPU
            devices: List of device IDs to use
            
        Returns:
            List of tensors on different GPUs
        """
        if devices is None:
            devices = self.devices
            
        chunks = data.split(batch_size, dim=0)
        distributed_data = []
        
        for i, chunk in enumerate(chunks):
            device_id = devices[i % len(devices)]
            distributed_data.append(chunk.to(f'cuda:{device_id}'))
            
        return distributed_data
    
    def gather_results(self, results: List[torch.Tensor], target_device: int = 0) -> torch.Tensor:
        """
        Gather results from multiple GPUs to a single device
        
        Args:
            results: List of result tensors on different devices
            target_device: Target device ID
            
        Returns:
            Concatenated results on target device
        """
        gathered = []
        
        for result in results:
            gathered.append(result.to(f'cuda:{target_device}'))
            
        return torch.cat(gathered, dim=0)
    
    def get_memory_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Get memory usage summary for all GPUs
        
        Returns:
            Dictionary mapping device ID to memory stats
        """
        summary = {}
        
        for device in self.devices:
            gpu_info = self.get_gpu_info(device)[0]
            summary[device] = {
                'total_gb': gpu_info.memory_total / (1024**3),
                'used_gb': gpu_info.memory_used / (1024**3),
                'free_gb': gpu_info.memory_free / (1024**3),
                'usage_percent': gpu_info.memory_usage_percent
            }
            
        return summary
    
    def optimize_batch_size(self, model: nn.Module, input_shape: Tuple[int, ...], 
                          target_memory_usage: float = 0.8) -> int:
        """
        Find optimal batch size for given model and input
        
        Args:
            model: PyTorch model
            input_shape: Shape of single input (without batch dimension)
            target_memory_usage: Target GPU memory usage (0-1)
            
        Returns:
            Optimal batch size
        """
        device = next(model.parameters()).device
        
        # Get available memory
        gpu_info = self.get_gpu_info(device.index)[0]
        available_memory = gpu_info.memory_free * target_memory_usage
        
        # Estimate memory per sample
        dummy_input = torch.zeros(1, *input_shape, device=device)
        
        # Clear cache
        torch.cuda.empty_cache()
        
        # Forward pass to estimate memory
        initial_memory = torch.cuda.memory_allocated(device)
        
        with torch.no_grad():
            _ = model(dummy_input)
            
        memory_per_sample = torch.cuda.memory_allocated(device) - initial_memory
        
        # Calculate optimal batch size
        optimal_batch_size = int(available_memory / memory_per_sample)
        
        # Ensure at least batch size of 1
        optimal_batch_size = max(1, optimal_batch_size)
        
        logger.info(f"Optimal batch size: {optimal_batch_size} "
                   f"(memory per sample: {memory_per_sample / (1024**2):.1f}MB)")
        
        return optimal_batch_size
    
    def cleanup(self):
        """Cleanup NVML"""
        if self.nvml_available:
            try:
                pynvml.nvmlShutdown()
            except Exception as e:
                logger.warning(f"Failed to shutdown NVML: {e}")


# Example usage for AIQToolkit integration
def create_multi_gpu_workflow_runner():
    """
    Create a workflow runner with multi-GPU support
    """
    from aiq.runtime.runner import Runner
    
    class MultiGPURunner(Runner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.gpu_manager = MultiGPUManager()
            
        def run(self, *args, **kwargs):
            # Select best GPU for this run
            if self.gpu_manager.device_count > 1:
                device_id = self.gpu_manager.select_best_gpu()
                # Set CUDA device for this run
                torch.cuda.set_device(device_id)
                
            return super().run(*args, **kwargs)
    
    return MultiGPURunner
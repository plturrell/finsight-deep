# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU-Direct RDMA support for high-performance distributed computing
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Check for NCCL availability
try:
    import torch.distributed as dist
    from torch.distributed import ProcessGroup
    HAS_NCCL = torch.cuda.is_available() and hasattr(dist, 'nccl')
except ImportError:
    HAS_NCCL = False
    logger.warning("NCCL not available, GPU-Direct RDMA will be limited")

# Check for UCX availability (for InfiniBand)
try:
    import ucp
    HAS_UCX = True
except ImportError:
    HAS_UCX = False
    logger.warning("UCX not available, InfiniBand support disabled")


@dataclass
class RDMAConfig:
    """Configuration for RDMA communication"""
    enable_gpu_direct: bool = True
    enable_infiniband: bool = True
    nccl_backend: str = "nccl"
    ucx_tls: str = "rc,cuda"  # Use RC transport with CUDA
    buffer_size: int = 1024 * 1024 * 16  # 16MB default
    num_streams: int = 4
    enable_compression: bool = True
    compression_threshold: int = 1024  # bytes


class GPUDirectManager:
    """Manages GPU-Direct RDMA communication"""
    
    def __init__(self, config: RDMAConfig = None):
        self.config = config or RDMAConfig()
        self.process_group = None
        self.ucx_endpoints: Dict[str, Any] = {}
        self.cuda_streams: List[torch.cuda.Stream] = []
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize CUDA streams
        if torch.cuda.is_available():
            for i in range(self.config.num_streams):
                self.cuda_streams.append(torch.cuda.Stream())
        
        # Check capabilities
        self._check_capabilities()
    
    def _check_capabilities(self):
        """Check system capabilities for GPU-Direct"""
        logger.info("Checking GPU-Direct capabilities...")
        
        if not torch.cuda.is_available():
            logger.warning("CUDA not available")
            return
        
        # Check GPU-Direct RDMA support
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name}")
            
            # Check compute capability (need 3.5+ for GPU-Direct)
            if props.major >= 3 and props.minor >= 5:
                logger.info(f"  GPU-Direct RDMA: Supported")
            else:
                logger.warning(f"  GPU-Direct RDMA: Not supported (compute {props.major}.{props.minor})")
        
        # Check NCCL
        if HAS_NCCL:
            logger.info("NCCL: Available")
            if dist.is_nccl_available():
                logger.info("  NCCL backend: Available")
        else:
            logger.warning("NCCL: Not available")
        
        # Check InfiniBand/UCX
        if HAS_UCX:
            logger.info("UCX: Available")
            logger.info(f"  TLS: {self.config.ucx_tls}")
        else:
            logger.warning("UCX: Not available")
    
    async def initialize_rdma(self, rank: int, world_size: int, master_addr: str):
        """Initialize RDMA communication"""
        logger.info(f"Initializing RDMA for rank {rank}/{world_size}")
        
        # Initialize process group for NCCL
        if HAS_NCCL and self.config.enable_gpu_direct:
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = '29500'
            os.environ['NCCL_DEBUG'] = 'INFO'
            
            # Enable GPU-Direct specific optimizations
            os.environ['NCCL_NET_GDR_LEVEL'] = '5'  # Maximum GPU-Direct usage
            os.environ['NCCL_P2P_DISABLE'] = '0'    # Enable P2P
            
            dist.init_process_group(
                backend=self.config.nccl_backend,
                rank=rank,
                world_size=world_size
            )
            self.process_group = dist.group.WORLD
            
            logger.info("NCCL process group initialized")
        
        # Initialize UCX for InfiniBand
        if HAS_UCX and self.config.enable_infiniband:
            await self._initialize_ucx(rank, world_size)
    
    async def _initialize_ucx(self, rank: int, world_size: int):
        """Initialize UCX for InfiniBand communication"""
        # Configure UCX
        config = {
            'TLS': self.config.ucx_tls,
            'SOCKADDR_TLS_PRIORITY': 'rdmacm',
            'NET_DEVICES': 'all'
        }
        
        # Initialize UCX context
        self.ucx_ctx = ucp.init(config)
        
        # Create listener for incoming connections
        if rank == 0:
            self.ucx_listener = ucp.create_listener(self._handle_ucx_connection)
            addr = self.ucx_listener.get_address()
            logger.info(f"UCX listener at {addr}")
            
            # Exchange addresses (simplified - use proper discovery in production)
            # This would typically use a coordination service
    
    async def _handle_ucx_connection(self, ep):
        """Handle incoming UCX connections"""
        logger.info("Received UCX connection")
        # Store endpoint for future communication
        self.ucx_endpoints[str(ep.get_address())] = ep
    
    def send_tensor_rdma(self, 
                        tensor: torch.Tensor, 
                        dst_rank: int,
                        tag: int = 0) -> asyncio.Future:
        """
        Send tensor using GPU-Direct RDMA
        
        Args:
            tensor: PyTorch tensor to send
            dst_rank: Destination rank
            tag: Communication tag
            
        Returns:
            Future for async completion
        """
        if not HAS_NCCL:
            raise RuntimeError("NCCL not available")
        
        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Use NCCL for GPU-to-GPU communication
        future = asyncio.Future()
        
        def _send():
            try:
                dist.send(tensor, dst=dst_rank, tag=tag)
                future.set_result(True)
            except Exception as e:
                future.set_exception(e)
        
        # Execute in thread pool to avoid blocking
        self.executor.submit(_send)
        
        return future
    
    def recv_tensor_rdma(self,
                        shape: Tuple[int, ...],
                        dtype: torch.dtype,
                        src_rank: int,
                        tag: int = 0) -> asyncio.Future:
        """
        Receive tensor using GPU-Direct RDMA
        
        Args:
            shape: Expected tensor shape
            dtype: Expected tensor dtype
            src_rank: Source rank
            tag: Communication tag
            
        Returns:
            Future containing received tensor
        """
        if not HAS_NCCL:
            raise RuntimeError("NCCL not available")
        
        # Allocate receive buffer on GPU
        tensor = torch.empty(shape, dtype=dtype, device='cuda')
        
        future = asyncio.Future()
        
        def _recv():
            try:
                dist.recv(tensor, src=src_rank, tag=tag)
                future.set_result(tensor)
            except Exception as e:
                future.set_exception(e)
        
        self.executor.submit(_recv)
        
        return future
    
    async def all_reduce_rdma(self, 
                             tensor: torch.Tensor,
                             op: str = 'sum') -> torch.Tensor:
        """
        All-reduce operation using GPU-Direct RDMA
        
        Args:
            tensor: Tensor to reduce
            op: Reduction operation ('sum', 'max', 'min')
            
        Returns:
            Reduced tensor
        """
        if not HAS_NCCL:
            raise RuntimeError("NCCL not available")
        
        # Map operation
        ops = {
            'sum': dist.ReduceOp.SUM,
            'max': dist.ReduceOp.MAX,
            'min': dist.ReduceOp.MIN,
            'product': dist.ReduceOp.PRODUCT
        }
        
        reduce_op = ops.get(op, dist.ReduceOp.SUM)
        
        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Perform all-reduce
        dist.all_reduce(tensor, op=reduce_op)
        
        return tensor
    
    def broadcast_rdma(self,
                      tensor: torch.Tensor,
                      src_rank: int = 0) -> torch.Tensor:
        """
        Broadcast tensor using GPU-Direct RDMA
        
        Args:
            tensor: Tensor to broadcast
            src_rank: Source rank for broadcast
            
        Returns:
            Broadcast tensor
        """
        if not HAS_NCCL:
            raise RuntimeError("NCCL not available")
        
        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Perform broadcast
        dist.broadcast(tensor, src=src_rank)
        
        return tensor
    
    def all_gather_rdma(self,
                       tensor: torch.Tensor,
                       world_size: int) -> List[torch.Tensor]:
        """
        All-gather operation using GPU-Direct RDMA
        
        Args:
            tensor: Local tensor
            world_size: Number of processes
            
        Returns:
            List of gathered tensors
        """
        if not HAS_NCCL:
            raise RuntimeError("NCCL not available")
        
        # Ensure tensor is on GPU
        if not tensor.is_cuda:
            tensor = tensor.cuda()
        
        # Create output tensors
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        
        # Perform all-gather
        dist.all_gather(gathered, tensor)
        
        return gathered
    
    def optimize_nccl_params(self):
        """Optimize NCCL parameters for best performance"""
        # Set NCCL environment variables for optimal performance
        optimizations = {
            'NCCL_NET_GDR_LEVEL': '5',       # Max GPU-Direct usage
            'NCCL_NET_GDR_READ': '1',        # Enable GPU-Direct Read
            'NCCL_P2P_DISABLE': '0',         # Enable P2P
            'NCCL_SHM_DISABLE': '0',         # Enable shared memory
            'NCCL_TREE_THRESHOLD': '0',      # Always use tree algorithm
            'NCCL_LL_THRESHOLD': '0',        # Disable LL for small messages
            'NCCL_BUFF_SIZE': '16777216',    # 16MB buffer
            'NCCL_NTHREADS': '512',          # Increase thread count
            'NCCL_NSOCKS_PERTHREAD': '8',    # Sockets per thread
        }
        
        for key, value in optimizations.items():
            os.environ[key] = value
        
        logger.info("NCCL parameters optimized for GPU-Direct RDMA")
    
    def benchmark_rdma_bandwidth(self, 
                               message_sizes: List[int] = None) -> Dict[int, float]:
        """
        Benchmark RDMA bandwidth for different message sizes
        
        Args:
            message_sizes: List of message sizes to test
            
        Returns:
            Dictionary of size -> bandwidth (GB/s)
        """
        if message_sizes is None:
            message_sizes = [
                1024,           # 1KB
                1024 * 1024,    # 1MB
                10 * 1024 * 1024,  # 10MB
                100 * 1024 * 1024, # 100MB
                1024 * 1024 * 1024 # 1GB
            ]
        
        results = {}
        
        for size in message_sizes:
            # Create test tensor
            tensor = torch.randn(size // 4, device='cuda')  # float32 = 4 bytes
            
            # Warm up
            for _ in range(5):
                self.all_reduce_rdma(tensor.clone())
            
            # Benchmark
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            for _ in range(10):
                self.all_reduce_rdma(tensor.clone())
            end.record()
            
            torch.cuda.synchronize()
            time_ms = start.elapsed_time(end) / 10  # Average per iteration
            
            # Calculate bandwidth (GB/s)
            bandwidth = (size / (1024**3)) / (time_ms / 1000)
            results[size] = bandwidth
            
            logger.info(f"Message size {size/1024**2:.1f}MB: {bandwidth:.2f} GB/s")
        
        return results


class GPUDirectScheduler:
    """Scheduler optimized for GPU-Direct RDMA"""
    
    def __init__(self, gpu_direct_manager: GPUDirectManager):
        self.gpu_direct = gpu_direct_manager
        self.topology: Dict[int, Dict[int, float]] = {}  # rank -> rank -> latency
        
    async def discover_topology(self, world_size: int):
        """Discover network topology and GPU connectivity"""
        logger.info("Discovering network topology...")
        
        rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Test P2P connectivity between GPUs
        for target_rank in range(world_size):
            if target_rank == rank:
                continue
            
            # Measure latency
            latency = await self._measure_latency(target_rank)
            
            if rank not in self.topology:
                self.topology[rank] = {}
            self.topology[rank][target_rank] = latency
            
            logger.info(f"Latency rank {rank} -> {target_rank}: {latency:.3f}ms")
    
    async def _measure_latency(self, target_rank: int, num_trials: int = 10) -> float:
        """Measure communication latency to target rank"""
        # Small message for latency test
        tensor = torch.ones(1, device='cuda')
        
        latencies = []
        
        for _ in range(num_trials):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            await self.gpu_direct.send_tensor_rdma(tensor, target_rank)
            end.record()
            
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
        
        return np.mean(latencies)
    
    def schedule_communication(self, 
                             transfers: List[Tuple[int, int, int]]) -> List[List[Tuple[int, int, int]]]:
        """
        Schedule communication to minimize congestion
        
        Args:
            transfers: List of (src, dst, size) tuples
            
        Returns:
            List of transfer groups to execute in parallel
        """
        # Group transfers to avoid congestion
        # This is a simplified scheduler - production would be more sophisticated
        
        groups = []
        remaining = transfers.copy()
        
        while remaining:
            group = []
            used_nodes = set()
            
            for transfer in remaining[:]:
                src, dst, size = transfer
                
                # Check if nodes are already in use
                if src not in used_nodes and dst not in used_nodes:
                    group.append(transfer)
                    used_nodes.add(src)
                    used_nodes.add(dst)
                    remaining.remove(transfer)
            
            groups.append(group)
        
        return groups


# Example usage
if __name__ == "__main__":
    # Initialize GPU-Direct manager
    config = RDMAConfig(
        enable_gpu_direct=True,
        enable_infiniband=True,
        nccl_backend="nccl"
    )
    
    manager = GPUDirectManager(config)
    
    # Check capabilities
    manager._check_capabilities()
    
    # Benchmark RDMA (would need actual distributed setup)
    if torch.cuda.is_available():
        # This would run in a distributed environment
        # results = manager.benchmark_rdma_bandwidth()
        pass
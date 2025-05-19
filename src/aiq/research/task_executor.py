# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

class TaskType(Enum):
    """Types of research tasks"""
    RETRIEVAL = "retrieval"
    REASONING = "reasoning"
    VERIFICATION = "verification"
    SYNTHESIS = "synthesis"
    EMBEDDING = "embedding"
    CLUSTERING = "clustering"

class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3

@dataclass
class ResearchTask:
    """Research task definition"""
    task_id: str
    task_type: TaskType
    query: str
    target_latency_ms: Optional[int] = None
    priority: int = TaskPriority.MEDIUM.value
    input_data: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class TaskResult:
    """Result of research task execution"""
    task_id: str
    result_data: Any
    execution_time_ms: float
    gpu_utilization: float
    memory_usage_mb: float
    success: bool
    error_message: Optional[str] = None

class GPUOptimizer:
    """GPU optimization utilities"""
    def __init__(self, use_tensor_cores: bool = True):
        self.use_tensor_cores = use_tensor_cores
        self._setup_optimization()
    
    def _setup_optimization(self):
        """Configure GPU optimization settings"""
        if torch.cuda.is_available():
            # Enable Tensor Core optimization
            if self.use_tensor_cores:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # Check for Hopper architecture (compute capability 9.0+)
                capability = torch.cuda.get_device_capability()
                if capability[0] >= 9:
                    # Enable FP8 for Hopper GPUs
                    torch.backends.cuda.matmul.allow_fp8_e4m3 = True
    
    def optimize_batch_size(self, model_memory_mb: float, available_memory_mb: float) -> int:
        """Calculate optimal batch size based on available memory"""
        # Reserve 20% for overhead
        usable_memory = available_memory_mb * 0.8
        optimal_batch_size = max(1, int(usable_memory / model_memory_mb))
        return optimal_batch_size

class CUDAStreamManager:
    """Manages CUDA streams for parallel execution"""
    def __init__(self, num_streams: int = 4):
        self.num_streams = num_streams
        self.streams = []
        self._initialize_streams()
    
    def _initialize_streams(self):
        """Initialize CUDA streams"""
        if torch.cuda.is_available():
            for _ in range(self.num_streams):
                self.streams.append(torch.cuda.Stream())
    
    def get_stream(self, task_id: str) -> torch.cuda.Stream:
        """Get a CUDA stream for task execution"""
        if not self.streams:
            return torch.cuda.default_stream()
        
        # Simple round-robin allocation
        stream_idx = hash(task_id) % len(self.streams)
        return self.streams[stream_idx]

class ResearchTaskExecutor:
    """
    Implements GPU-optimized research task execution with CUDA kernels
    """
    def __init__(
        self,
        num_gpus: Optional[int] = None,
        enable_optimization: bool = True,
        use_tensor_cores: bool = True,
        max_concurrent_tasks: int = 32
    ):
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.enable_optimization = enable_optimization
        self.gpu_optimizer = GPUOptimizer(use_tensor_cores=use_tensor_cores)
        self.stream_manager = CUDAStreamManager()
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Task execution components
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.executor = ThreadPoolExecutor(max_workers=self.max_concurrent_tasks)
        self.active_tasks: Dict[str, asyncio.Task] = {}
        
        self._setup_cuda_environment()
        logger.info(f"Initialized ResearchTaskExecutor with {self.num_gpus} GPUs")
    
    def _setup_cuda_environment(self):
        """Setup CUDA environment for all GPUs"""
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU execution")
            return
        
        # Initialize primary GPU
        torch.cuda.set_device(0)
        
        # Warm up CUDA kernels
        for i in range(self.num_gpus):
            with torch.cuda.device(i):
                # Allocate small tensor to initialize context
                torch.zeros(1, device=f'cuda:{i}')
    
    async def execute_task(self, task: ResearchTask) -> TaskResult:
        """Execute a research task with GPU optimization"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Select GPU based on task priority and current load
            device_id = self._select_gpu(task.priority)
            
            # Get CUDA stream for this task
            stream = self.stream_manager.get_stream(task.task_id)
            
            # Execute task based on type
            with torch.cuda.device(device_id):
                with torch.cuda.stream(stream):
                    result_data = await self._execute_task_type(task)
            
            # Wait for stream completion
            stream.synchronize()
            
            # Calculate metrics
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            gpu_utilization = self._get_gpu_utilization(device_id)
            memory_usage = self._get_memory_usage(device_id)
            
            return TaskResult(
                task_id=task.task_id,
                result_data=result_data,
                execution_time_ms=execution_time,
                gpu_utilization=gpu_utilization,
                memory_usage_mb=memory_usage,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {str(e)}")
            execution_time = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return TaskResult(
                task_id=task.task_id,
                result_data=None,
                execution_time_ms=execution_time,
                gpu_utilization=0.0,
                memory_usage_mb=0.0,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_task_type(self, task: ResearchTask) -> Any:
        """Execute specific task type"""
        if task.task_type == TaskType.RETRIEVAL:
            return await self._execute_retrieval(task)
        elif task.task_type == TaskType.REASONING:
            return await self._execute_reasoning(task)
        elif task.task_type == TaskType.VERIFICATION:
            return await self._execute_verification(task)
        elif task.task_type == TaskType.SYNTHESIS:
            return await self._execute_synthesis(task)
        elif task.task_type == TaskType.EMBEDDING:
            return await self._execute_embedding(task)
        elif task.task_type == TaskType.CLUSTERING:
            return await self._execute_clustering(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    async def _execute_retrieval(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute retrieval task with GPU acceleration"""
        # Placeholder for actual retrieval implementation
        # In production, this would use vector similarity search on GPU
        query_embedding = torch.randn(768, device='cuda')
        corpus_embeddings = torch.randn(10000, 768, device='cuda')
        
        # GPU-accelerated similarity computation
        similarities = torch.matmul(query_embedding, corpus_embeddings.T)
        top_k_values, top_k_indices = torch.topk(similarities, k=10)
        
        return {
            "top_k_indices": top_k_indices.cpu().tolist(),
            "scores": top_k_values.cpu().tolist()
        }
    
    async def _execute_reasoning(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute reasoning task"""
        # Placeholder for reasoning implementation
        return {"reasoning_result": f"Processed: {task.query}"}
    
    async def _execute_verification(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute verification task"""
        # Placeholder for verification implementation
        return {"verification_result": "Verified", "confidence": 0.95}
    
    async def _execute_synthesis(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute synthesis task"""
        # Placeholder for synthesis implementation
        return {"synthesized_content": f"Synthesis of: {task.query}"}
    
    async def _execute_embedding(self, task: ResearchTask) -> Dict[str, Any]:
        """Generate embeddings using GPU"""
        # Placeholder - in production, use actual embedding model
        text = task.query
        embedding = torch.randn(768, device='cuda')
        
        return {
            "embedding": embedding.cpu().numpy().tolist(),
            "dimension": 768
        }
    
    async def _execute_clustering(self, task: ResearchTask) -> Dict[str, Any]:
        """Execute clustering task on GPU"""
        # Placeholder for GPU-accelerated clustering
        data_points = task.input_data.get("data_points", [])
        num_clusters = task.input_data.get("num_clusters", 5)
        
        # In production, use GPU-accelerated clustering (e.g., RAPIDS)
        return {
            "clusters": list(range(num_clusters)),
            "num_points": len(data_points)
        }
    
    def _select_gpu(self, priority: int) -> int:
        """Select GPU based on priority and current load"""
        if self.num_gpus == 0:
            return 0
        
        # Simple round-robin for now, could be enhanced with load balancing
        # High priority tasks get GPU 0, others distributed
        if priority >= TaskPriority.HIGH.value:
            return 0
        else:
            return np.random.randint(0, self.num_gpus)
    
    def _get_gpu_utilization(self, device_id: int) -> float:
        """Get GPU utilization for specific device"""
        if not torch.cuda.is_available():
            return 0.0
        
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            return util.gpu / 100.0
        except:
            # Fallback method
            return 0.5  # Placeholder
    
    def _get_memory_usage(self, device_id: int) -> float:
        """Get memory usage in MB for specific device"""
        if not torch.cuda.is_available():
            return 0.0
        
        with torch.cuda.device(device_id):
            return torch.cuda.memory_allocated() / (1024 * 1024)
    
    async def batch_execute(self, tasks: List[ResearchTask]) -> List[TaskResult]:
        """Execute multiple tasks in batch for efficiency"""
        # Group tasks by type for better GPU utilization
        tasks_by_type = {}
        for task in tasks:
            if task.task_type not in tasks_by_type:
                tasks_by_type[task.task_type] = []
            tasks_by_type[task.task_type].append(task)
        
        all_results = []
        
        # Execute each type in parallel
        for task_type, typed_tasks in tasks_by_type.items():
            # Execute tasks of same type concurrently
            tasks_coro = [self.execute_task(task) for task in typed_tasks]
            results = await asyncio.gather(*tasks_coro)
            all_results.extend(results)
        
        return all_results
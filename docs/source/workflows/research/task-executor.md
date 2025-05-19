# Research Task Executor

## Overview

The ResearchTaskExecutor is a high-performance component for executing research tasks with GPU acceleration, multi-GPU support, and advanced optimization features. It provides efficient task management, resource allocation, and performance monitoring.

## Key Features

- **GPU Acceleration**: Leverages NVIDIA GPUs for faster execution
- **Multi-GPU Support**: Distributes tasks across multiple GPUs
- **Stream Management**: Efficient CUDA stream utilization
- **Task Prioritization**: Priority-based task scheduling
- **Resource Optimization**: Dynamic resource allocation
- **Performance Monitoring**: Real-time metrics and profiling

## Basic Usage

```python
from aiq.research import ResearchTaskExecutor, ResearchTask, TaskType, TaskPriority

# Initialize executor
executor = ResearchTaskExecutor(
    num_gpus=2,
    enable_optimization=True,
    use_tensor_cores=True,
    max_concurrent_tasks=32
)

# Create a task
task = ResearchTask(
    task_id="research_001",
    task_type=TaskType.RETRIEVAL,
    query="Latest advances in transformer architectures",
    priority=TaskPriority.HIGH,
    target_latency_ms=100
)

# Execute task
result = await executor.execute_task(task)
print(f"Result: {result.result_data}")
print(f"Execution time: {result.execution_time_ms}ms")
```

## Task Configuration

### Task Structure

```python
@dataclass
class ResearchTask:
    task_id: str                          # Unique identifier
    task_type: TaskType                   # Type of research task
    query: str                            # Research query
    priority: int = TaskPriority.MEDIUM   # Task priority
    target_latency_ms: Optional[int]      # Target latency
    input_data: Optional[Dict[str, Any]]  # Additional input
    metadata: Optional[Dict[str, Any]]    # Task metadata
```

### Task Types

```python
class TaskType(Enum):
    RETRIEVAL = "retrieval"        # Information retrieval
    REASONING = "reasoning"        # Logical reasoning
    VERIFICATION = "verification"  # Fact-checking
    SYNTHESIS = "synthesis"        # Content generation
    EMBEDDING = "embedding"        # Vector generation
    CLUSTERING = "clustering"      # Data clustering
```

### Priority Levels

```python
class TaskPriority(Enum):
    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3
```

## GPU Optimization

### Tensor Core Configuration

```python
# Enable Tensor Core optimization
executor = ResearchTaskExecutor(
    use_tensor_cores=True,
    mixed_precision=True
)

# Configure for specific GPU architecture
optimizer = GPUOptimizer(use_tensor_cores=True)
optimizer.configure_for_architecture("hopper")  # H100
```

### Multi-GPU Distribution

```python
# Automatic GPU selection
executor = ResearchTaskExecutor(num_gpus=torch.cuda.device_count())

# Manual GPU assignment
task.metadata = {"preferred_gpu": 1}
result = await executor.execute_task(task)

# GPU load balancing
executor.enable_load_balancing(
    strategy="least_loaded",
    rebalance_interval=60  # seconds
)
```

### CUDA Stream Management

```python
# Configure stream management
stream_manager = CUDAStreamManager(num_streams=8)
executor.set_stream_manager(stream_manager)

# Execute with specific stream
task.metadata = {"cuda_stream": 2}
result = await executor.execute_task(task)
```

## Task Execution

### Single Task Execution

```python
# Simple execution
result = await executor.execute_task(task)

# With timeout
result = await executor.execute_task(
    task,
    timeout_seconds=30
)

# With callbacks
async def on_progress(progress: float):
    print(f"Progress: {progress:.1%}")

result = await executor.execute_task(
    task,
    progress_callback=on_progress
)
```

### Batch Execution

```python
# Create multiple tasks
tasks = [
    ResearchTask(
        task_id=f"batch_{i}",
        task_type=TaskType.RETRIEVAL,
        query=f"Query {i}",
        priority=TaskPriority.MEDIUM
    )
    for i in range(100)
]

# Execute in batch
results = await executor.batch_execute(tasks)

# With concurrency limit
results = await executor.batch_execute(
    tasks,
    max_concurrent=10
)
```

### Streaming Execution

```python
# Stream results as they complete
async for result in executor.stream_execute(tasks):
    print(f"Completed: {result.task_id}")
    print(f"Time: {result.execution_time_ms}ms")
```

## Task Handlers

### Default Handlers

```python
# Built-in handlers for each task type
async def _execute_retrieval(self, task: ResearchTask) -> Any:
    """Execute retrieval task with GPU acceleration"""
    query_embedding = await self.generate_embedding(task.query)
    results = await self.search_vectors(query_embedding)
    return {"results": results, "scores": scores}

async def _execute_reasoning(self, task: ResearchTask) -> Any:
    """Execute reasoning task"""
    context = task.input_data.get("context", {})
    reasoning_chain = await self.reason(task.query, context)
    return {"reasoning": reasoning_chain, "conclusion": conclusion}
```

### Custom Handlers

```python
# Register custom task handler
@executor.register_handler(TaskType.CUSTOM)
async def custom_handler(task: ResearchTask) -> Dict[str, Any]:
    # Custom implementation
    result = await process_custom_task(task)
    return {"custom_result": result}

# Use custom task type
task = ResearchTask(
    task_id="custom_001",
    task_type=TaskType.CUSTOM,
    query="Custom processing request"
)
```

## Resource Management

### Memory Optimization

```python
# Configure memory limits
executor.configure_memory(
    max_memory_per_task_mb=512,
    enable_memory_pooling=True,
    cache_size_mb=1024
)

# Memory-aware batch sizing
optimal_batch_size = executor.calculate_optimal_batch_size(
    model_memory_mb=256,
    available_memory_mb=8192
)
```

### GPU Resource Allocation

```python
# Set GPU memory fraction
executor.set_gpu_memory_fraction(0.8)  # Use 80% of GPU memory

# Dynamic allocation
executor.enable_dynamic_allocation(
    min_memory_mb=512,
    max_memory_mb=4096,
    growth_factor=1.5
)
```

## Performance Monitoring

### Metrics Collection

```python
# Enable metrics
executor.enable_metrics(
    collect_gpu_metrics=True,
    collect_memory_metrics=True,
    collect_timing_metrics=True
)

# Execute with metrics
result = await executor.execute_task(task)

# Get metrics
metrics = result.metrics
print(f"GPU utilization: {metrics['gpu_utilization']:.1%}")
print(f"Memory usage: {metrics['memory_usage_mb']:.1f}MB")
print(f"Kernel time: {metrics['kernel_time_ms']:.1f}ms")
```

### Real-time Monitoring

```python
# Start monitoring server
executor.start_monitoring(
    port=8080,
    update_interval=1.0  # seconds
)

# Access metrics via HTTP
# GET http://localhost:8080/metrics
# GET http://localhost:8080/tasks
# GET http://localhost:8080/gpus
```

### Performance Profiling

```python
# Enable profiling
with executor.profile() as profiler:
    result = await executor.execute_task(task)

# Get profile data
profile = profiler.get_profile()
print(f"Total time: {profile['total_time_ms']:.1f}ms")
print(f"GPU time: {profile['gpu_time_ms']:.1f}ms")
print(f"CPU time: {profile['cpu_time_ms']:.1f}ms")

# Export profile
profiler.export_chrome_trace("profile.json")
```

## Advanced Features

### Task Dependencies

```python
# Create dependent tasks
task1 = ResearchTask(task_id="task1", task_type=TaskType.RETRIEVAL, query="Query 1")
task2 = ResearchTask(
    task_id="task2",
    task_type=TaskType.REASONING,
    query="Analyze results",
    input_data={"depends_on": ["task1"]}
)

# Execute with dependency resolution
results = await executor.execute_with_dependencies([task1, task2])
```

### Checkpointing

```python
# Enable checkpointing
executor.enable_checkpointing(
    checkpoint_dir="/tmp/research_checkpoints",
    checkpoint_interval=60  # seconds
)

# Resume from checkpoint
task = ResearchTask(
    task_id="long_running",
    task_type=TaskType.SYNTHESIS,
    query="Complex analysis"
)

result = await executor.execute_task(
    task,
    resume_from_checkpoint=True
)
```

### Fault Tolerance

```python
# Configure retry policy
executor.set_retry_policy(
    max_retries=3,
    retry_delay=1.0,
    exponential_backoff=True
)

# Enable fault recovery
executor.enable_fault_recovery(
    checkpoint_on_failure=True,
    preserve_partial_results=True
)
```

## Integration Examples

### With Verification System

```python
from aiq.verification import VerificationSystem

class VerifiedTaskExecutor(ResearchTaskExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = VerificationSystem()
    
    async def _execute_with_verification(self, task: ResearchTask):
        # Execute task
        result = await self._execute_task_type(task)
        
        # Verify if applicable
        if task.task_type in [TaskType.RETRIEVAL, TaskType.SYNTHESIS]:
            verification = await self.verifier.verify_claim(
                claim=result.get("content", ""),
                sources=result.get("sources", [])
            )
            result["verification"] = verification
        
        return result
```

### With Self-Correction

```python
from aiq.correction import SelfCorrectingResearchSystem

class SelfCorrectingExecutor(ResearchTaskExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corrector = SelfCorrectingResearchSystem()
    
    async def _execute_with_correction(self, task: ResearchTask):
        # Execute task
        result = await self._execute_task_type(task)
        
        # Apply self-correction
        if task.input_data.get("enable_correction", True):
            corrected = await self.corrector.process_query(
                query=task.query,
                initial_content=result.get("content", "")
            )
            result["corrected"] = corrected
        
        return result
```

## Best Practices

### Performance Optimization

1. **Batch Small Tasks**: Group small tasks for better GPU utilization
2. **Use Appropriate Task Types**: Choose the right task type for better routing
3. **Enable Caching**: Cache embeddings and intermediate results
4. **Monitor GPU Memory**: Prevent out-of-memory errors
5. **Profile Regularly**: Identify performance bottlenecks

### Error Handling

```python
try:
    result = await executor.execute_task(task)
except TaskExecutionError as e:
    logger.error(f"Task failed: {e.task_id}")
    # Handle specific error
except GPUMemoryError as e:
    logger.error(f"GPU memory error: {e}")
    # Free memory and retry
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # General error handling
```

### Resource Cleanup

```python
# Proper cleanup
async def cleanup():
    await executor.shutdown()
    torch.cuda.empty_cache()
    
# Using context manager
async with ResearchTaskExecutor() as executor:
    result = await executor.execute_task(task)
    # Automatic cleanup on exit
```

## Configuration Reference

```python
# Full configuration example
config = {
    "num_gpus": 4,
    "enable_optimization": True,
    "use_tensor_cores": True,
    "max_concurrent_tasks": 32,
    "memory_config": {
        "max_memory_per_task_mb": 1024,
        "enable_memory_pooling": True,
        "cache_size_mb": 2048
    },
    "stream_config": {
        "num_streams": 8,
        "stream_priority": "high"
    },
    "monitoring_config": {
        "enable_metrics": True,
        "metrics_port": 8080,
        "update_interval": 1.0
    },
    "retry_config": {
        "max_retries": 3,
        "retry_delay": 1.0,
        "exponential_backoff": True
    }
}

executor = ResearchTaskExecutor(**config)
```
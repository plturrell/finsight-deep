# Distributed Processing Examples

This directory contains examples demonstrating AIQToolkit's distributed processing capabilities.

## Features

- Multi-GPU parallel processing
- Automatic task distribution and load balancing
- GPU memory management and monitoring
- Integration with existing AIQToolkit workflows

## Requirements

- Multiple NVIDIA GPUs (for full functionality)
- CUDA 12.0+
- PyTorch with CUDA support
- pynvml (for GPU monitoring)

## Examples

### 1. Distributed Text Analysis

Demonstrates parallel text analysis across multiple GPUs:

```python
python distributed_text_analysis.py
```

This example:
- Splits text data into batches
- Distributes batches across available GPUs
- Runs LLM inference in parallel
- Collects and aggregates results
- Reports GPU usage and performance metrics

### 2. Multi-GPU Workflow (Coming Soon)

Example showing complex workflow with multiple parallel stages.

## How It Works

1. **GPU Detection**: Automatically detects available GPUs
2. **Task Distribution**: Assigns tasks to GPUs based on availability
3. **Parallel Execution**: Runs tasks concurrently on different devices
4. **Result Aggregation**: Collects results from all GPUs
5. **Performance Monitoring**: Tracks GPU usage and execution times

## Integration with AIQToolkit

The distributed processing seamlessly integrates with existing AIQToolkit components:

```python
from aiq.builder.distributed_workflow_builder import DistributedWorkflowBuilder

# Create distributed workflow
builder = DistributedWorkflowBuilder()
workflow = builder.build(config)

# Run with automatic GPU distribution
results = await workflow.run(session, inputs)
```

## Performance Tips

1. **Batch Size**: Adjust batch sizes based on GPU memory
2. **Load Balancing**: Distribute work evenly across GPUs
3. **Memory Management**: Monitor GPU memory usage
4. **Communication**: Minimize data transfer between GPUs

## Limitations

- Currently supports single-node multi-GPU setups
- Multi-node distributed processing is in development
- Best performance with similar GPU types

## Multi-Node Support (NEW)

AIQToolkit now supports true distributed processing across multiple machines:

### Starting a Distributed Cluster

1. **Start the Manager Node**:
```bash
python -m aiq.distributed.multi_node_example --mode manager --manager-port 50051
```

2. **Start Worker Nodes** (on different machines):
```bash
python -m aiq.distributed.multi_node_example \
    --mode worker \
    --manager-host <manager-ip> \
    --manager-port 50051 \
    --worker-port 50052
```

3. **Run Distributed Workflows**:
```bash
python examples/distributed_processing/multi_node_example.py --mode workflow
```

### Using the Cluster Script

For local testing, use the cluster script:
```bash
./scripts/start_distributed_cluster.sh
```

This starts:
- 1 manager node
- 3 worker nodes (configurable)
- All on localhost for testing

### Architecture

```
┌─────────────────┐
│ Manager Node    │
│ - Coordinates   │
│ - Schedules     │
│ - Monitors      │
└────────┬────────┘
         │
   ┌─────┴─────┬─────────┬─────────┐
   │           │         │         │
┌──┴───┐  ┌───┴──┐  ┌───┴──┐  ┌───┴──┐
│Worker│  │Worker│  │Worker│  │Worker│
│Node 1│  │Node 2│  │Node 3│  │Node N│
└──────┘  └──────┘  └──────┘  └──────┘
```

### Workflow Configuration

Enable multi-node processing:
```python
builder = DistributedWorkflowBuilder(enable_multi_node=True)
```

### Features

- **Automatic Node Discovery**: Workers register with manager
- **Health Monitoring**: Heartbeat checks every 10 seconds  
- **Task Scheduling**: Smart assignment based on resources
- **Fault Detection**: Marks offline nodes automatically
- **Load Balancing**: Distributes tasks evenly

## Future Enhancements

- Advanced scheduling algorithms
- Fault recovery with task reassignment
- Distributed model training
- GPU-Direct RDMA support
- Kubernetes operator
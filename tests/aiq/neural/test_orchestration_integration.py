"""Tests for neural orchestration integration."""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import numpy as np
import torch

from aiq.neural.orchestration_integration import (
    SupercomputerOrchestrator,
    GPUCluster,
    WorkloadDistributor,
    ResourceMonitor,
    TaskScheduler,
    DistributedTraining
)


class TestSupercomputerOrchestrator:
    """Test supercomputer orchestration functionality."""
    
    @pytest.fixture
    def orchestrator(self):
        """Create orchestrator instance."""
        return SupercomputerOrchestrator(
            num_nodes=4,
            gpus_per_node=8,
            interconnect="InfiniBand"
        )
    
    @pytest.fixture
    def mock_gpu_cluster(self):
        """Create mock GPU cluster."""
        cluster = Mock(spec=GPUCluster)
        cluster.nodes = [Mock() for _ in range(4)]
        cluster.total_gpus = 32
        cluster.available_gpus = 28
        return cluster
    
    @pytest.mark.asyncio
    async def test_initialization(self, orchestrator):
        """Test orchestrator initialization."""
        assert orchestrator.num_nodes == 4
        assert orchestrator.gpus_per_node == 8
        assert orchestrator.total_gpus == 32
        assert orchestrator.interconnect == "InfiniBand"
    
    @pytest.mark.asyncio
    async def test_resource_allocation(self, orchestrator, mock_gpu_cluster):
        """Test resource allocation."""
        orchestrator.cluster = mock_gpu_cluster
        
        # Request resources
        resources = await orchestrator.allocate_resources(
            num_gpus=16,
            memory_gb=256,
            cpu_cores=64
        )
        
        assert resources is not None
        assert resources['gpus'] == 16
        assert resources['memory_gb'] == 256
        assert resources['cpu_cores'] == 64
    
    @pytest.mark.asyncio
    async def test_job_submission(self, orchestrator):
        """Test job submission and scheduling."""
        job = {
            'id': 'job_001',
            'type': 'training',
            'model': 'llama3-70b',
            'gpus_required': 8,
            'priority': 'high'
        }
        
        with patch.object(orchestrator.scheduler, 'submit_job') as mock_submit:
            mock_submit.return_value = {'status': 'queued', 'position': 1}
            
            result = await orchestrator.submit_job(job)
            
            assert result['status'] == 'queued'
            assert result['position'] == 1
            mock_submit.assert_called_once_with(job)
    
    @pytest.mark.asyncio
    async def test_distributed_training(self, orchestrator):
        """Test distributed training setup."""
        model = Mock()
        dataset = Mock()
        
        with patch('torch.distributed.init_process_group') as mock_init:
            with patch('torch.nn.parallel.DistributedDataParallel') as mock_ddp:
                distributed_model = await orchestrator.setup_distributed_training(
                    model=model,
                    dataset=dataset,
                    batch_size=256,
                    num_gpus=16
                )
                
                mock_init.assert_called_once()
                mock_ddp.assert_called_once_with(model)
    
    @pytest.mark.asyncio
    async def test_monitoring(self, orchestrator, mock_gpu_cluster):
        """Test resource monitoring."""
        orchestrator.cluster = mock_gpu_cluster
        
        with patch.object(orchestrator.monitor, 'get_metrics') as mock_metrics:
            mock_metrics.return_value = {
                'gpu_utilization': 0.85,
                'memory_usage': 0.72,
                'network_throughput': 98.5,
                'temperature': 65.0
            }
            
            metrics = await orchestrator.get_cluster_metrics()
            
            assert metrics['gpu_utilization'] == 0.85
            assert metrics['memory_usage'] == 0.72
            assert metrics['network_throughput'] == 98.5
            assert metrics['temperature'] == 65.0
    
    @pytest.mark.asyncio
    async def test_failure_recovery(self, orchestrator):
        """Test failure recovery mechanisms."""
        # Simulate node failure
        failed_node = 2
        
        with patch.object(orchestrator, 'detect_failure') as mock_detect:
            mock_detect.return_value = {'node': failed_node, 'type': 'hardware'}
            
            with patch.object(orchestrator, 'recover_from_failure') as mock_recover:
                mock_recover.return_value = True
                
                recovery_result = await orchestrator.handle_node_failure(failed_node)
                
                assert recovery_result is True
                mock_recover.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_balancing(self, orchestrator):
        """Test load balancing across nodes."""
        workload = {
            'tasks': [Mock() for _ in range(100)],
            'total_compute': 1000.0
        }
        
        distribution = await orchestrator.balance_workload(workload)
        
        # Should distribute evenly across 4 nodes
        assert len(distribution) == 4
        assert all(len(tasks) == 25 for tasks in distribution.values())
    
    @pytest.mark.asyncio
    async def test_checkpointing(self, orchestrator):
        """Test distributed checkpointing."""
        checkpoint_data = {
            'model_state': torch.randn(1000, 1000),
            'optimizer_state': {'lr': 0.001},
            'epoch': 10
        }
        
        with patch('torch.save') as mock_save:
            await orchestrator.save_checkpoint(checkpoint_data, 'checkpoint_10.pt')
            mock_save.assert_called_once()
        
        with patch('torch.load') as mock_load:
            mock_load.return_value = checkpoint_data
            loaded = await orchestrator.load_checkpoint('checkpoint_10.pt')
            assert loaded['epoch'] == 10


class TestGPUCluster:
    """Test GPU cluster management."""
    
    @pytest.fixture
    def gpu_cluster(self):
        """Create GPU cluster instance."""
        return GPUCluster(
            nodes=[
                {'id': f'node_{i}', 'gpus': 8, 'memory_gb': 640}
                for i in range(4)
            ]
        )
    
    def test_initialization(self, gpu_cluster):
        """Test cluster initialization."""
        assert gpu_cluster.num_nodes == 4
        assert gpu_cluster.total_gpus == 32
        assert gpu_cluster.total_memory_gb == 2560
    
    def test_node_allocation(self, gpu_cluster):
        """Test node allocation."""
        allocation = gpu_cluster.allocate_nodes(
            num_gpus=16,
            memory_gb=1280
        )
        
        assert len(allocation) == 2  # Should allocate 2 nodes
        assert sum(node['gpus'] for node in allocation) == 16
    
    def test_gpu_topology(self, gpu_cluster):
        """Test GPU topology mapping."""
        topology = gpu_cluster.get_topology()
        
        assert 'nodes' in topology
        assert 'interconnects' in topology
        assert len(topology['nodes']) == 4
    
    def test_health_check(self, gpu_cluster):
        """Test cluster health checking."""
        with patch.object(gpu_cluster, 'check_node_health') as mock_check:
            mock_check.return_value = True
            
            health_status = gpu_cluster.health_check()
            
            assert health_status['healthy'] is True
            assert health_status['failed_nodes'] == []
    
    def test_dynamic_scaling(self, gpu_cluster):
        """Test dynamic cluster scaling."""
        # Add new node
        new_node = {'id': 'node_4', 'gpus': 8, 'memory_gb': 640}
        gpu_cluster.add_node(new_node)
        
        assert gpu_cluster.num_nodes == 5
        assert gpu_cluster.total_gpus == 40
        
        # Remove node
        gpu_cluster.remove_node('node_4')
        
        assert gpu_cluster.num_nodes == 4
        assert gpu_cluster.total_gpus == 32


class TestWorkloadDistributor:
    """Test workload distribution."""
    
    @pytest.fixture
    def distributor(self):
        """Create workload distributor."""
        return WorkloadDistributor(num_workers=4)
    
    @pytest.mark.asyncio
    async def test_task_distribution(self, distributor):
        """Test task distribution across workers."""
        tasks = [Mock() for _ in range(100)]
        
        distribution = await distributor.distribute_tasks(tasks)
        
        # Should distribute evenly
        assert len(distribution) == 4
        assert all(len(worker_tasks) == 25 for worker_tasks in distribution)
    
    @pytest.mark.asyncio
    async def test_data_parallelism(self, distributor):
        """Test data parallel distribution."""
        dataset = Mock()
        dataset.size = 10000
        
        partitions = await distributor.partition_dataset(
            dataset,
            batch_size=256
        )
        
        assert len(partitions) == 4
        assert sum(p.size for p in partitions) == 10000
    
    @pytest.mark.asyncio
    async def test_model_parallelism(self, distributor):
        """Test model parallel distribution."""
        model = Mock()
        model.num_layers = 48
        
        layer_distribution = await distributor.distribute_model_layers(
            model,
            num_gpus=8
        )
        
        # Should distribute 6 layers per GPU
        assert len(layer_distribution) == 8
        assert all(len(layers) == 6 for layers in layer_distribution.values())
    
    @pytest.mark.asyncio
    async def test_pipeline_parallelism(self, distributor):
        """Test pipeline parallel distribution."""
        stages = ['embed', 'encode', 'decode', 'output']
        
        pipeline = await distributor.setup_pipeline(
            stages=stages,
            num_gpus=4
        )
        
        assert len(pipeline) == 4
        assert all(stage in pipeline for stage in stages)
    
    @pytest.mark.asyncio
    async def test_dynamic_load_balancing(self, distributor):
        """Test dynamic load balancing."""
        # Simulate uneven workload
        worker_loads = [0.9, 0.3, 0.5, 0.7]  # Utilization
        
        with patch.object(distributor, 'get_worker_loads') as mock_loads:
            mock_loads.return_value = worker_loads
            
            rebalanced = await distributor.rebalance_workload()
            
            # Should move work from worker 0 to worker 1
            assert rebalanced['moved_tasks'] > 0
            assert rebalanced['from_worker'] == 0
            assert rebalanced['to_worker'] == 1


class TestResourceMonitor:
    """Test resource monitoring."""
    
    @pytest.fixture
    def monitor(self):
        """Create resource monitor."""
        return ResourceMonitor(
            nodes=['node_0', 'node_1', 'node_2', 'node_3'],
            update_interval=1.0
        )
    
    @pytest.mark.asyncio
    async def test_gpu_monitoring(self, monitor):
        """Test GPU monitoring."""
        with patch('pynvml.nvmlDeviceGetUtilizationRates') as mock_util:
            mock_util.return_value = Mock(gpu=75, memory=60)
            
            gpu_metrics = await monitor.get_gpu_metrics()
            
            assert gpu_metrics['utilization'] == 75
            assert gpu_metrics['memory_usage'] == 60
    
    @pytest.mark.asyncio
    async def test_network_monitoring(self, monitor):
        """Test network monitoring."""
        with patch.object(monitor, 'measure_bandwidth') as mock_bandwidth:
            mock_bandwidth.return_value = {
                'throughput_gbps': 100.0,
                'latency_us': 0.5
            }
            
            network_metrics = await monitor.get_network_metrics()
            
            assert network_metrics['throughput_gbps'] == 100.0
            assert network_metrics['latency_us'] == 0.5
    
    @pytest.mark.asyncio
    async def test_alert_system(self, monitor):
        """Test alert system for resource issues."""
        # Set up alert thresholds
        monitor.set_alert_threshold('gpu_utilization', 90.0)
        monitor.set_alert_threshold('temperature', 80.0)
        
        # Simulate high resource usage
        metrics = {
            'gpu_utilization': 95.0,
            'temperature': 85.0,
            'memory_usage': 70.0
        }
        
        with patch.object(monitor, 'get_metrics') as mock_metrics:
            mock_metrics.return_value = metrics
            
            alerts = await monitor.check_alerts()
            
            assert len(alerts) == 2
            assert any(alert['metric'] == 'gpu_utilization' for alert in alerts)
            assert any(alert['metric'] == 'temperature' for alert in alerts)
    
    @pytest.mark.asyncio
    async def test_historical_tracking(self, monitor):
        """Test historical metric tracking."""
        # Simulate collecting metrics over time
        for i in range(10):
            with patch.object(monitor, 'get_metrics') as mock_metrics:
                mock_metrics.return_value = {
                    'gpu_utilization': 70 + i,
                    'timestamp': i
                }
                await monitor.collect_metrics()
        
        history = monitor.get_metric_history('gpu_utilization')
        assert len(history) == 10
        assert history[-1]['value'] == 79


class TestTaskScheduler:
    """Test task scheduling."""
    
    @pytest.fixture
    def scheduler(self):
        """Create task scheduler."""
        return TaskScheduler(
            num_queues=3,  # high, medium, low priority
            preemption_enabled=True
        )
    
    @pytest.mark.asyncio
    async def test_job_submission(self, scheduler):
        """Test job submission and queuing."""
        job = {
            'id': 'job_001',
            'priority': 'high',
            'resources': {'gpus': 4, 'memory_gb': 128},
            'estimated_time': 3600
        }
        
        result = await scheduler.submit_job(job)
        
        assert result['status'] == 'queued'
        assert result['queue'] == 'high'
        assert result['position'] == 1
    
    @pytest.mark.asyncio
    async def test_priority_scheduling(self, scheduler):
        """Test priority-based scheduling."""
        # Submit jobs with different priorities
        jobs = [
            {'id': f'job_{i}', 'priority': p, 'resources': {'gpus': 1}}
            for i, p in enumerate(['low', 'high', 'medium', 'high'])
        ]
        
        for job in jobs:
            await scheduler.submit_job(job)
        
        # Get next job - should be high priority
        next_job = await scheduler.get_next_job()
        assert next_job['priority'] == 'high'
    
    @pytest.mark.asyncio
    async def test_preemption(self, scheduler):
        """Test job preemption."""
        # Low priority job running
        running_job = {
            'id': 'low_priority_job',
            'priority': 'low',
            'resources': {'gpus': 8}
        }
        await scheduler.start_job(running_job)
        
        # High priority job arrives
        high_priority_job = {
            'id': 'high_priority_job',
            'priority': 'high',
            'resources': {'gpus': 4}
        }
        
        preemption_result = await scheduler.check_preemption(high_priority_job)
        
        assert preemption_result['should_preempt'] is True
        assert preemption_result['job_to_preempt'] == 'low_priority_job'
    
    @pytest.mark.asyncio
    async def test_resource_matching(self, scheduler):
        """Test resource matching for jobs."""
        available_resources = {
            'gpus': 16,
            'memory_gb': 512,
            'cpu_cores': 128
        }
        
        job = {
            'id': 'job_001',
            'resources': {'gpus': 8, 'memory_gb': 256}
        }
        
        can_run = await scheduler.check_resources(job, available_resources)
        assert can_run is True
        
        # Job that exceeds resources
        large_job = {
            'id': 'job_002',
            'resources': {'gpus': 32, 'memory_gb': 1024}
        }
        
        can_run = await scheduler.check_resources(large_job, available_resources)
        assert can_run is False
    
    @pytest.mark.asyncio
    async def test_job_completion(self, scheduler):
        """Test job completion handling."""
        job = {'id': 'job_001', 'status': 'running'}
        
        with patch.object(scheduler, 'release_resources') as mock_release:
            await scheduler.complete_job(job)
            
            mock_release.assert_called_once_with(job['resources'])
            assert job['status'] == 'completed'


class TestDistributedTraining:
    """Test distributed training functionality."""
    
    @pytest.fixture
    def distributed_trainer(self):
        """Create distributed trainer."""
        return DistributedTraining(
            world_size=8,
            backend='nccl',
            gradient_compression=True
        )
    
    @pytest.mark.asyncio
    async def test_initialization(self, distributed_trainer):
        """Test distributed training initialization."""
        with patch('torch.distributed.init_process_group') as mock_init:
            await distributed_trainer.initialize(rank=0)
            
            mock_init.assert_called_once_with(
                backend='nccl',
                world_size=8,
                rank=0
            )
    
    @pytest.mark.asyncio
    async def test_gradient_synchronization(self, distributed_trainer):
        """Test gradient synchronization."""
        model = Mock()
        gradients = [torch.randn(100, 100) for _ in range(8)]
        
        with patch('torch.distributed.all_reduce') as mock_reduce:
            await distributed_trainer.sync_gradients(gradients)
            
            assert mock_reduce.call_count == len(gradients)
    
    @pytest.mark.asyncio
    async def test_gradient_compression(self, distributed_trainer):
        """Test gradient compression."""
        gradient = torch.randn(1000, 1000)
        
        compressed = await distributed_trainer.compress_gradient(gradient)
        
        # Should be smaller than original
        assert compressed.numel() < gradient.numel()
        
        # Should be able to decompress
        decompressed = await distributed_trainer.decompress_gradient(compressed)
        assert decompressed.shape == gradient.shape
    
    @pytest.mark.asyncio
    async def test_collective_operations(self, distributed_trainer):
        """Test collective operations."""
        tensor = torch.randn(100)
        
        # Test all-reduce
        with patch('torch.distributed.all_reduce') as mock_all_reduce:
            await distributed_trainer.all_reduce(tensor)
            mock_all_reduce.assert_called_once()
        
        # Test broadcast
        with patch('torch.distributed.broadcast') as mock_broadcast:
            await distributed_trainer.broadcast(tensor, src=0)
            mock_broadcast.assert_called_once()
        
        # Test all-gather
        with patch('torch.distributed.all_gather') as mock_gather:
            await distributed_trainer.all_gather(tensor)
            mock_gather.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_checkpoint_saving(self, distributed_trainer):
        """Test distributed checkpoint saving."""
        checkpoint = {
            'model_state': torch.randn(100, 100),
            'optimizer_state': {'lr': 0.001},
            'epoch': 5
        }
        
        with patch('torch.save') as mock_save:
            await distributed_trainer.save_checkpoint(
                checkpoint,
                'checkpoint_5.pt',
                rank=0
            )
            
            # Only rank 0 should save
            mock_save.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_training_loop(self, distributed_trainer):
        """Test distributed training loop."""
        model = Mock()
        optimizer = Mock()
        dataloader = Mock()
        dataloader.__iter__ = Mock(return_value=iter([
            (torch.randn(32, 100), torch.randint(0, 10, (32,)))
            for _ in range(10)
        ]))
        
        loss_fn = Mock(return_value=torch.tensor(0.5))
        
        await distributed_trainer.train_epoch(
            model=model,
            optimizer=optimizer,
            dataloader=dataloader,
            loss_fn=loss_fn
        )
        
        # Should have performed training steps
        assert optimizer.step.call_count == 10
        assert optimizer.zero_grad.call_count == 10
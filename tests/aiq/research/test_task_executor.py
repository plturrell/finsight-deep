"""
Comprehensive test suite for Research Task Executor
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

from aiq.research.task_executor import (
    ResearchTaskExecutor,
    ResearchTask,
    ResearchResult,
    ResearchContext,
    TaskScheduler,
    ResourceManager,
    ResearchConfig
)
from aiq.memory.research import ResearchContextMemory
from aiq.retriever.neural_symbolic import NeuralSymbolicRetriever
from aiq.gpu import GPUManager


class TestResearchTaskExecutor:
    """Test suite for ResearchTaskExecutor"""
    
    @pytest.fixture
    def research_executor(self):
        """Create research executor instance"""
        config = ResearchConfig(
            max_parallel_tasks=4,
            gpu_enabled=True,
            memory_limit_gb=8,
            timeout_seconds=300
        )
        
        with patch('aiq.research.task_executor.GPUManager'):
            with patch('aiq.research.task_executor.NeuralSymbolicRetriever'):
                with patch('aiq.research.task_executor.ResearchContextMemory'):
                    return ResearchTaskExecutor(config)
    
    @pytest.fixture
    def sample_tasks(self):
        """Create sample research tasks"""
        return [
            ResearchTask(
                id="task_1",
                name="Literature Review",
                objective="Review recent AI papers",
                dependencies=[],
                priority=1,
                estimated_duration=60
            ),
            ResearchTask(
                id="task_2", 
                name="Data Analysis",
                objective="Analyze experimental results",
                dependencies=["task_1"],
                priority=2,
                estimated_duration=120
            ),
            ResearchTask(
                id="task_3",
                name="Report Generation",
                objective="Generate research report",
                dependencies=["task_1", "task_2"],
                priority=3,
                estimated_duration=30
            )
        ]
    
    @pytest.mark.asyncio
    async def test_execute_single_task(self, research_executor):
        """Test executing a single research task"""
        task = ResearchTask(
            id="single_task",
            name="Simple Research",
            objective="Test single task execution"
        )
        
        # Mock the task processing
        with patch.object(research_executor, '_process_task') as mock_process:
            mock_process.return_value = ResearchResult(
                task_id="single_task",
                success=True,
                data={"findings": ["Result 1", "Result 2"]},
                duration=5.0
            )
            
            result = await research_executor.execute_task(task)
            
            assert result.success
            assert result.task_id == "single_task"
            assert len(result.data["findings"]) == 2
    
    @pytest.mark.asyncio
    async def test_execute_parallel_tasks(self, research_executor):
        """Test executing multiple tasks in parallel"""
        # Create independent tasks
        tasks = [
            ResearchTask(id=f"task_{i}", name=f"Task {i}", objective=f"Objective {i}")
            for i in range(3)
        ]
        
        start_time = asyncio.get_event_loop().time()
        results = await research_executor.execute_parallel(tasks)
        end_time = asyncio.get_event_loop().time()
        
        assert len(results) == 3
        assert all(r.success for r in results)
        
        # Verify parallel execution (should be faster than sequential)
        total_duration = end_time - start_time
        assert total_duration < sum(r.duration for r in results)
    
    @pytest.mark.asyncio
    async def test_dependency_resolution(self, research_executor, sample_tasks):
        """Test task dependency resolution"""
        scheduler = TaskScheduler()
        execution_order = scheduler.resolve_dependencies(sample_tasks)
        
        # Verify correct order
        task_ids = [task.id for task in execution_order]
        assert task_ids.index("task_1") < task_ids.index("task_2")
        assert task_ids.index("task_2") < task_ids.index("task_3")
        assert task_ids.index("task_1") < task_ids.index("task_3")
    
    @pytest.mark.asyncio
    async def test_gpu_accelerated_task(self, research_executor):
        """Test GPU-accelerated task execution"""
        task = ResearchTask(
            id="gpu_task",
            name="GPU Computation",
            objective="Test GPU acceleration",
            gpu_required=True
        )
        
        # Mock GPU manager
        research_executor.gpu_manager.allocate_gpu.return_value = 0
        research_executor.gpu_manager.release_gpu.return_value = None
        
        # Mock GPU computation
        with patch('torch.cuda.is_available', return_value=True):
            with patch.object(research_executor, '_gpu_compute') as mock_compute:
                mock_compute.return_value = torch.tensor([1.0, 2.0, 3.0])
                
                result = await research_executor.execute_task(task)
                
                assert result.success
                assert result.gpu_used
                research_executor.gpu_manager.allocate_gpu.assert_called_once()
                research_executor.gpu_manager.release_gpu.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_memory_context_integration(self, research_executor):
        """Test integration with research context memory"""
        task = ResearchTask(
            id="memory_task",
            name="Context Research",
            objective="Test memory integration",
            context_required=True
        )
        
        # Mock memory context
        mock_context = ResearchContext(
            topic="AI Research",
            previous_findings=["Finding 1", "Finding 2"],
            sources=["Paper 1", "Paper 2"]
        )
        
        research_executor.context_memory.retrieve_context.return_value = mock_context
        
        result = await research_executor.execute_task(task)
        
        assert result.success
        assert result.context == mock_context
        research_executor.context_memory.retrieve_context.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_neural_symbolic_retrieval(self, research_executor):
        """Test neural-symbolic retrieval integration"""
        task = ResearchTask(
            id="retrieval_task",
            name="Document Research",
            objective="Find relevant papers",
            retrieval_required=True
        )
        
        # Mock retriever
        mock_documents = [
            {"title": "Paper 1", "content": "AI research content"},
            {"title": "Paper 2", "content": "ML research content"}
        ]
        
        research_executor.retriever.retrieve.return_value = mock_documents
        
        result = await research_executor.execute_task(task)
        
        assert result.success
        assert len(result.data["documents"]) == 2
        research_executor.retriever.retrieve.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_task_timeout(self, research_executor):
        """Test task timeout handling"""
        task = ResearchTask(
            id="timeout_task",
            name="Long Task",
            objective="Test timeout",
            timeout=1  # 1 second timeout
        )
        
        # Mock a long-running task
        async def slow_process():
            await asyncio.sleep(5)
            return ResearchResult(task_id="timeout_task", success=True)
        
        with patch.object(research_executor, '_process_task', side_effect=slow_process):
            with pytest.raises(asyncio.TimeoutError):
                await research_executor.execute_task(task)
    
    @pytest.mark.asyncio
    async def test_error_handling(self, research_executor):
        """Test error handling in task execution"""
        task = ResearchTask(
            id="error_task",
            name="Error Task",
            objective="Test error handling"
        )
        
        # Mock task processing with error
        with patch.object(research_executor, '_process_task') as mock_process:
            mock_process.side_effect = Exception("Processing error")
            
            result = await research_executor.execute_task(task)
            
            assert not result.success
            assert result.error == "Processing error"
            assert result.task_id == "error_task"
    
    @pytest.mark.asyncio
    async def test_resource_management(self, research_executor):
        """Test resource management during execution"""
        resource_manager = ResourceManager(
            max_memory_gb=8,
            max_cpu_percent=80,
            max_gpu_memory_gb=6
        )
        
        # Test resource allocation
        allocated = await resource_manager.allocate_resources(
            memory_gb=4,
            cpu_percent=50,
            gpu_memory_gb=3
        )
        
        assert allocated
        assert resource_manager.available_memory_gb == 4
        assert resource_manager.available_cpu_percent == 30
        assert resource_manager.available_gpu_memory_gb == 3
        
        # Test resource release
        await resource_manager.release_resources(
            memory_gb=4,
            cpu_percent=50,
            gpu_memory_gb=3
        )
        
        assert resource_manager.available_memory_gb == 8
        assert resource_manager.available_cpu_percent == 80
        assert resource_manager.available_gpu_memory_gb == 6
    
    @pytest.mark.asyncio
    async def test_batch_processing(self, research_executor):
        """Test batch processing of similar tasks"""
        # Create batch of similar tasks
        batch_tasks = [
            ResearchTask(
                id=f"batch_{i}",
                name=f"Batch Task {i}",
                objective="Process batch data",
                batch_compatible=True
            )
            for i in range(5)
        ]
        
        # Execute as batch
        results = await research_executor.execute_batch(batch_tasks)
        
        assert len(results) == 5
        assert all(r.success for r in results)
        
        # Verify batch processing efficiency
        total_duration = sum(r.duration for r in results)
        average_duration = total_duration / len(results)
        
        # Batch processing should be more efficient
        assert average_duration < 2.0  # Assuming individual tasks take ~2s
    
    @pytest.mark.asyncio
    async def test_incremental_results(self, research_executor):
        """Test incremental result reporting"""
        task = ResearchTask(
            id="incremental_task",
            name="Long Research",
            objective="Test incremental updates",
            report_progress=True
        )
        
        progress_updates = []
        
        async def progress_callback(progress: float, message: str):
            progress_updates.append((progress, message))
        
        # Execute with progress callback
        result = await research_executor.execute_task(
            task,
            progress_callback=progress_callback
        )
        
        assert result.success
        assert len(progress_updates) > 0
        assert progress_updates[-1][0] == 1.0  # 100% complete
    
    @pytest.mark.asyncio 
    async def test_caching_results(self, research_executor):
        """Test caching of research results"""
        task = ResearchTask(
            id="cached_task",
            name="Cacheable Research",
            objective="Test result caching",
            cacheable=True
        )
        
        # First execution
        result1 = await research_executor.execute_task(task)
        assert result1.success
        
        # Second execution (should use cache)
        with patch.object(research_executor, '_process_task') as mock_process:
            result2 = await research_executor.execute_task(task)
            mock_process.assert_not_called()  # Should not process again
            
        assert result2.success
        assert result2.data == result1.data
        assert result2.cached


class TestTaskScheduler:
    """Test suite for TaskScheduler"""
    
    def test_simple_dependency_resolution(self):
        """Test simple dependency resolution"""
        scheduler = TaskScheduler()
        
        tasks = [
            ResearchTask(id="A", name="A", dependencies=[]),
            ResearchTask(id="B", name="B", dependencies=["A"]),
            ResearchTask(id="C", name="C", dependencies=["B"])
        ]
        
        ordered = scheduler.resolve_dependencies(tasks)
        ids = [t.id for t in ordered]
        
        assert ids == ["A", "B", "C"]
    
    def test_complex_dependency_graph(self):
        """Test complex dependency graph resolution"""
        scheduler = TaskScheduler()
        
        tasks = [
            ResearchTask(id="A", name="A", dependencies=[]),
            ResearchTask(id="B", name="B", dependencies=["A"]),
            ResearchTask(id="C", name="C", dependencies=["A"]),
            ResearchTask(id="D", name="D", dependencies=["B", "C"]),
            ResearchTask(id="E", name="E", dependencies=["D"])
        ]
        
        ordered = scheduler.resolve_dependencies(tasks)
        ids = [t.id for t in ordered]
        
        # A must come first
        assert ids[0] == "A"
        
        # B and C must come after A but before D
        assert ids.index("B") > ids.index("A")
        assert ids.index("C") > ids.index("A")
        assert ids.index("D") > ids.index("B")
        assert ids.index("D") > ids.index("C")
        
        # E must come last
        assert ids[-1] == "E"
    
    def test_circular_dependency_detection(self):
        """Test circular dependency detection"""
        scheduler = TaskScheduler()
        
        tasks = [
            ResearchTask(id="A", name="A", dependencies=["C"]),
            ResearchTask(id="B", name="B", dependencies=["A"]),
            ResearchTask(id="C", name="C", dependencies=["B"])
        ]
        
        with pytest.raises(ValueError, match="Circular dependency"):
            scheduler.resolve_dependencies(tasks)
    
    def test_priority_scheduling(self):
        """Test priority-based scheduling"""
        scheduler = TaskScheduler()
        
        tasks = [
            ResearchTask(id="A", name="A", priority=3),
            ResearchTask(id="B", name="B", priority=1),
            ResearchTask(id="C", name="C", priority=2)
        ]
        
        scheduled = scheduler.schedule_by_priority(tasks)
        priorities = [t.priority for t in scheduled]
        
        assert priorities == [1, 2, 3]  # Ascending priority order


class TestResearchContext:
    """Test suite for ResearchContext"""
    
    def test_context_creation(self):
        """Test research context creation"""
        context = ResearchContext(
            topic="Machine Learning",
            domain="Computer Science",
            previous_findings=["Finding 1", "Finding 2"],
            sources=["Paper 1", "Website 1"],
            constraints={"time_limit": 3600, "quality_threshold": 0.8}
        )
        
        assert context.topic == "Machine Learning"
        assert len(context.previous_findings) == 2
        assert len(context.sources) == 2
        assert context.constraints["quality_threshold"] == 0.8
    
    def test_context_merging(self):
        """Test merging multiple research contexts"""
        context1 = ResearchContext(
            topic="AI",
            previous_findings=["Finding 1"],
            sources=["Source 1"]
        )
        
        context2 = ResearchContext(
            topic="AI",
            previous_findings=["Finding 2"],
            sources=["Source 2"]
        )
        
        merged = context1.merge(context2)
        
        assert merged.topic == "AI"
        assert len(merged.previous_findings) == 2
        assert len(merged.sources) == 2
        assert "Finding 1" in merged.previous_findings
        assert "Finding 2" in merged.previous_findings
    
    def test_context_serialization(self):
        """Test context serialization and deserialization"""
        context = ResearchContext(
            topic="Data Science",
            metadata={"created_at": datetime.now().isoformat()}
        )
        
        # Serialize
        serialized = context.to_dict()
        assert isinstance(serialized, dict)
        assert serialized["topic"] == "Data Science"
        
        # Deserialize
        restored = ResearchContext.from_dict(serialized)
        assert restored.topic == context.topic
        assert restored.metadata == context.metadata


class TestResearchResult:
    """Test suite for ResearchResult"""
    
    def test_result_creation(self):
        """Test research result creation"""
        result = ResearchResult(
            task_id="test_task",
            success=True,
            data={"findings": ["Result 1", "Result 2"]},
            duration=10.5,
            metadata={"method": "neural_search"}
        )
        
        assert result.task_id == "test_task"
        assert result.success
        assert len(result.data["findings"]) == 2
        assert result.duration == 10.5
        assert result.metadata["method"] == "neural_search"
    
    def test_result_aggregation(self):
        """Test aggregating multiple research results"""
        results = [
            ResearchResult(
                task_id=f"task_{i}",
                success=True,
                data={"score": i * 10},
                duration=i
            )
            for i in range(1, 4)
        ]
        
        aggregated = ResearchResult.aggregate(results)
        
        assert aggregated.success
        assert aggregated.data["total_tasks"] == 3
        assert aggregated.data["average_score"] == 20.0
        assert aggregated.duration == 6  # Sum of durations
    
    def test_result_validation(self):
        """Test result validation"""
        # Valid result
        valid_result = ResearchResult(
            task_id="valid",
            success=True,
            data={"key": "value"}
        )
        assert valid_result.is_valid()
        
        # Invalid result (missing data)
        invalid_result = ResearchResult(
            task_id="invalid",
            success=True,
            data=None
        )
        assert not invalid_result.is_valid()


class TestGPUAcceleration:
    """Test GPU acceleration features"""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    @pytest.mark.asyncio
    async def test_gpu_computation(self):
        """Test GPU-accelerated computation"""
        executor = ResearchTaskExecutor(
            ResearchConfig(gpu_enabled=True)
        )
        
        # Create tensor computation task
        task = ResearchTask(
            id="gpu_compute",
            name="Tensor Operations",
            objective="Matrix multiplication",
            gpu_required=True
        )
        
        # Define GPU computation
        async def gpu_compute():
            device = torch.device("cuda:0")
            a = torch.randn(1000, 1000, device=device)
            b = torch.randn(1000, 1000, device=device)
            c = torch.matmul(a, b)
            return c.cpu().numpy()
        
        with patch.object(executor, '_gpu_compute', side_effect=gpu_compute):
            result = await executor.execute_task(task)
            
            assert result.success
            assert result.gpu_used
            assert isinstance(result.data["output"], np.ndarray)
    
    @pytest.mark.asyncio
    async def test_multi_gpu_distribution(self):
        """Test multi-GPU task distribution"""
        config = ResearchConfig(
            gpu_enabled=True,
            num_gpus=2
        )
        
        executor = ResearchTaskExecutor(config)
        
        # Create GPU-intensive tasks
        tasks = [
            ResearchTask(
                id=f"gpu_task_{i}",
                name=f"GPU Task {i}",
                objective="GPU computation",
                gpu_required=True
            )
            for i in range(4)
        ]
        
        # Execute tasks (should distribute across GPUs)
        results = await executor.execute_parallel(tasks)
        
        assert len(results) == 4
        assert all(r.gpu_used for r in results)
        
        # Check GPU distribution
        gpu_usage = [r.metadata.get("gpu_id", 0) for r in results]
        assert len(set(gpu_usage)) > 1  # Used multiple GPUs
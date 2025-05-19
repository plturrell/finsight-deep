# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

from aiq.distributed.task_scheduler import TaskScheduler, Task, TaskStatus
from aiq.distributed.node_manager import NodeManager, NodeInfo


class TestTaskScheduler:
    """Test task scheduler functionality"""
    
    @pytest.fixture
    def mock_node_manager(self):
        """Create a mock node manager"""
        manager = Mock(spec=NodeManager)
        
        # Mock nodes
        node1 = NodeInfo(
            node_id="node1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=2,
            gpus=[],
            status="online"
        )
        
        manager.get_node_info.return_value = node1
        manager.assign_task_to_node.return_value = "node1"
        
        return manager
    
    @pytest.fixture
    def task_scheduler(self, mock_node_manager):
        """Create a task scheduler instance"""
        return TaskScheduler(mock_node_manager)
    
    @pytest.mark.asyncio
    async def test_submit_task(self, task_scheduler):
        """Test task submission"""
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"},
            metadata={"priority": "high"}
        )
        
        assert task_id in task_scheduler.tasks
        task = task_scheduler.tasks[task_id]
        
        assert task.function_name == "test_function"
        assert task.inputs == {"param1": "value1"}
        assert task.metadata == {"priority": "high"}
        assert task.status == TaskStatus.PENDING
    
    @pytest.mark.asyncio
    async def test_get_task_status(self, task_scheduler):
        """Test getting task status"""
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"}
        )
        
        task = await task_scheduler.get_task_status(task_id)
        
        assert task is not None
        assert task.task_id == task_id
        assert task.status == TaskStatus.PENDING
        
        # Non-existent task
        task = await task_scheduler.get_task_status("invalid-id")
        assert task is None
    
    @pytest.mark.asyncio
    async def test_wait_for_task_completion(self, task_scheduler):
        """Test waiting for task completion"""
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"}
        )
        
        # Simulate task completion in background
        async def complete_task():
            await asyncio.sleep(0.1)
            task = task_scheduler.tasks[task_id]
            task.status = TaskStatus.COMPLETED
            task.result = {"output": "result"}
        
        complete_task_future = asyncio.create_task(complete_task())
        
        # Wait for task
        completed_task = await task_scheduler.wait_for_task(task_id)
        
        assert completed_task.status == TaskStatus.COMPLETED
        assert completed_task.result == {"output": "result"}
        
        await complete_task_future
    
    @pytest.mark.asyncio
    async def test_wait_for_task_timeout(self, task_scheduler):
        """Test task wait timeout"""
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"}
        )
        
        # Try to wait with short timeout
        with pytest.raises(TimeoutError):
            await task_scheduler.wait_for_task(task_id, timeout=0.1)
    
    @pytest.mark.asyncio
    async def test_cancel_task(self, task_scheduler):
        """Test task cancellation"""
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"}
        )
        
        # Cancel the task
        success = await task_scheduler.cancel_task(task_id)
        assert success is True
        
        task = task_scheduler.tasks[task_id]
        assert task.status == TaskStatus.CANCELLED
        
        # Try to cancel non-existent task
        success = await task_scheduler.cancel_task("invalid-id")
        assert success is False
    
    def test_get_queue_status(self, task_scheduler):
        """Test getting queue status"""
        # Add some tasks
        task1 = Task(
            task_id="task1",
            function_name="func1",
            inputs={},
            status=TaskStatus.PENDING
        )
        task2 = Task(
            task_id="task2",
            function_name="func2",
            inputs={},
            status=TaskStatus.RUNNING
        )
        task3 = Task(
            task_id="task3",
            function_name="func3",
            inputs={},
            status=TaskStatus.COMPLETED
        )
        
        task_scheduler.tasks = {
            "task1": task1,
            "task2": task2,
            "task3": task3
        }
        
        status = task_scheduler.get_queue_status()
        
        assert status["total_tasks"] == 3
        assert status["status_counts"][TaskStatus.PENDING.value] == 1
        assert status["status_counts"][TaskStatus.RUNNING.value] == 1
        assert status["status_counts"][TaskStatus.COMPLETED.value] == 1
    
    @pytest.mark.asyncio
    async def test_distribute_batch(self, task_scheduler):
        """Test batch task distribution"""
        inputs_list = [
            {"data": "item1"},
            {"data": "item2"},
            {"data": "item3"}
        ]
        
        # Submit batch without waiting
        tasks = await task_scheduler.distribute_batch(
            function_name="batch_function",
            inputs_list=inputs_list,
            wait=False
        )
        
        assert len(tasks) == 3
        assert all(task.status == TaskStatus.PENDING for task in tasks)
        
        # Simulate completion
        for task in tasks:
            task.status = TaskStatus.COMPLETED
            task.result = {"processed": task.inputs["data"]}
        
        # Submit batch with waiting
        with patch.object(task_scheduler, 'wait_for_task') as mock_wait:
            mock_wait.side_effect = tasks
            
            completed_tasks = await task_scheduler.distribute_batch(
                function_name="batch_function",
                inputs_list=inputs_list,
                wait=True
            )
            
            assert len(completed_tasks) == 3
    
    @pytest.mark.asyncio
    async def test_scheduler_loop(self, task_scheduler, mock_node_manager):
        """Test the main scheduler loop"""
        # Start scheduler
        task_scheduler._running = True
        scheduler_task = asyncio.create_task(task_scheduler._schedule_tasks())
        
        # Submit a task
        task_id = await task_scheduler.submit_task(
            function_name="test_function",
            inputs={"param1": "value1"}
        )
        
        # Wait for scheduler to process
        await asyncio.sleep(0.1)
        
        # Check task was assigned
        task = task_scheduler.tasks[task_id]
        assert task.status == TaskStatus.ASSIGNED
        assert task.node_id == "node1"
        
        # Stop scheduler
        task_scheduler._running = False
        scheduler_task.cancel()
        try:
            await scheduler_task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self, task_scheduler, mock_node_manager):
        """Test successful task execution"""
        # Create a task
        task = Task(
            task_id="task123",
            function_name="test_function",
            inputs={"param": "value"},
            node_id="node1"
        )
        
        # Mock gRPC client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.success = True
        mock_response.serialized_output = '{"result": "success"}'
        mock_client.ExecuteTask.return_value = mock_response
        
        with patch.object(task_scheduler, '_get_node_client', return_value=mock_client):
            await task_scheduler._execute_task(task)
        
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": "success"}
        assert task.completed_at is not None
    
    @pytest.mark.asyncio
    async def test_execute_task_failure(self, task_scheduler, mock_node_manager):
        """Test failed task execution"""
        # Create a task
        task = Task(
            task_id="task123",
            function_name="test_function",
            inputs={"param": "value"},
            node_id="node1"
        )
        
        # Mock gRPC client
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.success = False
        mock_response.error_message = "Test error"
        mock_client.ExecuteTask.return_value = mock_response
        
        with patch.object(task_scheduler, '_get_node_client', return_value=mock_client):
            await task_scheduler._execute_task(task)
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Test error"
        assert task.completed_at is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
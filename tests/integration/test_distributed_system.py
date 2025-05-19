# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for the distributed system
Tests the full stack: node manager, workers, and scheduler
"""

import pytest
import asyncio
from typing import Dict, Any
import time

from aiq.distributed.node_manager import NodeManager
from aiq.distributed.worker_node import WorkerNode
from aiq.distributed.task_scheduler import TaskScheduler
from aiq.builder.distributed_workflow_builder import DistributedWorkflowBuilder
from aiq.builder.function import Function
from aiq.data_models.function import FunctionConfig
from aiq.data_models.workflow import WorkflowConfig


class DummyFunction(Function):
    """Simple function for testing"""
    
    def run(self, session, inputs: Dict[str, Any]) -> Any:
        """Process inputs and return result"""
        x = inputs.get("x", 0)
        y = inputs.get("y", 0)
        operation = inputs.get("operation", "add")
        
        if operation == "add":
            return {"result": x + y}
        elif operation == "multiply":
            return {"result": x * y}
        else:
            return {"error": f"Unknown operation: {operation}"}


@pytest.mark.integration
@pytest.mark.asyncio
class TestDistributedSystem:
    """Integration tests for distributed system"""
    
    async def test_single_node_cluster(self):
        """Test a simple single-node cluster"""
        manager_port = 50061
        worker_port = 50062
        
        # Start manager
        manager = NodeManager(port=manager_port)
        server_task = asyncio.create_task(manager.run())
        
        # Wait for manager to start
        await asyncio.sleep(1)
        
        # Start worker
        worker = WorkerNode(
            manager_host="localhost",
            manager_port=manager_port,
            worker_port=worker_port
        )
        
        # Register function
        config = FunctionConfig(
            name="dummy_function",
            inputs=["x", "y", "operation"],
            outputs=["result"]
        )
        dummy_func = DummyFunction(config)
        worker.register_function("dummy_function", dummy_func)
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        # Wait for worker to register
        await asyncio.sleep(2)
        
        # Create scheduler
        scheduler = TaskScheduler(manager)
        scheduler_task = asyncio.create_task(scheduler.start())
        
        # Submit tasks
        task_ids = []
        for i in range(5):
            task_id = await scheduler.submit_task(
                function_name="dummy_function",
                inputs={
                    "x": i,
                    "y": i + 1,
                    "operation": "add"
                }
            )
            task_ids.append(task_id)
        
        # Wait for tasks to complete
        results = []
        for task_id in task_ids:
            task = await scheduler.wait_for_task(task_id, timeout=10)
            results.append(task.result)
        
        # Verify results
        for i, result in enumerate(results):
            expected = i + (i + 1)  # x + y
            assert result["result"] == expected
        
        # Check cluster status
        status = manager.get_cluster_status()
        assert status["summary"]["total_nodes"] >= 1
        assert status["summary"]["online_nodes"] >= 1
        
        # Cleanup
        await scheduler.stop()
        await worker.stop()
        manager.stop()
        
        # Cancel tasks
        server_task.cancel()
        worker_task.cancel()
        scheduler_task.cancel()
        
        # Wait for cleanup
        try:
            await server_task
            await worker_task
            await scheduler_task
        except asyncio.CancelledError:
            pass
    
    async def test_multi_node_cluster(self):
        """Test a multi-node cluster with multiple workers"""
        manager_port = 50063
        worker_ports = [50064, 50065, 50066]
        
        # Start manager
        manager = NodeManager(port=manager_port)
        server_task = asyncio.create_task(manager.run())
        
        # Wait for manager to start
        await asyncio.sleep(1)
        
        # Start multiple workers
        workers = []
        worker_tasks = []
        
        for i, port in enumerate(worker_ports):
            worker = WorkerNode(
                node_id=f"worker-{i}",
                manager_host="localhost",
                manager_port=manager_port,
                worker_port=port
            )
            
            # Register function
            config = FunctionConfig(
                name=f"worker_{i}_function",
                inputs=["data"],
                outputs=["result"]
            )
            
            # Create function that identifies which worker processed it
            class WorkerFunction(Function):
                def __init__(self, config, worker_id):
                    super().__init__(config)
                    self.worker_id = worker_id
                
                def run(self, session, inputs: Dict[str, Any]) -> Any:
                    return {
                        "worker_id": self.worker_id,
                        "data": inputs["data"],
                        "processed": True
                    }
            
            worker_func = WorkerFunction(config, i)
            worker.register_function(f"worker_{i}_function", worker_func)
            workers.append(worker)
            
            # Start worker
            task = asyncio.create_task(worker.start())
            worker_tasks.append(task)
        
        # Wait for workers to register
        await asyncio.sleep(3)
        
        # Verify all workers registered
        assert len(manager.nodes) >= len(workers)
        
        # Create scheduler
        scheduler = TaskScheduler(manager)
        scheduler_task = asyncio.create_task(scheduler.start())
        
        # Submit tasks to different workers
        task_ids = []
        for i in range(10):
            worker_id = i % len(workers)
            task_id = await scheduler.submit_task(
                function_name=f"worker_{worker_id}_function",
                inputs={"data": f"task-{i}"}
            )
            task_ids.append((task_id, worker_id))
        
        # Wait for tasks to complete
        for task_id, expected_worker in task_ids:
            task = await scheduler.wait_for_task(task_id, timeout=10)
            assert task.result["worker_id"] == expected_worker
            assert task.result["processed"] is True
        
        # Check resource distribution
        resources = manager.get_total_resources()
        assert resources["total_nodes"] == len(workers) + 1  # workers + manager
        
        # Cleanup
        await scheduler.stop()
        for worker in workers:
            await worker.stop()
        manager.stop()
        
        # Cancel all tasks
        server_task.cancel()
        scheduler_task.cancel()
        for task in worker_tasks:
            task.cancel()
        
        # Wait for cleanup
        try:
            await server_task
            await scheduler_task
            for task in worker_tasks:
                await task
        except asyncio.CancelledError:
            pass
    
    async def test_distributed_workflow_integration(self):
        """Test the distributed workflow builder integration"""
        manager_port = 50067
        worker_port = 50068
        
        # Start manager
        manager = NodeManager(port=manager_port)
        server_task = asyncio.create_task(manager.run())
        
        # Wait for manager to start
        await asyncio.sleep(1)
        
        # Start worker
        worker = WorkerNode(
            manager_host="localhost",
            manager_port=manager_port,
            worker_port=worker_port
        )
        
        # Register function
        config = FunctionConfig(
            name="test_function",
            inputs=["value"],
            outputs=["result"]
        )
        
        class TestFunction(Function):
            def run(self, session, inputs: Dict[str, Any]) -> Any:
                value = inputs["value"]
                return {"result": value * 2}
        
        test_func = TestFunction(config)
        worker.register_function("test_function", test_func)
        
        # Start worker
        worker_task = asyncio.create_task(worker.start())
        
        # Wait for worker to register
        await asyncio.sleep(2)
        
        # Create distributed workflow
        builder = DistributedWorkflowBuilder(enable_multi_node=True)
        
        # Manually set the node manager and scheduler
        builder.node_manager = manager
        builder.task_scheduler = TaskScheduler(manager)
        
        # Start scheduler
        scheduler_task = asyncio.create_task(builder.task_scheduler.start())
        
        # Create workflow config
        workflow_config = WorkflowConfig(
            name="test_distributed_workflow",
            description="Integration test workflow"
        )
        
        # Build workflow
        workflow = builder.build(workflow_config)
        
        # Submit tasks through workflow
        task_futures = []
        
        for i in range(5):
            task_id = await workflow.task_scheduler.submit_task(
                function_name="test_function",
                inputs={"value": i}
            )
            task_futures.append(task_id)
        
        # Wait for results
        results = []
        for task_id in task_futures:
            task = await workflow.task_scheduler.wait_for_task(task_id)
            results.append(task.result["result"])
        
        # Verify results
        for i, result in enumerate(results):
            assert result == i * 2
        
        # Cleanup
        await builder.task_scheduler.stop()
        await worker.stop()
        manager.stop()
        
        # Cancel tasks
        server_task.cancel()
        worker_task.cancel()
        scheduler_task.cancel()
        
        # Wait for cleanup
        try:
            await server_task
            await worker_task
            await scheduler_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])
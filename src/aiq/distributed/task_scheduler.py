# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Task Scheduler for distributed AIQToolkit
Manages task distribution across nodes
"""

import asyncio
import grpc
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import logging
import uuid
from enum import Enum

from aiq.distributed.node_manager import NodeManager, NodeInfo
from aiq.distributed.grpc import node_pb2, node_pb2_grpc

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Task:
    """Represents a distributed task"""
    task_id: str
    function_name: str
    inputs: Dict[str, Any]
    status: TaskStatus = TaskStatus.PENDING
    node_id: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary"""
        return {
            "task_id": self.task_id,
            "function_name": self.function_name,
            "status": self.status.value,
            "node_id": self.node_id,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "metadata": self.metadata
        }


class TaskScheduler:
    """
    Manages task scheduling and execution across distributed nodes
    """
    
    def __init__(self, node_manager: NodeManager):
        self.node_manager = node_manager
        self.tasks: Dict[str, Task] = {}
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self._scheduler_task = None
        self._running = False
        
        # Track node clients for task execution
        self.node_clients: Dict[str, node_pb2_grpc.TaskExecutorStub] = {}
    
    async def start(self):
        """Start the task scheduler"""
        self._running = True
        self._scheduler_task = asyncio.create_task(self._schedule_tasks())
        logger.info("Task scheduler started")
    
    async def stop(self):
        """Stop the task scheduler"""
        self._running = False
        if self._scheduler_task:
            self._scheduler_task.cancel()
        logger.info("Task scheduler stopped")
    
    async def submit_task(self, 
                         function_name: str,
                         inputs: Dict[str, Any],
                         metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Submit a task for distributed execution
        
        Args:
            function_name: Name of function to execute
            inputs: Function inputs
            metadata: Optional metadata
            
        Returns:
            Task ID
        """
        task_id = str(uuid.uuid4())
        
        task = Task(
            task_id=task_id,
            function_name=function_name,
            inputs=inputs,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task)
        
        logger.info(f"Submitted task {task_id} for function {function_name}")
        return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Task]:
        """Get the status of a task"""
        return self.tasks.get(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Task:
        """
        Wait for a task to complete
        
        Args:
            task_id: Task ID to wait for
            timeout: Optional timeout in seconds
            
        Returns:
            Completed task
        """
        start_time = datetime.now()
        
        while True:
            task = self.tasks.get(task_id)
            
            if not task:
                raise ValueError(f"Unknown task: {task_id}")
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                return task
            
            # Check timeout
            if timeout:
                elapsed = (datetime.now() - start_time).seconds
                if elapsed > timeout:
                    raise TimeoutError(f"Task {task_id} timed out after {timeout} seconds")
            
            await asyncio.sleep(0.1)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a pending or running task"""
        task = self.tasks.get(task_id)
        
        if not task:
            return False
            
        if task.status in [TaskStatus.PENDING, TaskStatus.ASSIGNED]:
            task.status = TaskStatus.CANCELLED
            logger.info(f"Cancelled task {task_id}")
            return True
            
        return False
    
    async def _schedule_tasks(self):
        """Main scheduling loop"""
        while self._running:
            try:
                # Get next task from queue
                task = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Find available node
                resource_requirements = task.metadata.get("resources", {})
                node_id = self.node_manager.assign_task_to_node(
                    task.task_id,
                    resource_requirements
                )
                
                if node_id:
                    # Assign task to node
                    task.node_id = node_id
                    task.status = TaskStatus.ASSIGNED
                    
                    # Execute task on node
                    asyncio.create_task(self._execute_task(task))
                else:
                    # No available nodes, requeue task
                    await self.task_queue.put(task)
                    await asyncio.sleep(1)  # Wait before retrying
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in scheduler loop: {e}")
    
    async def _execute_task(self, task: Task):
        """Execute a task on its assigned node"""
        try:
            # Get node info
            node_info = self.node_manager.get_node_info(task.node_id)
            if not node_info:
                raise RuntimeError(f"Node {task.node_id} not found")
            
            # Get or create client for node
            client = await self._get_node_client(node_info)
            
            # Mark task as running
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now()
            
            # Prepare execution request
            request = node_pb2.ExecuteTaskRequest(
                task_id=task.task_id,
                function_name=task.function_name,
                serialized_inputs=json.dumps(task.inputs),
                metadata=task.metadata
            )
            
            # Execute task
            logger.info(f"Executing task {task.task_id} on node {task.node_id}")
            response = await client.ExecuteTask(request)
            
            # Process response
            if response.success:
                task.result = json.loads(response.serialized_output)
                task.status = TaskStatus.COMPLETED
                logger.info(f"Task {task.task_id} completed successfully")
            else:
                task.error = response.error_message
                task.status = TaskStatus.FAILED
                logger.error(f"Task {task.task_id} failed: {response.error_message}")
            
            task.completed_at = datetime.now()
            
        except Exception as e:
            logger.error(f"Error executing task {task.task_id}: {e}")
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now()
        finally:
            # Release task from node
            self.node_manager.release_task(task.task_id, task.node_id)
    
    async def _get_node_client(self, node_info: NodeInfo) -> node_pb2_grpc.TaskExecutorStub:
        """Get or create gRPC client for a node"""
        if node_info.node_id not in self.node_clients:
            channel = grpc.aio.insecure_channel(
                f'{node_info.ip_address}:{node_info.port}'
            )
            self.node_clients[node_info.node_id] = node_pb2_grpc.TaskExecutorStub(channel)
        
        return self.node_clients[node_info.node_id]
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = sum(
                1 for task in self.tasks.values() 
                if task.status == status
            )
        
        return {
            "total_tasks": len(self.tasks),
            "queue_size": self.task_queue.qsize(),
            "status_counts": status_counts,
            "oldest_pending": self._get_oldest_pending_task()
        }
    
    def _get_oldest_pending_task(self) -> Optional[Dict[str, Any]]:
        """Get the oldest pending task"""
        pending_tasks = [
            task for task in self.tasks.values()
            if task.status == TaskStatus.PENDING
        ]
        
        if not pending_tasks:
            return None
            
        oldest = min(pending_tasks, key=lambda t: t.created_at)
        return oldest.to_dict()
    
    def get_task_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get task execution history"""
        # Sort tasks by creation time
        sorted_tasks = sorted(
            self.tasks.values(),
            key=lambda t: t.created_at,
            reverse=True
        )
        
        # Return limited history
        return [task.to_dict() for task in sorted_tasks[:limit]]
    
    async def distribute_batch(self,
                             function_name: str,
                             inputs_list: List[Dict[str, Any]],
                             wait: bool = True) -> List[Task]:
        """
        Distribute a batch of tasks across nodes
        
        Args:
            function_name: Function to execute
            inputs_list: List of inputs for each task
            wait: Whether to wait for all tasks to complete
            
        Returns:
            List of completed tasks
        """
        task_ids = []
        
        # Submit all tasks
        for inputs in inputs_list:
            task_id = await self.submit_task(function_name, inputs)
            task_ids.append(task_id)
        
        if not wait:
            return [self.tasks[tid] for tid in task_ids]
        
        # Wait for all tasks to complete
        completed_tasks = []
        for task_id in task_ids:
            task = await self.wait_for_task(task_id)
            completed_tasks.append(task)
        
        return completed_tasks
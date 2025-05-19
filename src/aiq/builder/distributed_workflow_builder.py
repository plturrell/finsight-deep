# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed Workflow Builder for AIQToolkit
Extends the base workflow builder with multi-GPU and multi-node capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any, Tuple
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import torch

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.builder.workflow import Workflow
from aiq.data_models.workflow import WorkflowConfig
from aiq.gpu.multi_gpu_manager import MultiGPUManager
from aiq.runtime.session import AIQSessionManager as Session
from aiq.data_models.intermediate_step import IntermediateStep
from aiq.distributed.node_manager import NodeManager
from aiq.distributed.task_scheduler import TaskScheduler

logger = logging.getLogger(__name__)


@dataclass
class DistributedTask:
    """Represents a task that can be distributed"""
    task_id: str
    function_name: str
    inputs: Dict[str, Any]
    device_id: Optional[int] = None
    priority: int = 0


class DistributedWorkflowBuilder(WorkflowBuilder):
    """
    Workflow builder with distributed processing capabilities
    """
    
    def __init__(self, enable_multi_node: bool = False):
        super().__init__()
        self.gpu_manager = MultiGPUManager()
        self.executor = ThreadPoolExecutor(
            max_workers=self.gpu_manager.device_count or 1
        )
        
        # Multi-node support
        self.enable_multi_node = enable_multi_node
        self.node_manager = None
        self.task_scheduler = None
        
        if enable_multi_node:
            self.node_manager = NodeManager()
            self.task_scheduler = TaskScheduler(self.node_manager)
        
    def build(self, config: WorkflowConfig) -> Workflow:
        """
        Build a distributed workflow from configuration
        
        Args:
            config: Workflow configuration
            
        Returns:
            Distributed workflow instance
        """
        # Build base workflow
        workflow = super().build(config)
        
        # Enhance with distributed capabilities
        return DistributedWorkflow(
            workflow=workflow,
            gpu_manager=self.gpu_manager,
            executor=self.executor,
            node_manager=self.node_manager,
            task_scheduler=self.task_scheduler
        )


class DistributedWorkflow(Workflow):
    """
    Workflow with distributed execution capabilities
    """
    
    def __init__(self, workflow: Workflow, gpu_manager: MultiGPUManager, 
                 executor: ThreadPoolExecutor, node_manager: Optional[NodeManager] = None,
                 task_scheduler: Optional[TaskScheduler] = None):
        # Copy attributes from base workflow
        self.__dict__.update(workflow.__dict__)
        
        self.gpu_manager = gpu_manager
        self.executor = executor
        self.node_manager = node_manager
        self.task_scheduler = task_scheduler
        self.device_assignments: Dict[str, int] = {}
        self.enable_multi_node = node_manager is not None
        
    async def run(self, session: Session, inputs: Dict[str, Any]) -> Any:
        """
        Run the workflow with distributed processing
        
        Args:
            session: Runtime session
            inputs: Input values
            
        Returns:
            Workflow results
        """
        # Analyze workflow for parallelization opportunities
        parallel_tasks = self._identify_parallel_tasks(inputs)
        
        if not parallel_tasks:
            # No parallelization possible, run normally
            return await super().run(session, inputs)
        
        # Assign tasks to GPUs
        self._assign_devices(parallel_tasks)
        
        # Execute parallel tasks
        results = await self._execute_parallel(session, parallel_tasks)
        
        # Continue with sequential parts
        return await self._complete_workflow(session, results)
    
    def _identify_parallel_tasks(self, inputs: Dict[str, Any]) -> List[DistributedTask]:
        """
        Identify tasks that can be executed in parallel
        
        This is a simplified implementation. In practice, you would
        analyze the workflow DAG to find independent branches.
        """
        parallel_tasks = []
        
        # For now, identify functions that can run in parallel
        # based on their inputs not depending on each other
        for func_name, func in self.functions.items():
            # Check if function inputs are available
            func_inputs = {}
            can_run = True
            
            for input_name, input_value in func.inputs.items():
                if input_name in inputs:
                    func_inputs[input_name] = inputs[input_name]
                else:
                    can_run = False
                    break
                    
            if can_run:
                task = DistributedTask(
                    task_id=f"{func_name}_0",
                    function_name=func_name,
                    inputs=func_inputs
                )
                parallel_tasks.append(task)
        
        return parallel_tasks
    
    def _assign_devices(self, tasks: List[DistributedTask]):
        """
        Assign tasks to available GPU devices
        
        Simple round-robin assignment. Could be enhanced with
        load balancing based on task complexity and GPU utilization.
        """
        if self.gpu_manager.device_count == 0:
            return
            
        for i, task in enumerate(tasks):
            device_id = i % self.gpu_manager.device_count
            task.device_id = device_id
            self.device_assignments[task.task_id] = device_id
            
        logger.info(f"Assigned {len(tasks)} tasks across "
                   f"{self.gpu_manager.device_count} GPUs")
    
    async def _execute_parallel(self, session: Session, 
                               tasks: List[DistributedTask]) -> Dict[str, Any]:
        """
        Execute tasks in parallel across multiple GPUs or nodes
        """
        results = {}
        
        # Use multi-node execution if available
        if self.enable_multi_node and self.task_scheduler:
            # Submit tasks to distributed nodes
            task_futures = []
            
            for task in tasks:
                # Submit to scheduler
                task_id = await self.task_scheduler.submit_task(
                    function_name=task.function_name,
                    inputs=task.inputs,
                    metadata={
                        "task_id": task.task_id,
                        "priority": task.priority
                    }
                )
                task_futures.append((task.task_id, task_id))
            
            # Wait for all tasks to complete
            for orig_task_id, scheduler_task_id in task_futures:
                completed_task = await self.task_scheduler.wait_for_task(scheduler_task_id)
                results[orig_task_id] = completed_task.result
                
        else:
            # Use local multi-GPU execution
            # Group tasks by device
            device_tasks: Dict[int, List[DistributedTask]] = {}
            for task in tasks:
                device_id = task.device_id or 0
                if device_id not in device_tasks:
                    device_tasks[device_id] = []
                device_tasks[device_id].append(task)
            
            # Execute tasks per device
            futures = []
            
            for device_id, device_task_list in device_tasks.items():
                future = self.executor.submit(
                    self._execute_on_device,
                    session,
                    device_id,
                    device_task_list
                )
                futures.append(future)
            
            # Wait for all tasks to complete
            for future in futures:
                device_results = future.result()
                results.update(device_results)
            
        return results
    
    def _execute_on_device(self, session: Session, device_id: int,
                          tasks: List[DistributedTask]) -> Dict[str, Any]:
        """
        Execute tasks on a specific GPU device
        """
        # Set CUDA device for this thread
        if self.gpu_manager.device_count > 0:
            torch.cuda.set_device(device_id)
            
        results = {}
        
        for task in tasks:
            logger.info(f"Executing task {task.task_id} on GPU {device_id}")
            
            # Get the function
            func = self.functions[task.function_name]
            
            # Execute function
            try:
                result = func.run(session, task.inputs)
                results[task.task_id] = result
                
                # Record intermediate step
                step = IntermediateStep(
                    function_name=task.function_name,
                    inputs=task.inputs,
                    output=result,
                    metadata={
                        'device_id': device_id,
                        'gpu_name': self.gpu_manager.get_gpu_info(device_id)[0].name
                    }
                )
                session.add_intermediate_step(step)
                
            except Exception as e:
                logger.error(f"Error executing task {task.task_id}: {e}")
                results[task.task_id] = None
                
        return results
    
    async def _complete_workflow(self, session: Session, 
                               parallel_results: Dict[str, Any]) -> Any:
        """
        Complete the workflow after parallel execution
        
        This would handle any remaining sequential steps that depend
        on the parallel results.
        """
        # For now, just return the parallel results
        # In a full implementation, this would continue with the workflow DAG
        return parallel_results


class DistributedSession(Session):
    """
    Enhanced session for distributed execution
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gpu_memory_usage: Dict[int, Dict[str, float]] = {}
        self.task_timings: Dict[str, float] = {}
        
    def record_gpu_usage(self, gpu_manager: MultiGPUManager):
        """Record GPU memory usage across all devices"""
        self.gpu_memory_usage = gpu_manager.get_memory_summary()
        
    def add_task_timing(self, task_id: str, duration: float):
        """Record task execution time"""
        self.task_timings[task_id] = duration
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary of distributed execution"""
        return {
            'gpu_memory_usage': self.gpu_memory_usage,
            'task_timings': self.task_timings,
            'total_tasks': len(self.task_timings),
            'avg_task_time': sum(self.task_timings.values()) / len(self.task_timings)
                            if self.task_timings else 0
        }


# Convenience function for creating distributed workflows
def create_distributed_workflow(config: WorkflowConfig) -> DistributedWorkflow:
    """
    Create a distributed workflow from configuration
    
    Args:
        config: Workflow configuration
        
    Returns:
        Distributed workflow instance
    """
    builder = DistributedWorkflowBuilder()
    return builder.build(config)
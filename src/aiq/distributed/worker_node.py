# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Worker Node for distributed AIQToolkit
Executes tasks assigned by the node manager
"""

import asyncio
import grpc
import json
import torch
from typing import Dict, Any, Optional
import logging
import uuid
import socket
from datetime import datetime
from concurrent import futures

from aiq.distributed.grpc import node_pb2, node_pb2_grpc
from aiq.distributed.node_manager import NodeInfo
from aiq.gpu.multi_gpu_manager import MultiGPUManager
from aiq.runtime.session import AIQSessionManager as Session
from aiq.builder.function import Function

logger = logging.getLogger(__name__)


class TaskExecutorService(node_pb2_grpc.TaskExecutorServicer):
    """gRPC service for task execution"""
    
    def __init__(self, worker_node: 'WorkerNode'):
        self.worker_node = worker_node
    
    async def ExecuteTask(self, request, context):
        """Execute a task on this worker"""
        try:
            # Deserialize inputs
            inputs = json.loads(request.serialized_inputs)
            
            # Execute task
            result = await self.worker_node.execute_task(
                task_id=request.task_id,
                function_name=request.function_name,
                inputs=inputs,
                metadata=dict(request.metadata)
            )
            
            # Serialize output
            serialized_output = json.dumps(result)
            
            return node_pb2.ExecuteTaskResponse(
                task_id=request.task_id,
                success=True,
                serialized_output=serialized_output
            )
        except Exception as e:
            logger.error(f"Error executing task {request.task_id}: {e}")
            return node_pb2.ExecuteTaskResponse(
                task_id=request.task_id,
                success=False,
                error_message=str(e)
            )


class WorkerNode:
    """
    Worker node that executes distributed tasks
    """
    
    def __init__(self, 
                 node_id: Optional[str] = None,
                 manager_host: str = "localhost",
                 manager_port: int = 50051,
                 worker_port: int = 50052):
        self.node_id = node_id or str(uuid.uuid4())
        self.manager_host = manager_host
        self.manager_port = manager_port
        self.worker_port = worker_port
        
        self.gpu_manager = MultiGPUManager()
        self.server = None
        self.manager_client = None
        self._running = False
        self._heartbeat_task = None
        
        # Function registry for task execution
        self.functions: Dict[str, Function] = {}
        
        # Task tracking
        self.current_tasks: Dict[str, Any] = {}
    
    async def start(self):
        """Start the worker node"""
        # Start executor service
        await self._start_executor_service()
        
        # Connect to manager
        await self._connect_to_manager()
        
        # Register with manager
        await self._register_with_manager()
        
        # Start heartbeat
        self._heartbeat_task = asyncio.create_task(self._send_heartbeats())
        
        self._running = True
        logger.info(f"Worker node {self.node_id} started")
    
    async def stop(self):
        """Stop the worker node"""
        self._running = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
        
        if self.server:
            await self.server.stop(grace=5)
        
        logger.info(f"Worker node {self.node_id} stopped")
    
    async def _start_executor_service(self):
        """Start gRPC service for task execution"""
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        node_pb2_grpc.add_TaskExecutorServicer_to_server(
            TaskExecutorService(self),
            self.server
        )
        
        self.server.add_insecure_port(f'[::]:{self.worker_port}')
        await self.server.start()
        
        logger.info(f"Task executor service started on port {self.worker_port}")
    
    async def _connect_to_manager(self):
        """Connect to the node manager"""
        channel = grpc.aio.insecure_channel(
            f'{self.manager_host}:{self.manager_port}'
        )
        self.manager_client = node_pb2_grpc.NodeManagerStub(channel)
    
    async def _register_with_manager(self):
        """Register this worker with the node manager"""
        import psutil
        
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        request = node_pb2.RegisterRequest(
            node_id=self.node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=self.worker_port,
            num_gpus=self.gpu_manager.device_count,
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3)
        )
        
        response = await self.manager_client.RegisterNode(request)
        
        if response.success:
            logger.info(f"Successfully registered with manager: {response.message}")
        else:
            raise RuntimeError(f"Failed to register with manager: {response.message}")
    
    async def _send_heartbeats(self):
        """Send periodic heartbeats to the manager"""
        while self._running:
            try:
                request = node_pb2.HeartbeatRequest(
                    node_id=self.node_id,
                    status="online",
                    current_tasks=list(self.current_tasks.keys())
                )
                
                response = await self.manager_client.Heartbeat(request)
                
                if not response.success:
                    logger.warning("Heartbeat failed")
                    
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
            
            await asyncio.sleep(10)  # Send heartbeat every 10 seconds
    
    def register_function(self, name: str, function: Function):
        """Register a function that can be executed"""
        self.functions[name] = function
        logger.info(f"Registered function: {name}")
    
    async def execute_task(self, 
                          task_id: str,
                          function_name: str,
                          inputs: Dict[str, Any],
                          metadata: Dict[str, Any]) -> Any:
        """
        Execute a task on this worker
        
        Args:
            task_id: Unique task identifier
            function_name: Name of function to execute
            inputs: Function inputs
            metadata: Additional metadata
            
        Returns:
            Task result
        """
        if function_name not in self.functions:
            raise ValueError(f"Unknown function: {function_name}")
        
        logger.info(f"Executing task {task_id} with function {function_name}")
        
        # Track task
        self.current_tasks[task_id] = {
            "function": function_name,
            "start_time": datetime.now(),
            "metadata": metadata
        }
        
        try:
            # Get the function
            function = self.functions[function_name]
            
            # Select GPU if specified in metadata
            gpu_id = metadata.get("gpu_id")
            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)
                logger.info(f"Using GPU {gpu_id} for task {task_id}")
            
            # Create session for this task
            session = Session()
            
            # Execute function
            result = function.run(session, inputs)
            
            logger.info(f"Task {task_id} completed successfully")
            return result
            
        finally:
            # Clean up task tracking
            if task_id in self.current_tasks:
                del self.current_tasks[task_id]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the worker node"""
        return {
            "node_id": self.node_id,
            "running": self._running,
            "current_tasks": len(self.current_tasks),
            "available_functions": list(self.functions.keys()),
            "gpu_info": self.gpu_manager.get_memory_summary()
        }


async def run_worker_node(
    manager_host: str = "localhost",
    manager_port: int = 50051,
    worker_port: int = 50052
):
    """
    Run a worker node
    
    Args:
        manager_host: Hostname of the node manager
        manager_port: Port of the node manager
        worker_port: Port for this worker's executor service
    """
    worker = WorkerNode(
        manager_host=manager_host,
        manager_port=manager_port,
        worker_port=worker_port
    )
    
    # Register example functions
    # In practice, these would be loaded from configuration
    from aiq.builder.function import Function
    from aiq.data_models.function import FunctionConfig
    
    # Example function
    config = FunctionConfig(
        name="example_function",
        inputs=["text"],
        outputs=["result"]
    )
    example_function = Function(config)
    worker.register_function("example_function", example_function)
    
    try:
        await worker.start()
        
        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Worker node shutting down...")
    finally:
        await worker.stop()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AIQToolkit Worker Node")
    parser.add_argument("--manager-host", default="localhost",
                       help="Node manager hostname")
    parser.add_argument("--manager-port", type=int, default=50051,
                       help="Node manager port")
    parser.add_argument("--worker-port", type=int, default=50052,
                       help="Worker executor port")
    
    args = parser.parse_args()
    
    asyncio.run(run_worker_node(
        manager_host=args.manager_host,
        manager_port=args.manager_port,
        worker_port=args.worker_port
    ))
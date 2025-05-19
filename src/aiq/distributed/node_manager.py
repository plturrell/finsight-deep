# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Node Manager for distributed AIQToolkit
Manages communication and coordination across multiple machines
"""

import asyncio
import grpc
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import uuid
from concurrent import futures

from aiq.distributed.grpc import node_pb2, node_pb2_grpc
from aiq.gpu.multi_gpu_manager import MultiGPUManager, GPUInfo

logger = logging.getLogger(__name__)


@dataclass
class NodeInfo:
    """Information about a compute node"""
    node_id: str
    hostname: str
    ip_address: str
    port: int
    num_gpus: int
    gpus: List[GPUInfo]
    status: str = "online"  # online, offline, busy
    last_heartbeat: datetime = field(default_factory=datetime.now)
    current_tasks: Set[str] = field(default_factory=set)
    cpu_count: int = 0
    memory_gb: float = 0.0
    
    def is_available(self) -> bool:
        """Check if node is available for tasks"""
        return self.status == "online" and len(self.current_tasks) < self.num_gpus
    
    def get_free_gpu_count(self) -> int:
        """Get number of free GPUs"""
        return max(0, self.num_gpus - len(self.current_tasks))


class NodeManagerService(node_pb2_grpc.NodeManagerServicer):
    """gRPC service for node management"""
    
    def __init__(self, node_manager: 'NodeManager'):
        self.node_manager = node_manager
    
    async def RegisterNode(self, request, context):
        """Handle node registration"""
        node_info = NodeInfo(
            node_id=request.node_id,
            hostname=request.hostname,
            ip_address=request.ip_address,
            port=request.port,
            num_gpus=request.num_gpus,
            gpus=[],  # Will be populated separately
            cpu_count=request.cpu_count,
            memory_gb=request.memory_gb
        )
        
        self.node_manager.register_node(node_info)
        
        return node_pb2.RegisterResponse(
            success=True,
            message=f"Node {request.node_id} registered successfully"
        )
    
    async def Heartbeat(self, request, context):
        """Handle node heartbeat"""
        success = self.node_manager.update_heartbeat(
            request.node_id,
            request.status,
            request.current_tasks
        )
        
        return node_pb2.HeartbeatResponse(success=success)
    
    async def AssignTask(self, request, context):
        """Assign task to a node"""
        node_id = self.node_manager.assign_task_to_node(
            request.task_id,
            request.resource_requirements
        )
        
        return node_pb2.AssignTaskResponse(
            node_id=node_id,
            success=node_id is not None
        )


class NodeManager:
    """
    Manages distributed nodes for AIQToolkit
    """
    
    def __init__(self, port: int = 50051):
        self.nodes: Dict[str, NodeInfo] = {}
        self.port = port
        self.server = None
        self.gpu_manager = MultiGPUManager()
        self.local_node_id = str(uuid.uuid4())
        self._running = False
        
    def start_server(self):
        """Start gRPC server for node management"""
        self.server = grpc.aio.server(
            futures.ThreadPoolExecutor(max_workers=10)
        )
        
        node_pb2_grpc.add_NodeManagerServicer_to_server(
            NodeManagerService(self),
            self.server
        )
        
        self.server.add_insecure_port(f'[::]:{self.port}')
        
        logger.info(f"Starting node manager server on port {self.port}")
        return self.server
    
    async def run(self):
        """Run the node manager"""
        await self.start_server()
        await self.server.start()
        self._running = True
        
        # Register self as a node
        self._register_local_node()
        
        # Start heartbeat monitor
        asyncio.create_task(self._monitor_heartbeats())
        
        try:
            await self.server.wait_for_termination()
        finally:
            self._running = False
    
    def stop(self):
        """Stop the node manager"""
        self._running = False
        if self.server:
            self.server.stop(grace=5)
    
    def register_node(self, node_info: NodeInfo):
        """Register a new node"""
        self.nodes[node_info.node_id] = node_info
        logger.info(f"Registered node {node_info.node_id} "
                   f"({node_info.hostname}) with {node_info.num_gpus} GPUs")
    
    def unregister_node(self, node_id: str):
        """Remove a node from the registry"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            logger.info(f"Unregistered node {node_id}")
    
    def update_heartbeat(self, node_id: str, status: str, 
                        current_tasks: List[str]) -> bool:
        """Update node heartbeat information"""
        if node_id not in self.nodes:
            return False
            
        node = self.nodes[node_id]
        node.last_heartbeat = datetime.now()
        node.status = status
        node.current_tasks = set(current_tasks)
        
        return True
    
    def get_available_nodes(self) -> List[NodeInfo]:
        """Get list of available nodes"""
        return [node for node in self.nodes.values() if node.is_available()]
    
    def get_total_resources(self) -> Dict[str, Any]:
        """Get total resources across all nodes"""
        total_gpus = sum(node.num_gpus for node in self.nodes.values())
        available_gpus = sum(node.get_free_gpu_count() 
                           for node in self.nodes.values())
        total_memory = sum(node.memory_gb for node in self.nodes.values())
        
        return {
            "total_nodes": len(self.nodes),
            "online_nodes": len([n for n in self.nodes.values() 
                               if n.status == "online"]),
            "total_gpus": total_gpus,
            "available_gpus": available_gpus,
            "total_memory_gb": total_memory
        }
    
    def assign_task_to_node(self, task_id: str, 
                           resource_requirements: Dict[str, Any]) -> Optional[str]:
        """
        Assign a task to the best available node
        
        Args:
            task_id: Unique task identifier
            resource_requirements: Required resources (gpus, memory, etc)
            
        Returns:
            Node ID if assignment successful, None otherwise
        """
        required_gpus = resource_requirements.get('gpus', 1)
        required_memory = resource_requirements.get('memory_gb', 0)
        
        # Find suitable nodes
        suitable_nodes = []
        for node in self.get_available_nodes():
            if (node.get_free_gpu_count() >= required_gpus and
                node.memory_gb >= required_memory):
                suitable_nodes.append(node)
        
        if not suitable_nodes:
            logger.warning(f"No suitable nodes found for task {task_id}")
            return None
        
        # Sort by available resources (simple load balancing)
        suitable_nodes.sort(
            key=lambda n: (n.get_free_gpu_count(), n.memory_gb),
            reverse=True
        )
        
        # Assign to best node
        selected_node = suitable_nodes[0]
        selected_node.current_tasks.add(task_id)
        
        logger.info(f"Assigned task {task_id} to node {selected_node.node_id}")
        return selected_node.node_id
    
    def release_task(self, task_id: str, node_id: str):
        """Release a task from a node"""
        if node_id in self.nodes:
            self.nodes[node_id].current_tasks.discard(task_id)
            logger.info(f"Released task {task_id} from node {node_id}")
    
    def _register_local_node(self):
        """Register the local node"""
        import socket
        import psutil
        
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        
        # Get GPU information
        gpu_infos = self.gpu_manager.get_gpu_info()
        
        local_node = NodeInfo(
            node_id=self.local_node_id,
            hostname=hostname,
            ip_address=ip_address,
            port=self.port,
            num_gpus=self.gpu_manager.device_count,
            gpus=gpu_infos,
            cpu_count=psutil.cpu_count(),
            memory_gb=psutil.virtual_memory().total / (1024**3)
        )
        
        self.register_node(local_node)
    
    async def _monitor_heartbeats(self):
        """Monitor node heartbeats and mark offline nodes"""
        while self._running:
            current_time = datetime.now()
            offline_nodes = []
            
            for node_id, node in self.nodes.items():
                # Skip local node
                if node_id == self.local_node_id:
                    continue
                    
                # Check if heartbeat is stale (> 30 seconds)
                time_since_heartbeat = (current_time - node.last_heartbeat).seconds
                if time_since_heartbeat > 30 and node.status == "online":
                    node.status = "offline"
                    offline_nodes.append(node_id)
            
            if offline_nodes:
                logger.warning(f"Marked nodes as offline: {offline_nodes}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    def get_node_info(self, node_id: str) -> Optional[NodeInfo]:
        """Get information about a specific node"""
        return self.nodes.get(node_id)
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get detailed cluster status"""
        status = {
            "cluster_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "nodes": {}
        }
        
        for node_id, node in self.nodes.items():
            status["nodes"][node_id] = {
                "hostname": node.hostname,
                "ip_address": node.ip_address,
                "status": node.status,
                "num_gpus": node.num_gpus,
                "free_gpus": node.get_free_gpu_count(),
                "current_tasks": list(node.current_tasks),
                "last_heartbeat": node.last_heartbeat.isoformat()
            }
        
        status["summary"] = self.get_total_resources()
        
        return status
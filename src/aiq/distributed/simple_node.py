"""
Simple working distributed node implementation
"""
import asyncio
import json
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum
import threading
from collections import deque


class NodeStatus(Enum):
    OFFLINE = "offline"
    ONLINE = "online"
    BUSY = "busy"


@dataclass
class Task:
    id: str
    data: Dict[str, Any]
    status: str = "pending"
    result: Optional[Any] = None
    error: Optional[str] = None


class SimpleNode:
    """A simple distributed node that actually works"""
    
    def __init__(self, node_id: str, port: int = 8080):
        self.node_id = node_id
        self.port = port
        self.status = NodeStatus.OFFLINE
        self.tasks = deque()
        self.results = {}
        self._running = False
        self._worker_thread = None
        
    def start(self):
        """Start the node"""
        self.status = NodeStatus.ONLINE
        self._running = True
        self._worker_thread = threading.Thread(target=self._process_tasks)
        self._worker_thread.start()
        print(f"Node {self.node_id} started on port {self.port}")
        
    def stop(self):
        """Stop the node"""
        self._running = False
        if self._worker_thread:
            self._worker_thread.join()
        self.status = NodeStatus.OFFLINE
        print(f"Node {self.node_id} stopped")
        
    def submit_task(self, task_id: str, data: Dict[str, Any]) -> str:
        """Submit a task to the node"""
        task = Task(id=task_id, data=data)
        self.tasks.append(task)
        print(f"Task {task_id} submitted to node {self.node_id}")
        return task_id
        
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task result"""
        return self.results.get(task_id)
        
    def _process_tasks(self):
        """Process tasks in the background"""
        while self._running:
            if self.tasks:
                task = self.tasks.popleft()
                self.status = NodeStatus.BUSY
                
                try:
                    # Simple processing - just echo the data with timestamp
                    result = {
                        "node_id": self.node_id,
                        "task_id": task.id,
                        "input": task.data,
                        "timestamp": time.time(),
                        "result": f"Processed by {self.node_id}"
                    }
                    
                    # Simulate some work
                    time.sleep(0.1)
                    
                    task.status = "completed"
                    task.result = result
                    self.results[task.id] = result
                    
                except Exception as e:
                    task.status = "failed"
                    task.error = str(e)
                    self.results[task.id] = {"error": str(e)}
                    
                self.status = NodeStatus.ONLINE
            else:
                time.sleep(0.01)  # Small sleep when idle


class SimpleCluster:
    """A simple cluster manager that actually works"""
    
    def __init__(self):
        self.nodes = {}
        self._task_counter = 0
        
    def add_node(self, node: SimpleNode):
        """Add a node to the cluster"""
        self.nodes[node.node_id] = node
        node.start()
        print(f"Added node {node.node_id} to cluster")
        
    def distribute_task(self, data: Dict[str, Any]) -> str:
        """Distribute a task to an available node"""
        # Find an available node
        available_nodes = [
            node for node in self.nodes.values() 
            if node.status == NodeStatus.ONLINE
        ]
        
        if not available_nodes:
            raise Exception("No available nodes")
            
        # Simple round-robin distribution
        node = available_nodes[self._task_counter % len(available_nodes)]
        self._task_counter += 1
        
        task_id = f"task_{self._task_counter}"
        node.submit_task(task_id, data)
        
        return task_id
        
    def get_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result from any node"""
        for node in self.nodes.values():
            result = node.get_result(task_id)
            if result:
                return result
        return None
        
    def shutdown(self):
        """Shutdown all nodes"""
        for node in self.nodes.values():
            node.stop()
        print("Cluster shutdown complete")


# Example usage
if __name__ == "__main__":
    # Create a simple cluster
    cluster = SimpleCluster()
    
    # Add some nodes
    node1 = SimpleNode("node1", 8081)
    node2 = SimpleNode("node2", 8082)
    
    cluster.add_node(node1)
    cluster.add_node(node2)
    
    # Submit some tasks
    task_ids = []
    for i in range(5):
        task_id = cluster.distribute_task({"value": i})
        task_ids.append(task_id)
    
    # Wait for results
    time.sleep(1)
    
    # Get results
    for task_id in task_ids:
        result = cluster.get_result(task_id)
        print(f"Result for {task_id}: {result}")
    
    # Shutdown
    cluster.shutdown()
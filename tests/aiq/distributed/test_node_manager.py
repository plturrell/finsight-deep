# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from aiq.distributed.node_manager import NodeManager, NodeInfo
from aiq.gpu.multi_gpu_manager import GPUInfo


class TestNodeManager:
    """Test node manager functionality"""
    
    @pytest.fixture
    def node_manager(self):
        """Create a node manager instance"""
        return NodeManager(port=50051)
    
    @pytest.fixture
    def sample_node(self):
        """Create a sample node info"""
        return NodeInfo(
            node_id="test-node-1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=2,
            gpus=[
                GPUInfo(
                    device_id=0,
                    name="A100",
                    memory_total=40 * 1024**3,
                    memory_free=35 * 1024**3,
                    utilization=10.0,
                    temperature=45.0,
                    power_draw=150.0
                ),
                GPUInfo(
                    device_id=1,
                    name="A100",
                    memory_total=40 * 1024**3,
                    memory_free=38 * 1024**3,
                    utilization=5.0,
                    temperature=42.0,
                    power_draw=140.0
                )
            ],
            cpu_count=32,
            memory_gb=256
        )
    
    def test_register_node(self, node_manager, sample_node):
        """Test node registration"""
        node_manager.register_node(sample_node)
        
        assert sample_node.node_id in node_manager.nodes
        assert node_manager.nodes[sample_node.node_id] == sample_node
    
    def test_unregister_node(self, node_manager, sample_node):
        """Test node unregistration"""
        node_manager.register_node(sample_node)
        node_manager.unregister_node(sample_node.node_id)
        
        assert sample_node.node_id not in node_manager.nodes
    
    def test_update_heartbeat(self, node_manager, sample_node):
        """Test heartbeat update"""
        node_manager.register_node(sample_node)
        
        # Update heartbeat
        success = node_manager.update_heartbeat(
            sample_node.node_id,
            "online",
            ["task1", "task2"]
        )
        
        assert success is True
        
        updated_node = node_manager.nodes[sample_node.node_id]
        assert updated_node.status == "online"
        assert updated_node.current_tasks == {"task1", "task2"}
    
    def test_get_available_nodes(self, node_manager):
        """Test getting available nodes"""
        # Register multiple nodes
        node1 = NodeInfo(
            node_id="node1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=2,
            gpus=[],
            status="online"
        )
        
        node2 = NodeInfo(
            node_id="node2",
            hostname="worker2",
            ip_address="192.168.1.11",
            port=50053,
            num_gpus=2,
            gpus=[],
            status="offline"
        )
        
        node3 = NodeInfo(
            node_id="node3",
            hostname="worker3",
            ip_address="192.168.1.12",
            port=50054,
            num_gpus=2,
            gpus=[],
            status="online",
            current_tasks={"task1", "task2"}  # Fully occupied
        )
        
        node_manager.register_node(node1)
        node_manager.register_node(node2)
        node_manager.register_node(node3)
        
        available = node_manager.get_available_nodes()
        
        # Only node1 should be available (online and has free capacity)
        assert len(available) == 1
        assert available[0].node_id == "node1"
    
    def test_get_total_resources(self, node_manager):
        """Test getting total cluster resources"""
        # Register nodes
        node1 = NodeInfo(
            node_id="node1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=2,
            gpus=[],
            status="online",
            memory_gb=128
        )
        
        node2 = NodeInfo(
            node_id="node2",
            hostname="worker2",
            ip_address="192.168.1.11",
            port=50053,
            num_gpus=4,
            gpus=[],
            status="online",
            current_tasks={"task1"},
            memory_gb=256
        )
        
        node_manager.register_node(node1)
        node_manager.register_node(node2)
        
        resources = node_manager.get_total_resources()
        
        assert resources["total_nodes"] == 2
        assert resources["online_nodes"] == 2
        assert resources["total_gpus"] == 6
        assert resources["available_gpus"] == 5  # 2 + (4 - 1)
        assert resources["total_memory_gb"] == 384
    
    def test_assign_task_to_node(self, node_manager):
        """Test task assignment to nodes"""
        # Register nodes with different capabilities
        node1 = NodeInfo(
            node_id="node1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=1,
            gpus=[],
            status="online",
            memory_gb=64
        )
        
        node2 = NodeInfo(
            node_id="node2",
            hostname="worker2",
            ip_address="192.168.1.11",
            port=50053,
            num_gpus=4,
            gpus=[],
            status="online",
            memory_gb=256
        )
        
        node_manager.register_node(node1)
        node_manager.register_node(node2)
        
        # Task requiring 2 GPUs should go to node2
        task_id = "task123"
        requirements = {"gpus": 2, "memory_gb": 100}
        
        assigned_node = node_manager.assign_task_to_node(task_id, requirements)
        
        assert assigned_node == "node2"
        assert task_id in node_manager.nodes["node2"].current_tasks
    
    def test_release_task(self, node_manager, sample_node):
        """Test releasing a task from a node"""
        node_manager.register_node(sample_node)
        
        # Assign a task
        task_id = "task123"
        sample_node.current_tasks.add(task_id)
        
        # Release the task
        node_manager.release_task(task_id, sample_node.node_id)
        
        assert task_id not in sample_node.current_tasks
    
    @pytest.mark.asyncio
    async def test_monitor_heartbeats(self, node_manager):
        """Test heartbeat monitoring"""
        # Register a node
        node = NodeInfo(
            node_id="node1",
            hostname="worker1",
            ip_address="192.168.1.10",
            port=50052,
            num_gpus=2,
            gpus=[],
            status="online"
        )
        node_manager.register_node(node)
        
        # Set running flag
        node_manager._running = True
        
        # Create a task for monitoring
        monitor_task = asyncio.create_task(node_manager._monitor_heartbeats())
        
        # Wait a bit
        await asyncio.sleep(0.1)
        
        # Simulate stale heartbeat
        node.last_heartbeat = datetime.now() - timedelta(seconds=40)
        
        # Wait for monitor to check
        await asyncio.sleep(0.1)
        
        # Node should be marked offline
        assert node.status == "offline"
        
        # Cleanup
        node_manager._running = False
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
#!/usr/bin/env python3
"""
Test a specific distributed component to see if it actually works
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Test the MultiGPUManager
print("Testing MultiGPUManager...")
try:
    from aiq.gpu.multi_gpu_manager import MultiGPUManager
    
    # Create an instance
    manager = MultiGPUManager()
    print(f"✓ Created MultiGPUManager")
    print(f"  Device count: {manager.device_count}")
    print(f"  Devices: {manager.devices}")
    
    # Try to get device info
    info = manager.get_device_info()
    print(f"  Device info: {info}")
    
except Exception as e:
    print(f"✗ MultiGPUManager failed: {e}")
    import traceback
    traceback.print_exc()

# Test the EdgeNode without external dependencies
print("\nTesting EdgeNode...")
try:
    from aiq.distributed.edge.edge_node import EdgeNode, EdgeNodeConfig, EdgeMode
    
    # Create a simple config
    config = EdgeNodeConfig(
        node_id="test_edge",
        device_type="workstation",
        mode=EdgeMode.OFFLINE_FIRST,
        sync_interval=3600,
        cache_size_mb=100
    )
    
    # Create the node
    node = EdgeNode(config)
    print(f"✓ Created EdgeNode")
    print(f"  Node ID: {node.config.node_id}")
    print(f"  Status: {node.status}")
    print(f"  Device type: {node.config.device_type}")
    print(f"  Mode: {node.config.mode}")
    
    # Test basic functionality
    node._queue_for_sync({"test": "data"})
    print(f"  Offline queue size: {len(node.offline_queue)}")
    
except Exception as e:
    print(f"✗ EdgeNode failed: {e}")
    import traceback
    traceback.print_exc()

# Test a simple workflow builder
print("\nTesting WorkflowBuilder...")
try:
    from aiq.data_models.config import WorkflowConfig
    from aiq.builder.workflow import Workflow
    
    # Create a minimal workflow
    workflow = Workflow(
        id="test_workflow",
        name="Test Workflow",
        description="Testing if this works"
    )
    print(f"✓ Created Workflow")
    print(f"  ID: {workflow.id}")
    print(f"  Name: {workflow.name}")
    
except Exception as e:
    print(f"✗ Workflow failed: {e}")
    import traceback
    traceback.print_exc()

print("\nConclusion: Most components fail due to missing dependencies.")
print("The code was written but never tested with actual imports/dependencies.")
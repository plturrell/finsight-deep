#!/usr/bin/env python3
"""
Basic test to verify what actually works
"""

import sys
import os

# Test 1: Can we import the AIQ modules?
print("=== Test 1: Import AIQ modules ===")
try:
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    import aiq
    print("✓ Can import aiq")
except Exception as e:
    print(f"✗ Cannot import aiq: {e}")

# Test 2: Check if our distributed modules exist
print("\n=== Test 2: Check distributed modules ===")
try:
    from aiq.distributed.node_manager import NodeManager
    print("✓ Can import NodeManager")
except Exception as e:
    print(f"✗ Cannot import NodeManager: {e}")

# Test 3: Try to create a basic object
print("\n=== Test 3: Create basic objects ===")
try:
    from aiq.builder.workflow import Workflow
    workflow = Workflow(id="test", name="Test Workflow", description="Basic test")
    print(f"✓ Created workflow: {workflow.name}")
except Exception as e:
    print(f"✗ Cannot create workflow: {e}")

# Test 4: Check GPU functionality (without PyTorch)
print("\n=== Test 4: Basic GPU check ===")
try:
    import subprocess
    result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        gpus = result.stdout.strip().split('\n')
        print(f"✓ Found {len(gpus)} GPU(s): {gpus}")
    else:
        print("✗ No NVIDIA GPUs found or nvidia-smi not available")
except Exception as e:
    print(f"✗ GPU check failed: {e}")

# Test 5: Check if gRPC works
print("\n=== Test 5: gRPC basic test ===")
try:
    import grpc
    print("✓ gRPC module available")
    # Try to create a basic server
    server = grpc.server(None)
    print("✓ Can create gRPC server object")
except Exception as e:
    print(f"✗ gRPC not available: {e}")

# Test 6: Test our actual node manager
print("\n=== Test 6: Test NodeManager creation ===")
try:
    from aiq.distributed.node_manager import NodeManager
    node_manager = NodeManager(port=50051)
    print("✓ Created NodeManager instance")
    print(f"  - Port: {node_manager.port}")
    print(f"  - Nodes: {len(node_manager.nodes)}")
except Exception as e:
    print(f"✗ Cannot create NodeManager: {e}")

# Test 7: Test edge node
print("\n=== Test 7: Test EdgeNode creation ===")
try:
    from aiq.distributed.edge.edge_node import EdgeNode, EdgeNodeConfig, EdgeMode
    config = EdgeNodeConfig(
        node_id="test_edge",
        device_type="workstation",
        mode=EdgeMode.INTERMITTENT
    )
    edge = EdgeNode(config)
    print("✓ Created EdgeNode instance")
    print(f"  - Node ID: {edge.config.node_id}")
    print(f"  - Status: {edge.status}")
except Exception as e:
    print(f"✗ Cannot create EdgeNode: {e}")

print("\n=== Summary ===")
print("This is a basic functionality test.")
print("For real testing, we need:")
print("1. PyTorch installed")
print("2. Actual GPUs")
print("3. Network configuration")
print("4. Proper test environment")
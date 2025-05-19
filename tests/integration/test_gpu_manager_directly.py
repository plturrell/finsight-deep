#!/usr/bin/env python3
"""
Test MultiGPUManager directly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

# Test MultiGPUManager in isolation
print("Testing MultiGPUManager directly...")

try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    print(f"✓ Number of GPUs: {torch.cuda.device_count()}")
    
    # Now test our actual MultiGPUManager
    from aiq.gpu.multi_gpu_manager import MultiGPUManager
    
    manager = MultiGPUManager()
    print(f"✓ Created MultiGPUManager")
    print(f"  - Device count: {manager.device_count}")
    print(f"  - Available devices: {manager.devices}")
    
    # Test device info
    info = manager.get_device_info()
    print(f"  - Device info: {info}")
    
    # Test simple model distribution
    if manager.device_count > 0:
        # Create a simple model
        model = torch.nn.Linear(10, 5)
        print(f"✓ Created test model")
        
        # Try to distribute it
        distributed_model = manager.distribute_model(model)
        print(f"✓ Distributed model across devices")
        
        # Test inference
        test_input = torch.randn(1, 10)
        result = distributed_model(test_input)
        print(f"✓ Model inference successful: {result.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
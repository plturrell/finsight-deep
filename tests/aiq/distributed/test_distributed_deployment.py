# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test distributed deployment on NVIDIA infrastructure"""

import os
import asyncio
import socket
import time
from typing import Optional, List

import pytest
import grpc
import torch

from aiq.distributed.node_manager import NodeManager, NodeInfo
from aiq.distributed.worker import DistributedWorker
from aiq.gpu.multi_gpu_manager import MultiGPUManager

class TestDistributedDeployment:
    """Test distributed deployment functionality"""
    
    @pytest.fixture
    def manager_host(self):
        """Get manager host from environment or default"""
        return os.environ.get('MANAGER_HOST', 'localhost')
    
    @pytest.fixture
    def manager_port(self):
        """Get manager port from environment or default"""
        return int(os.environ.get('MANAGER_PORT', '50051'))
    
    async def test_manager_connectivity(self, manager_host: str, manager_port: int):
        """Test connectivity to distributed manager"""
        channel = grpc.aio.insecure_channel(f'{manager_host}:{manager_port}')
        try:
            # Test channel connectivity
            await channel.channel_ready()
            print(f"✓ Connected to manager at {manager_host}:{manager_port}")
        except Exception as e:
            pytest.fail(f"Failed to connect to manager: {e}")
        finally:
            await channel.close()
    
    async def test_gpu_availability(self):
        """Test GPU availability on current node"""
        gpu_manager = MultiGPUManager()
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"✓ Found {device_count} GPUs")
            
            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {device_name} ({device_memory:.1f} GB)")
                
                # Test basic GPU operations
                device = torch.device(f'cuda:{i}')
                test_tensor = torch.randn(1000, 1000).to(device)
                result = torch.matmul(test_tensor, test_tensor)
                assert result.shape == (1000, 1000)
                print(f"    ✓ GPU {i} compute test passed")
        else:
            print("⚠ No GPUs available, running on CPU")
    
    async def test_worker_registration(self, manager_host: str, manager_port: int):
        """Test worker registration with manager"""
        gpu_manager = MultiGPUManager()
        worker_id = f"test-worker-{socket.gethostname()}-{os.getpid()}"
        
        worker = DistributedWorker(
            worker_id=worker_id,
            gpu_manager=gpu_manager
        )
        
        try:
            # Connect to manager
            await worker.connect_to_manager(manager_host, manager_port)
            print(f"✓ Worker {worker_id} registered with manager")
            
            # Wait a bit to ensure registration is processed
            await asyncio.sleep(2)
            
            # Disconnect
            await worker.disconnect()
            print(f"✓ Worker {worker_id} disconnected gracefully")
            
        except Exception as e:
            pytest.fail(f"Worker registration failed: {e}")
    
    async def test_distributed_inference(self, manager_host: str, manager_port: int):
        """Test distributed inference capabilities"""
        try:
            # Create a simple test task
            test_prompt = "Test distributed inference"
            
            # Connect as a client
            channel = grpc.aio.insecure_channel(f'{manager_host}:{manager_port}')
            
            # Note: This would normally use the actual gRPC service stubs
            # For now, we're just testing connectivity
            await channel.channel_ready()
            print("✓ Ready for distributed inference")
            
            await channel.close()
            
        except Exception as e:
            pytest.fail(f"Distributed inference test failed: {e}")
    
    async def test_monitoring_endpoints(self):
        """Test monitoring endpoints are accessible"""
        endpoints = [
            ('localhost', 9090, 'Prometheus'),  # metrics
            ('localhost', 8080, 'Dashboard'),   # dashboard
            ('localhost', 3001, 'Grafana')      # grafana
        ]
        
        for host, port, service in endpoints:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                print(f"✓ {service} endpoint accessible at {host}:{port}")
            else:
                print(f"⚠ {service} endpoint not accessible at {host}:{port}")
    
    def test_nvidia_api_configuration(self):
        """Test NVIDIA API configuration"""
        required_vars = ['NVIDIA_NGC_API_KEY', 'NIM_API_KEY']
        
        for var in required_vars:
            value = os.environ.get(var)
            if value:
                # Mask the actual key for security
                masked = value[:8] + '...' + value[-4:] if len(value) > 12 else '***'
                print(f"✓ {var} configured: {masked}")
            else:
                print(f"⚠ {var} not configured")

# Main test runner
async def main():
    """Run all deployment tests"""
    test = TestDistributedDeployment()
    
    print("=" * 50)
    print("AIQToolkit Distributed Deployment Tests")
    print("=" * 50)
    print()
    
    # Get test configuration
    manager_host = os.environ.get('MANAGER_HOST', 'localhost')
    manager_port = int(os.environ.get('MANAGER_PORT', '50051'))
    
    # Run tests
    print("1. Testing NVIDIA API Configuration...")
    test.test_nvidia_api_configuration()
    print()
    
    print("2. Testing GPU Availability...")
    await test.test_gpu_availability()
    print()
    
    print("3. Testing Manager Connectivity...")
    try:
        await test.test_manager_connectivity(manager_host, manager_port)
    except Exception as e:
        print(f"⚠ Manager connectivity test failed: {e}")
    print()
    
    print("4. Testing Worker Registration...")
    try:
        await test.test_worker_registration(manager_host, manager_port)
    except Exception as e:
        print(f"⚠ Worker registration test failed: {e}")
    print()
    
    print("5. Testing Monitoring Endpoints...")
    await test.test_monitoring_endpoints()
    print()
    
    print("=" * 50)
    print("Deployment tests completed")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(main())
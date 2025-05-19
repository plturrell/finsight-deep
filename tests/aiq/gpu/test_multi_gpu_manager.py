# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

from aiq.gpu.multi_gpu_manager import MultiGPUManager, GPUInfo


class TestMultiGPUManager:
    """Test multi-GPU management functionality"""
    
    @pytest.fixture
    def mock_cuda_env(self):
        """Mock CUDA environment"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2), \
             patch('torch.cuda.get_device_properties') as mock_props:
            
            # Mock device properties
            mock_device_props = Mock()
            mock_device_props.name = "NVIDIA A100"
            mock_device_props.total_memory = 40 * 1024 * 1024 * 1024  # 40GB
            mock_props.return_value = mock_device_props
            
            yield
    
    @pytest.fixture
    def gpu_manager(self, mock_cuda_env):
        """Create GPU manager with mocked environment"""
        with patch('aiq.gpu.multi_gpu_manager.pynvml'):
            manager = MultiGPUManager()
            manager.nvml_available = False  # Disable NVML for tests
            return manager
    
    def test_initialization(self, gpu_manager):
        """Test GPU manager initialization"""
        assert gpu_manager.device_count == 2
        assert gpu_manager.devices == [0, 1]
    
    def test_get_gpu_info(self, gpu_manager):
        """Test getting GPU information"""
        gpu_infos = gpu_manager.get_gpu_info()
        
        assert len(gpu_infos) == 2
        assert all(isinstance(info, GPUInfo) for info in gpu_infos)
        assert gpu_infos[0].device_id == 0
        assert gpu_infos[0].name == "NVIDIA A100"
    
    def test_select_best_gpu(self, gpu_manager):
        """Test GPU selection based on memory"""
        with patch.object(gpu_manager, 'get_gpu_info') as mock_get_info:
            # Mock GPU info with different memory availability
            mock_get_info.return_value = [
                GPUInfo(
                    device_id=0,
                    name="GPU 0",
                    memory_total=40 * 1024**3,
                    memory_free=10 * 1024**3,
                    utilization=80.0,
                    temperature=70.0,
                    power_draw=250.0
                ),
                GPUInfo(
                    device_id=1,
                    name="GPU 1",
                    memory_total=40 * 1024**3,
                    memory_free=30 * 1024**3,
                    utilization=20.0,
                    temperature=60.0,
                    power_draw=200.0
                )
            ]
            
            # Should select GPU 1 (more free memory)
            best_gpu = gpu_manager.select_best_gpu(min_memory_mb=5000)
            assert best_gpu == 1
    
    def test_distribute_model(self, gpu_manager):
        """Test model distribution across GPUs"""
        model = nn.Linear(100, 10)
        
        # Single GPU case
        with patch.object(gpu_manager, 'device_count', 1):
            distributed = gpu_manager.distribute_model(model)
            assert not isinstance(distributed, nn.DataParallel)
        
        # Multi-GPU case
        distributed = gpu_manager.distribute_model(model)
        assert isinstance(distributed, nn.DataParallel)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_distribute_data(self, gpu_manager):
        """Test data distribution across GPUs"""
        data = torch.randn(100, 10)
        batch_size = 32
        
        distributed_data = gpu_manager.distribute_data(data, batch_size)
        
        # Check that data is properly distributed
        assert len(distributed_data) == 4  # 100 / 32 = 3.125, so 4 chunks
        assert sum(chunk.size(0) for chunk in distributed_data) == 100
    
    def test_get_memory_summary(self, gpu_manager):
        """Test memory summary retrieval"""
        with patch.object(gpu_manager, 'get_gpu_info') as mock_get_info:
            mock_get_info.side_effect = lambda device_id: [
                GPUInfo(
                    device_id=device_id[0] if isinstance(device_id, list) else device_id,
                    name=f"GPU {device_id[0] if isinstance(device_id, list) else device_id}",
                    memory_total=40 * 1024**3,
                    memory_free=30 * 1024**3,
                    utilization=25.0,
                    temperature=65.0,
                    power_draw=225.0
                )
            ]
            
            summary = gpu_manager.get_memory_summary()
            
            assert len(summary) == 2
            assert all(device in summary for device in [0, 1])
            assert summary[0]['total_gb'] == pytest.approx(40.0, rel=0.1)
            assert summary[0]['free_gb'] == pytest.approx(30.0, rel=0.1)
    
    def test_optimize_batch_size(self, gpu_manager):
        """Test batch size optimization"""
        model = nn.Linear(1000, 1000)
        input_shape = (1000,)
        
        with patch.object(gpu_manager, 'get_gpu_info') as mock_get_info:
            mock_get_info.return_value = [
                GPUInfo(
                    device_id=0,
                    name="GPU 0",
                    memory_total=40 * 1024**3,
                    memory_free=20 * 1024**3,
                    utilization=50.0,
                    temperature=70.0,
                    power_draw=250.0
                )
            ]
            
            with patch('torch.cuda.memory_allocated', side_effect=[0, 100 * 1024**2]):
                optimal_batch = gpu_manager.optimize_batch_size(
                    model, 
                    input_shape,
                    target_memory_usage=0.8
                )
                
                assert optimal_batch > 0
                assert optimal_batch < 1000  # Should be reasonable
    
    def test_gpu_info_properties(self):
        """Test GPUInfo properties"""
        gpu_info = GPUInfo(
            device_id=0,
            name="Test GPU",
            memory_total=40 * 1024**3,
            memory_free=30 * 1024**3,
            utilization=25.0,
            temperature=65.0,
            power_draw=225.0
        )
        
        assert gpu_info.memory_used == 10 * 1024**3
        assert gpu_info.memory_usage_percent == pytest.approx(25.0, rel=0.1)
    
    def test_no_gpu_error_handling(self):
        """Test error handling when no GPUs available"""
        with patch('torch.cuda.device_count', return_value=0):
            gpu_manager = MultiGPUManager()
            assert gpu_manager.device_count == 0
            assert gpu_manager.devices == []


# Integration test with actual PyTorch model
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestMultiGPUIntegration:
    """Integration tests requiring actual GPU"""
    
    def test_real_model_distribution(self):
        """Test distributing a real model across GPUs"""
        gpu_manager = MultiGPUManager()
        
        if gpu_manager.device_count < 2:
            pytest.skip("Multiple GPUs not available")
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        # Distribute model
        distributed_model = gpu_manager.distribute_model(model)
        
        # Test forward pass
        input_data = torch.randn(32, 100).cuda()
        output = distributed_model(input_data)
        
        assert output.shape == (32, 10)
        assert output.device.type == 'cuda'
    
    def test_real_data_distribution(self):
        """Test distributing real data across GPUs"""
        gpu_manager = MultiGPUManager()
        
        # Create test data
        data = torch.randn(1000, 100)
        
        # Distribute data
        distributed = gpu_manager.distribute_data(data, batch_size=256)
        
        # Verify distribution
        assert len(distributed) == 4  # 1000 / 256 = 3.9, so 4 chunks
        assert all(chunk.device.type == 'cuda' for chunk in distributed)
        
        # Test gathering
        gathered = gpu_manager.gather_results(distributed)
        assert gathered.shape == data.shape
        assert torch.allclose(gathered.cpu(), data)
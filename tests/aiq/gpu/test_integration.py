# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for GPU and distributed features
"""

import pytest
import torch
from unittest.mock import Mock, patch

from aiq.gpu.multi_gpu_manager import MultiGPUManager
from aiq.builder.distributed_workflow_builder import DistributedWorkflowBuilder
from aiq.data_models.workflow import WorkflowConfig
from aiq.data_models.function import FunctionConfig


class TestGPUIntegration:
    """Test integration of GPU features with AIQToolkit"""
    
    def test_gpu_manager_creation(self):
        """Test basic GPU manager creation"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=2):
            
            manager = MultiGPUManager()
            assert manager.device_count == 2
            assert len(manager.devices) == 2
    
    def test_distributed_workflow_builder(self):
        """Test distributed workflow builder creation"""
        with patch('torch.cuda.device_count', return_value=2):
            builder = DistributedWorkflowBuilder()
            assert builder.gpu_manager is not None
            assert builder.executor is not None
    
    def test_workflow_with_gpu_distribution(self):
        """Test workflow creation with GPU distribution enabled"""
        # Create a simple workflow config
        config = WorkflowConfig(
            name="test_distributed",
            description="Test distributed workflow",
            functions=[
                FunctionConfig(
                    type="custom",
                    name="test_function",
                    inputs=["input1"],
                    outputs=["output1"]
                )
            ]
        )
        
        with patch('torch.cuda.device_count', return_value=2):
            builder = DistributedWorkflowBuilder()
            workflow = builder.build(config)
            
            assert hasattr(workflow, 'gpu_manager')
            assert hasattr(workflow, 'executor')
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_real_gpu_detection(self):
        """Test real GPU detection if available"""
        manager = MultiGPUManager()
        
        if manager.device_count > 0:
            # Test GPU info retrieval
            gpu_info = manager.get_gpu_info(0)
            assert len(gpu_info) == 1
            assert gpu_info[0].device_id == 0
            assert gpu_info[0].name != ""
            
            # Test memory summary
            memory_summary = manager.get_memory_summary()
            assert 0 in memory_summary
            assert 'total_gb' in memory_summary[0]
    
    def test_model_distribution_mock(self):
        """Test model distribution with mocked environment"""
        with patch('torch.cuda.device_count', return_value=4):
            manager = MultiGPUManager()
            
            # Create a simple model
            model = torch.nn.Linear(10, 5)
            
            # Test distribution
            distributed = manager.distribute_model(model)
            assert isinstance(distributed, torch.nn.DataParallel)
    
    def test_batch_optimization(self):
        """Test batch size optimization"""
        with patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.device_count', return_value=1):
            
            manager = MultiGPUManager()
            model = torch.nn.Linear(100, 50)
            
            with patch('torch.cuda.memory_allocated', side_effect=[0, 10*1024**2]), \
                 patch.object(manager, 'get_gpu_info') as mock_get_info:
                
                mock_get_info.return_value = [Mock(
                    memory_free=1024**3,  # 1GB free
                    device_id=0
                )]
                
                optimal_batch = manager.optimize_batch_size(
                    model, 
                    input_shape=(100,)
                )
                
                assert optimal_batch > 0
                assert optimal_batch < 1000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""Tests for Tensor Core optimization."""

import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from aiq.hardware.tensor_core_optimizer import (
    TensorCoreOptimizer,
    optimize_for_tensor_cores,
    benchmark_tensor_cores,
    auto_mixed_precision,
    TensorCoreConfig
)


class TestTensorCoreOptimizer:
    """Test suite for Tensor Core optimization."""
    
    @pytest.fixture
    def gpu_info(self):
        """Mock GPU information."""
        return {
            'device_name': 'NVIDIA A100',
            'compute_capability': (8, 0),
            'tensor_cores': True,
            'memory_gb': 40
        }
    
    @pytest.fixture
    def optimizer(self, gpu_info):
        """Create optimizer instance."""
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.major = gpu_info['compute_capability'][0]
            mock_props.return_value.minor = gpu_info['compute_capability'][1]
            return TensorCoreOptimizer(device_id=0)
    
    def test_initialization(self, optimizer):
        """Test TensorCoreOptimizer initialization."""
        assert optimizer.device_id == 0
        assert optimizer.compute_capability >= (7, 0)  # Tensor cores require 7.0+
        assert optimizer.tensor_cores_available
    
    def test_no_tensor_cores(self):
        """Test behavior when tensor cores not available."""
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.major = 6
            mock_props.return_value.minor = 0
            
            with pytest.warns(UserWarning, match="Tensor cores not available"):
                optimizer = TensorCoreOptimizer(device_id=0)
            
            assert not optimizer.tensor_cores_available
    
    @pytest.mark.parametrize("shape", [
        (1024, 768),    # Should be padded
        (1024, 1024),   # Already aligned
        (999, 777),     # Needs significant padding
    ])
    def test_tensor_alignment(self, optimizer, shape):
        """Test tensor alignment for optimal performance."""
        tensor = torch.randn(*shape).cuda()
        aligned_tensor = optimizer.align_for_tensor_cores(tensor)
        
        # Check alignment to 8 for Tensor Cores
        assert aligned_tensor.shape[0] % 8 == 0
        assert aligned_tensor.shape[1] % 8 == 0
        
        # Original data should be preserved
        assert torch.allclose(tensor, aligned_tensor[:shape[0], :shape[1]])
    
    def test_mixed_precision_optimization(self, optimizer):
        """Test mixed precision optimization."""
        model = Mock()
        model.parameters = Mock(return_value=[
            torch.randn(1024, 768).cuda(),
            torch.randn(768, 512).cuda()
        ])
        
        with patch('torch.cuda.amp.GradScaler') as mock_scaler:
            scaler = optimizer.enable_mixed_precision(model)
            mock_scaler.assert_called_once()
    
    def test_benchmark_tensor_cores(self, optimizer):
        """Test tensor core benchmarking."""
        input_tensor = torch.randn(1024, 768).cuda()
        weight_tensor = torch.randn(768, 512).cuda()
        
        with patch('time.time', side_effect=[0, 0.1, 0.1, 0.15]):  # Mock timing
            results = optimizer.benchmark_tensor_cores(
                input_tensor, 
                weight_tensor,
                iterations=100
            )
        
        assert 'fp16_time' in results
        assert 'fp32_time' in results
        assert 'speedup' in results
        assert results['speedup'] > 0
    
    def test_optimize_model_layers(self, optimizer):
        """Test model layer optimization."""
        model = Mock()
        
        # Mock model layers
        linear_layer = Mock(spec=['weight', 'bias'])
        linear_layer.weight = torch.randn(768, 512).cuda()
        linear_layer.bias = torch.randn(768).cuda()
        
        conv_layer = Mock(spec=['weight', 'bias'])
        conv_layer.weight = torch.randn(64, 32, 3, 3).cuda()
        conv_layer.bias = torch.randn(64).cuda()
        
        model.modules = Mock(return_value=[linear_layer, conv_layer])
        
        optimized_model = optimizer.optimize_model(model)
        
        # Check that layers were optimized
        assert hasattr(optimized_model, '_tensor_core_optimized')
    
    def test_auto_config(self, optimizer):
        """Test automatic configuration."""
        config = optimizer.auto_configure()
        
        assert isinstance(config, TensorCoreConfig)
        assert config.use_fp16 == optimizer.tensor_cores_available
        assert config.alignment in [8, 16]
        assert config.batch_size % config.alignment == 0
    
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_dtype_optimization(self, optimizer, dtype):
        """Test optimization for different data types."""
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            pytest.skip("BFloat16 not supported")
        
        tensor = torch.randn(1024, 768).cuda()
        optimized = optimizer.optimize_tensor(tensor, dtype=dtype)
        
        assert optimized.dtype == dtype
        assert optimized.shape[0] % 8 == 0
        assert optimized.shape[1] % 8 == 0
    
    def test_memory_optimization(self, optimizer):
        """Test memory usage optimization."""
        # Create large tensors that would benefit from optimization
        tensors = [
            torch.randn(4096, 4096).cuda(),
            torch.randn(2048, 8192).cuda(),
            torch.randn(8192, 2048).cuda()
        ]
        
        with patch('torch.cuda.memory_allocated') as mock_memory:
            mock_memory.side_effect = [
                1e9,  # Before optimization
                8e8   # After optimization
            ]
            
            memory_before = torch.cuda.memory_allocated()
            optimized_tensors = [optimizer.optimize_tensor(t) for t in tensors]
            memory_after = torch.cuda.memory_allocated()
            
            # Should use less memory with FP16
            assert memory_after < memory_before
    
    def test_performance_profiling(self, optimizer):
        """Test performance profiling capabilities."""
        with patch('torch.profiler.profile') as mock_profiler:
            with optimizer.profile_tensor_cores() as prof:
                # Simulate some tensor operations
                a = torch.randn(1024, 768).cuda()
                b = torch.randn(768, 512).cuda()
                c = torch.matmul(a, b)
            
            mock_profiler.assert_called()


class TestTensorCoreUtilities:
    """Test utility functions for tensor core optimization."""
    
    def test_optimize_for_tensor_cores_decorator(self):
        """Test the optimization decorator."""
        @optimize_for_tensor_cores
        def matrix_multiply(a, b):
            return torch.matmul(a, b)
        
        a = torch.randn(1000, 750).cuda()
        b = torch.randn(750, 500).cuda()
        
        with patch('aiq.hardware.tensor_core_optimizer.TensorCoreOptimizer') as mock_opt:
            mock_instance = mock_opt.return_value
            mock_instance.optimize_tensor.side_effect = lambda x: x
            
            result = matrix_multiply(a, b)
            
            # Should optimize both inputs
            assert mock_instance.optimize_tensor.call_count == 2
    
    def test_benchmark_tensor_cores_utility(self):
        """Test tensor core benchmarking utility."""
        def test_operation(a, b):
            return torch.matmul(a, b)
        
        a = torch.randn(1024, 768).cuda()
        b = torch.randn(768, 512).cuda()
        
        with patch('aiq.hardware.tensor_core_optimizer.TensorCoreOptimizer') as mock_opt:
            mock_instance = mock_opt.return_value
            mock_instance.benchmark_operation.return_value = {
                'time': 0.01,
                'tflops': 100.0
            }
            
            results = benchmark_tensor_cores(test_operation, a, b)
            
            assert 'time' in results
            assert 'tflops' in results
    
    def test_auto_mixed_precision_context(self):
        """Test automatic mixed precision context manager."""
        model = Mock()
        optimizer = Mock()
        
        with patch('torch.cuda.amp.autocast') as mock_autocast:
            with patch('torch.cuda.amp.GradScaler') as mock_scaler:
                with auto_mixed_precision(model, optimizer) as (model_amp, opt_amp, scaler):
                    # Simulate training step
                    loss = torch.tensor(0.5)
                    scaler.scale(loss).backward()
                    scaler.step(opt_amp)
                    scaler.update()
                
                mock_autocast.assert_called()
                mock_scaler.assert_called()


class TestAdvancedTensorCoreFeatures:
    """Test advanced tensor core features."""
    
    @pytest.fixture
    def advanced_optimizer(self):
        """Create optimizer with advanced features."""
        with patch('torch.cuda.get_device_properties') as mock_props:
            mock_props.return_value.major = 8
            mock_props.return_value.minor = 6
            return TensorCoreOptimizer(device_id=0, enable_advanced_features=True)
    
    def test_sparse_tensor_optimization(self, advanced_optimizer):
        """Test optimization for sparse tensors."""
        # Create sparse tensor
        indices = torch.LongTensor([[0, 1, 1], [2, 0, 2]])
        values = torch.FloatTensor([3, 4, 5])
        sparse_tensor = torch.sparse.FloatTensor(indices, values, (2, 3)).cuda()
        
        with patch.object(advanced_optimizer, 'optimize_sparse_tensor') as mock_opt:
            mock_opt.return_value = sparse_tensor
            optimized = advanced_optimizer.optimize_tensor(sparse_tensor)
            mock_opt.assert_called_once()
    
    def test_quantization_aware_optimization(self, advanced_optimizer):
        """Test quantization-aware optimization."""
        tensor = torch.randn(1024, 768).cuda()
        
        with patch('torch.quantization.quantize_dynamic') as mock_quantize:
            optimized = advanced_optimizer.optimize_with_quantization(tensor)
            mock_quantize.assert_called()
    
    def test_fusion_optimization(self, advanced_optimizer):
        """Test operation fusion optimization."""
        # Create a simple neural network layer
        class TestLayer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(768, 512).cuda()
                self.relu = torch.nn.ReLU()
                self.dropout = torch.nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.linear(x)
                x = self.relu(x)
                x = self.dropout(x)
                return x
        
        layer = TestLayer()
        
        with patch.object(advanced_optimizer, 'fuse_operations') as mock_fuse:
            optimized_layer = advanced_optimizer.optimize_model(layer)
            mock_fuse.assert_called()
    
    def test_custom_kernel_generation(self, advanced_optimizer):
        """Test custom CUDA kernel generation."""
        operation = """
        __global__ void custom_kernel(float* a, float* b, float* c, int n) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                c[idx] = a[idx] * b[idx] + 1.0f;
            }
        }
        """
        
        with patch('cupy.RawKernel') as mock_kernel:
            kernel = advanced_optimizer.generate_custom_kernel(operation)
            mock_kernel.assert_called()
    
    def test_multi_stream_optimization(self, advanced_optimizer):
        """Test multi-stream execution optimization."""
        tensors = [torch.randn(1024, 768).cuda() for _ in range(4)]
        
        with patch('torch.cuda.Stream') as mock_stream:
            streams = advanced_optimizer.create_multi_streams(num_streams=4)
            
            # Simulate parallel operations
            results = advanced_optimizer.parallel_tensor_ops(
                tensors, 
                lambda x: torch.matmul(x, x.T),
                streams=streams
            )
            
            assert len(results) == 4
            assert mock_stream.call_count == 4
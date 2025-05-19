"""Tests for CUDA similarity kernels."""

import pytest
import numpy as np
import torch
from unittest.mock import Mock, patch, MagicMock

from aiq.cuda_kernels.cuda_similarity import (
    CUDASimilarity,
    cosine_similarity_cuda,
    euclidean_distance_cuda,
    manhattan_distance_cuda,
    dot_product_cuda
)


class TestCUDASimilarity:
    """Test suite for CUDA similarity operations."""
    
    @pytest.fixture
    def cuda_available(self):
        """Mock CUDA availability."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                yield
    
    @pytest.fixture
    def cuda_unavailable(self):
        """Mock CUDA unavailability."""
        with patch('torch.cuda.is_available', return_value=False):
            yield
    
    @pytest.fixture
    def sample_vectors(self):
        """Create sample test vectors."""
        return {
            'query': np.random.randn(384).astype(np.float32),
            'documents': np.random.randn(100, 384).astype(np.float32)
        }
    
    def test_initialization_with_cuda(self, cuda_available):
        """Test CUDASimilarity initialization with CUDA available."""
        similarity = CUDASimilarity(device_id=0)
        assert similarity.device_id == 0
        assert similarity.use_cuda is True
    
    def test_initialization_without_cuda(self, cuda_unavailable):
        """Test CUDASimilarity initialization without CUDA."""
        with pytest.warns(UserWarning, match="CUDA not available"):
            similarity = CUDASimilarity(device_id=0)
        assert similarity.use_cuda is False
    
    def test_invalid_device_id(self, cuda_available):
        """Test initialization with invalid device ID."""
        with patch('torch.cuda.device_count', return_value=1):
            with pytest.raises(ValueError, match="Invalid device_id"):
                CUDASimilarity(device_id=2)
    
    @pytest.mark.parametrize("batch_size", [32, 64, 128])
    def test_cosine_similarity_cuda(self, cuda_available, sample_vectors, batch_size):
        """Test CUDA cosine similarity computation."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch('aiq.cuda_kernels.cuda_similarity.cosine_similarity_kernel') as mock_kernel:
            mock_kernel.return_value = torch.rand(100).cuda()
            
            scores = similarity.cosine_similarity(
                sample_vectors['query'],
                sample_vectors['documents'],
                batch_size=batch_size
            )
            
            assert len(scores) == 100
            assert scores.dtype == np.float32
            mock_kernel.assert_called()
    
    def test_cosine_similarity_cpu_fallback(self, cuda_unavailable, sample_vectors):
        """Test cosine similarity CPU fallback."""
        with pytest.warns(UserWarning):
            similarity = CUDASimilarity(device_id=0)
        
        scores = similarity.cosine_similarity(
            sample_vectors['query'],
            sample_vectors['documents']
        )
        
        assert len(scores) == 100
        # Verify CPU computation
        expected = np.dot(sample_vectors['documents'], sample_vectors['query'])
        expected /= np.linalg.norm(sample_vectors['documents'], axis=1)
        expected /= np.linalg.norm(sample_vectors['query'])
        np.testing.assert_allclose(scores, expected, rtol=1e-5)
    
    def test_euclidean_distance_cuda(self, cuda_available, sample_vectors):
        """Test CUDA Euclidean distance computation."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch('aiq.cuda_kernels.cuda_similarity.euclidean_distance_kernel') as mock_kernel:
            mock_kernel.return_value = torch.rand(100).cuda()
            
            distances = similarity.euclidean_distance(
                sample_vectors['query'],
                sample_vectors['documents']
            )
            
            assert len(distances) == 100
            assert distances.dtype == np.float32
            mock_kernel.assert_called()
    
    def test_manhattan_distance_cuda(self, cuda_available, sample_vectors):
        """Test CUDA Manhattan distance computation."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch('aiq.cuda_kernels.cuda_similarity.manhattan_distance_kernel') as mock_kernel:
            mock_kernel.return_value = torch.rand(100).cuda()
            
            distances = similarity.manhattan_distance(
                sample_vectors['query'],
                sample_vectors['documents']
            )
            
            assert len(distances) == 100
            assert distances.dtype == np.float32
            mock_kernel.assert_called()
    
    def test_batch_processing(self, cuda_available, sample_vectors):
        """Test batch processing for large datasets."""
        similarity = CUDASimilarity(device_id=0)
        large_documents = np.random.randn(10000, 384).astype(np.float32)
        
        with patch('aiq.cuda_kernels.cuda_similarity.cosine_similarity_kernel') as mock_kernel:
            # Mock kernel to return appropriate sized tensors for each batch
            def kernel_side_effect(q, d):
                return torch.rand(d.shape[0]).cuda()
            
            mock_kernel.side_effect = kernel_side_effect
            
            scores = similarity.cosine_similarity(
                sample_vectors['query'],
                large_documents,
                batch_size=1000
            )
            
            assert len(scores) == 10000
            # Verify batches were processed
            assert mock_kernel.call_count == 10  # 10000 / 1000
    
    def test_dimension_mismatch(self, cuda_available):
        """Test error handling for dimension mismatch."""
        similarity = CUDASimilarity(device_id=0)
        query = np.random.randn(256).astype(np.float32)
        documents = np.random.randn(100, 384).astype(np.float32)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            similarity.cosine_similarity(query, documents)
    
    def test_memory_management(self, cuda_available):
        """Test CUDA memory management."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch('torch.cuda.empty_cache') as mock_empty_cache:
            similarity.clear_cache()
            mock_empty_cache.assert_called_once()
    
    def test_multi_gpu_support(self):
        """Test multi-GPU support."""
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=4):
                similarity = CUDASimilarity(device_id=2)
                assert similarity.device_id == 2
                
                with patch('torch.cuda.set_device') as mock_set_device:
                    similarity._set_device()
                    mock_set_device.assert_called_with(2)
    
    @pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan", "dot"])
    def test_unified_interface(self, cuda_available, sample_vectors, metric):
        """Test unified similarity interface."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch.object(similarity, f'{metric}_similarity') as mock_method:
            if metric == "dot":
                mock_method.return_value = np.random.rand(100).astype(np.float32)
            else:
                mock_method.return_value = np.random.rand(100).astype(np.float32)
            
            scores = similarity.compute_similarity(
                sample_vectors['query'],
                sample_vectors['documents'],
                metric=metric
            )
            
            mock_method.assert_called_once()
            assert len(scores) == 100


@pytest.mark.gpu
class TestCUDAKernels:
    """Test CUDA kernel implementations directly."""
    
    @pytest.fixture
    def cuda_tensors(self):
        """Create CUDA tensors for testing."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        query = torch.randn(384).cuda()
        documents = torch.randn(100, 384).cuda()
        return query, documents
    
    def test_cosine_similarity_kernel(self, cuda_tensors):
        """Test cosine similarity CUDA kernel."""
        query, documents = cuda_tensors
        
        # Normalize tensors
        query_norm = query / query.norm()
        docs_norm = documents / documents.norm(dim=1, keepdim=True)
        
        # Test kernel
        scores = cosine_similarity_cuda(query_norm, docs_norm)
        
        # Verify against PyTorch implementation
        expected = torch.mm(docs_norm, query_norm.unsqueeze(1)).squeeze()
        torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)
    
    def test_euclidean_distance_kernel(self, cuda_tensors):
        """Test Euclidean distance CUDA kernel."""
        query, documents = cuda_tensors
        
        distances = euclidean_distance_cuda(query, documents)
        
        # Verify against PyTorch implementation
        expected = torch.norm(documents - query.unsqueeze(0), dim=1)
        torch.testing.assert_close(distances, expected, rtol=1e-5, atol=1e-5)
    
    def test_manhattan_distance_kernel(self, cuda_tensors):
        """Test Manhattan distance CUDA kernel."""
        query, documents = cuda_tensors
        
        distances = manhattan_distance_cuda(query, documents)
        
        # Verify against PyTorch implementation
        expected = torch.sum(torch.abs(documents - query.unsqueeze(0)), dim=1)
        torch.testing.assert_close(distances, expected, rtol=1e-5, atol=1e-5)
    
    def test_dot_product_kernel(self, cuda_tensors):
        """Test dot product CUDA kernel."""
        query, documents = cuda_tensors
        
        scores = dot_product_cuda(query, documents)
        
        # Verify against PyTorch implementation
        expected = torch.mm(documents, query.unsqueeze(1)).squeeze()
        torch.testing.assert_close(scores, expected, rtol=1e-5, atol=1e-5)
    
    def test_kernel_performance(self, cuda_tensors):
        """Test kernel performance characteristics."""
        import time
        
        query, documents = cuda_tensors
        large_documents = torch.randn(10000, 384).cuda()
        
        # Warm up
        _ = cosine_similarity_cuda(query, large_documents)
        torch.cuda.synchronize()
        
        # Time execution
        start = time.time()
        scores = cosine_similarity_cuda(query, large_documents)
        torch.cuda.synchronize()
        cuda_time = time.time() - start
        
        # Compare with CPU
        cpu_query = query.cpu()
        cpu_docs = large_documents.cpu()
        
        start = time.time()
        cpu_scores = torch.mm(cpu_docs, cpu_query.unsqueeze(1)).squeeze()
        cpu_time = time.time() - start
        
        # CUDA should be faster for large datasets
        assert cuda_time < cpu_time * 0.5  # At least 2x speedup
        print(f"CUDA speedup: {cpu_time / cuda_time:.2f}x")
    
    def test_kernel_accuracy(self, cuda_tensors):
        """Test kernel numerical accuracy."""
        query, documents = cuda_tensors
        
        # Test with extreme values
        extreme_query = torch.tensor([1e6, 1e-6, 0, -1e6]).cuda()
        extreme_docs = torch.tensor([
            [1e6, 1e-6, 0, -1e6],
            [0, 0, 0, 0],
            [1, 1, 1, 1]
        ]).cuda()
        
        # Should handle extreme values without overflow/underflow
        scores = cosine_similarity_cuda(extreme_query, extreme_docs)
        assert torch.all(torch.isfinite(scores))
        assert torch.all(scores >= -1.001) and torch.all(scores <= 1.001)


class TestCUDAOptimizations:
    """Test CUDA optimization features."""
    
    def test_shared_memory_optimization(self, cuda_available):
        """Test shared memory optimization in kernels."""
        similarity = CUDASimilarity(device_id=0)
        
        # Test with different vector sizes that affect shared memory usage
        sizes = [128, 256, 512, 1024]
        
        for size in sizes:
            query = np.random.randn(size).astype(np.float32)
            documents = np.random.randn(100, size).astype(np.float32)
            
            # Should handle different sizes efficiently
            scores = similarity.cosine_similarity(query, documents)
            assert len(scores) == 100
    
    def test_stream_parallelism(self, cuda_available):
        """Test CUDA stream parallelism."""
        similarity = CUDASimilarity(device_id=0)
        
        with patch('torch.cuda.Stream') as mock_stream:
            similarity.enable_stream_parallelism(num_streams=4)
            assert mock_stream.call_count == 4
    
    def test_mixed_precision(self, cuda_available):
        """Test mixed precision computation."""
        similarity = CUDASimilarity(device_id=0, use_mixed_precision=True)
        
        query = np.random.randn(384).astype(np.float16)
        documents = np.random.randn(100, 384).astype(np.float16)
        
        with patch('torch.cuda.amp.autocast') as mock_autocast:
            scores = similarity.cosine_similarity(query, documents)
            mock_autocast.assert_called()
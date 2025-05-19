import pytest
import time
import torch
import numpy as np
from aiq.cuda_kernels.cuda_similarity import CUDASimilarityCalculator
from aiq.neural.advanced_architectures import FlashAttention, MixtureOfExperts


class TestPerformance:
    """Performance tests for GPU acceleration"""
    
    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_similarity_performance(self, benchmark):
        """Benchmark similarity computation"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        calc = CUDASimilarityCalculator()
        size = 1000
        embeddings = torch.randn(size, 768).cuda()
        
        result = benchmark(calc.cosine_similarity_cuda, embeddings, embeddings)
        assert result.shape == (size, size)
    
    @pytest.mark.gpu
    @pytest.mark.benchmark
    def test_attention_performance(self, benchmark):
        """Benchmark Flash Attention"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        attention = FlashAttention(dim=768).cuda()
        batch_size = 8
        seq_len = 512
        
        q = torch.randn(batch_size, seq_len, 768).cuda()
        k = torch.randn(batch_size, seq_len, 768).cuda()
        v = torch.randn(batch_size, seq_len, 768).cuda()
        
        result = benchmark(attention, q, k, v)
        assert result.shape == (batch_size, seq_len, 768)
    
    @pytest.mark.gpu
    @pytest.mark.benchmark  
    def test_moe_performance(self, benchmark):
        """Benchmark Mixture of Experts"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        moe = MixtureOfExperts(input_dim=768, num_experts=8).cuda()
        batch_size = 32
        x = torch.randn(batch_size, 768).cuda()
        
        result = benchmark(moe, x)
        assert result.shape == (batch_size, 768)

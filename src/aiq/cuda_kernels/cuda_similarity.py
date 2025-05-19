# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
import numpy as np
from typing import Optional, Union
import ctypes
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class CUDASimilarity:
    """
    Python wrapper for CUDA similarity kernels
    """
    def __init__(self):
        self.cuda_lib = None
        self._load_cuda_library()
    
    def _load_cuda_library(self):
        """Load the compiled CUDA library"""
        try:
            # Find the compiled .so file
            lib_path = Path(__file__).parent / "libsimilarity_kernels.so"
            if not lib_path.exists():
                logger.warning(f"CUDA library not found at {lib_path}")
                return
            
            self.cuda_lib = ctypes.CDLL(str(lib_path))
            
            # Define function signatures
            self.cuda_lib.compute_similarities.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # query
                ctypes.POINTER(ctypes.c_float),  # corpus
                ctypes.POINTER(ctypes.c_float),  # similarities
                ctypes.c_int,                    # query_dim
                ctypes.c_int                     # corpus_size
            ]
            
            self.cuda_lib.compute_similarities_fp16.argtypes = [
                ctypes.c_void_p,  # query (half*)
                ctypes.c_void_p,  # corpus (half*)
                ctypes.c_void_p,  # similarities (half*)
                ctypes.c_int,     # query_dim
                ctypes.c_int      # corpus_size
            ]
            
            self.cuda_lib.compute_batch_similarities.argtypes = [
                ctypes.POINTER(ctypes.c_float),  # queries
                ctypes.POINTER(ctypes.c_float),  # corpus
                ctypes.POINTER(ctypes.c_float),  # similarities
                ctypes.c_int,                    # num_queries
                ctypes.c_int,                    # query_dim
                ctypes.c_int                     # corpus_size
            ]
            
            logger.info("CUDA similarity library loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load CUDA library: {e}")
    
    def cosine_similarity(
        self,
        query: Union[torch.Tensor, np.ndarray],
        corpus: Union[torch.Tensor, np.ndarray],
        use_fp16: bool = False
    ) -> torch.Tensor:
        """
        Compute cosine similarity between query and corpus using CUDA
        
        Args:
            query: Query vector (1D)
            corpus: Corpus matrix (2D)
            use_fp16: Whether to use FP16 computation
        
        Returns:
            Similarity scores
        """
        if self.cuda_lib is None:
            # Fallback to PyTorch implementation
            return self._pytorch_cosine_similarity(query, corpus)
        
        # Convert to tensors if needed
        if isinstance(query, np.ndarray):
            query = torch.from_numpy(query)
        if isinstance(corpus, np.ndarray):
            corpus = torch.from_numpy(corpus)
        
        # Ensure CUDA tensors
        query = query.cuda()
        corpus = corpus.cuda()
        
        # Ensure correct shapes
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if corpus.dim() == 1:
            corpus = corpus.unsqueeze(0)
        
        batch_size, dim = corpus.shape
        similarities = torch.zeros(batch_size, device='cuda')
        
        if use_fp16:
            # Convert to FP16
            query = query.half()
            corpus = corpus.half()
            similarities = similarities.half()
            
            # Call CUDA kernel
            self.cuda_lib.compute_similarities_fp16(
                query.data_ptr(),
                corpus.data_ptr(),
                similarities.data_ptr(),
                dim,
                batch_size
            )
        else:
            # Call CUDA kernel
            self.cuda_lib.compute_similarities(
                query.data_ptr(),
                corpus.data_ptr(),
                similarities.data_ptr(),
                dim,
                batch_size
            )
        
        return similarities
    
    def batch_cosine_similarity(
        self,
        queries: Union[torch.Tensor, np.ndarray],
        corpus: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Compute batch cosine similarity using CUDA
        
        Args:
            queries: Query matrix (2D)
            corpus: Corpus matrix (2D)
        
        Returns:
            Similarity matrix
        """
        if self.cuda_lib is None:
            # Fallback to PyTorch implementation
            return self._pytorch_batch_cosine_similarity(queries, corpus)
        
        # Convert to tensors if needed
        if isinstance(queries, np.ndarray):
            queries = torch.from_numpy(queries)
        if isinstance(corpus, np.ndarray):
            corpus = torch.from_numpy(corpus)
        
        # Ensure CUDA tensors
        queries = queries.cuda()
        corpus = corpus.cuda()
        
        num_queries, dim = queries.shape
        corpus_size, _ = corpus.shape
        
        similarities = torch.zeros(num_queries, corpus_size, device='cuda')
        
        # Call CUDA kernel
        self.cuda_lib.compute_batch_similarities(
            queries.data_ptr(),
            corpus.data_ptr(),
            similarities.data_ptr(),
            num_queries,
            dim,
            corpus_size
        )
        
        return similarities
    
    def _pytorch_cosine_similarity(
        self,
        query: torch.Tensor,
        corpus: torch.Tensor
    ) -> torch.Tensor:
        """Fallback PyTorch implementation"""
        query = query.cuda()
        corpus = corpus.cuda()
        
        # Normalize
        query = torch.nn.functional.normalize(query, dim=-1)
        corpus = torch.nn.functional.normalize(corpus, dim=-1)
        
        # Compute similarity
        return torch.matmul(corpus, query.T).squeeze()
    
    def _pytorch_batch_cosine_similarity(
        self,
        queries: torch.Tensor,
        corpus: torch.Tensor
    ) -> torch.Tensor:
        """Fallback PyTorch batch implementation"""
        queries = queries.cuda()
        corpus = corpus.cuda()
        
        # Normalize
        queries = torch.nn.functional.normalize(queries, dim=-1)
        corpus = torch.nn.functional.normalize(corpus, dim=-1)
        
        # Compute similarity
        return torch.matmul(queries, corpus.T)

# Global instance
_cuda_similarity = None

def get_cuda_similarity() -> CUDASimilarity:
    """Get global CUDA similarity instance"""
    global _cuda_similarity
    if _cuda_similarity is None:
        _cuda_similarity = CUDASimilarity()
    return _cuda_similarity

# Convenience functions
def cosine_similarity(
    query: Union[torch.Tensor, np.ndarray],
    corpus: Union[torch.Tensor, np.ndarray],
    use_fp16: bool = False
) -> torch.Tensor:
    """Compute cosine similarity using CUDA acceleration"""
    return get_cuda_similarity().cosine_similarity(query, corpus, use_fp16)

def batch_cosine_similarity(
    queries: Union[torch.Tensor, np.ndarray],
    corpus: Union[torch.Tensor, np.ndarray]
) -> torch.Tensor:
    """Compute batch cosine similarity using CUDA acceleration"""
    return get_cuda_similarity().batch_cosine_similarity(queries, corpus)
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit CUDA Kernels

This module provides CUDA-accelerated operations for high-performance
computing on NVIDIA GPUs.
"""

from aiq.cuda_kernels.cuda_similarity import (
    cosine_similarity,
    batch_cosine_similarity,
    CUDASimilarity,
    get_cuda_similarity
)

__all__ = [
    'cosine_similarity',
    'batch_cosine_similarity',
    'CUDASimilarity',
    'get_cuda_similarity'
]
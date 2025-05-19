# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
AIQ Toolkit Hardware Optimization

This module provides hardware-specific optimizations for NVIDIA GPUs,
including Tensor Core utilization and resource prediction.
"""

from aiq.hardware.tensor_core_optimizer import (
    TensorCoreOptimizer,
    ResourcePredictor,
    OptimizationConfig,
    ResourceRequirements,
    GPUArchitecture
)

__all__ = [
    'TensorCoreOptimizer',
    'ResourcePredictor',
    'OptimizationConfig',
    'ResourceRequirements',
    'GPUArchitecture'
]
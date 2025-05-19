# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Setup script for compiling CUDA kernels
"""

import subprocess
import sys
from pathlib import Path

def compile_cuda_kernels():
    """Compile CUDA kernels into shared library"""
    cuda_source = Path(__file__).parent / "similarity_kernels.cu"
    output_lib = Path(__file__).parent / "libsimilarity_kernels.so"
    
    # Compilation command
    compile_cmd = [
        "nvcc",
        "-shared",
        "-fPIC",
        "-O3",
        "-arch=sm_70",  # Minimum for Volta (V100)
        "-gencode=arch=compute_75,code=sm_75",  # Turing
        "-gencode=arch=compute_80,code=sm_80",  # Ampere
        "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30 series)
        "-gencode=arch=compute_90,code=sm_90",  # Hopper (H100)
        "-o", str(output_lib),
        str(cuda_source)
    ]
    
    try:
        # Check if nvcc is available
        subprocess.run(["nvcc", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("NVCC compiler not found. CUDA kernels will not be compiled.")
        print("Install CUDA toolkit to enable GPU acceleration.")
        return False
    
    try:
        print(f"Compiling CUDA kernels: {cuda_source}")
        subprocess.run(compile_cmd, check=True)
        print(f"Successfully compiled to: {output_lib}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to compile CUDA kernels: {e}")
        return False

if __name__ == "__main__":
    success = compile_cuda_kernels()
    sys.exit(0 if success else 1)
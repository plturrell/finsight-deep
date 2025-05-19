// SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math.h>

// Constants for kernel configuration
#define TILE_SIZE 16
#define VECTOR_SIZE 4
#define WARP_SIZE 32

// Helper function for warp-level reduction
__device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Fast cosine similarity kernel using shared memory and warp primitives
__global__ void fast_cosine_similarity(
    const float* __restrict__ query,
    const float* __restrict__ corpus,
    float* __restrict__ similarities,
    int query_dim,
    int corpus_size,
    int dim_padded
) {
    extern __shared__ float shared_data[];
    
    // Shared memory layout
    float* shared_query = shared_data;
    float* shared_corpus = &shared_data[dim_padded];
    
    int tid = threadIdx.x;
    int corpus_idx = blockIdx.x;
    
    if (corpus_idx >= corpus_size) return;
    
    // Load query into shared memory (coalesced)
    for (int i = tid; i < query_dim; i += blockDim.x) {
        shared_query[i] = query[i];
    }
    
    // Zero padding if needed
    for (int i = query_dim + tid; i < dim_padded; i += blockDim.x) {
        shared_query[i] = 0.0f;
    }
    
    __syncthreads();
    
    // Process corpus vector
    float dot_product = 0.0f;
    float query_norm_sq = 0.0f;
    float corpus_norm_sq = 0.0f;
    
    // Vectorized memory access
    for (int i = tid * VECTOR_SIZE; i < dim_padded; i += blockDim.x * VECTOR_SIZE) {
        float4 q_vec, c_vec;
        
        if (i + VECTOR_SIZE <= query_dim) {
            // Load vectorized data
            q_vec = *reinterpret_cast<const float4*>(&shared_query[i]);
            c_vec = *reinterpret_cast<const float4*>(&corpus[corpus_idx * query_dim + i]);
            
            // Compute dot product and norms
            dot_product += q_vec.x * c_vec.x + q_vec.y * c_vec.y + 
                          q_vec.z * c_vec.z + q_vec.w * c_vec.w;
            
            query_norm_sq += q_vec.x * q_vec.x + q_vec.y * q_vec.y + 
                            q_vec.z * q_vec.z + q_vec.w * q_vec.w;
            
            corpus_norm_sq += c_vec.x * c_vec.x + c_vec.y * c_vec.y + 
                             c_vec.z * c_vec.z + c_vec.w * c_vec.w;
        }
    }
    
    // Warp-level reduction
    dot_product = warpReduceSum(dot_product);
    query_norm_sq = warpReduceSum(query_norm_sq);
    corpus_norm_sq = warpReduceSum(corpus_norm_sq);
    
    // Write result (first thread in warp)
    if (tid % WARP_SIZE == 0) {
        atomicAdd(&similarities[corpus_idx], 
                  dot_product / (sqrtf(query_norm_sq) * sqrtf(corpus_norm_sq)));
    }
}

// FP16 version for newer GPUs
__global__ void fast_cosine_similarity_fp16(
    const half* __restrict__ query,
    const half* __restrict__ corpus,
    half* __restrict__ similarities,
    int query_dim,
    int corpus_size
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= corpus_size) return;
    
    // Use half2 for vectorized operations on Tensor Cores
    float dot_product = 0.0f;
    float query_norm = 0.0f;
    float corpus_norm = 0.0f;
    
    for (int i = 0; i < query_dim; i += 2) {
        half2 q = *reinterpret_cast<const half2*>(&query[i]);
        half2 c = *reinterpret_cast<const half2*>(&corpus[tid * query_dim + i]);
        
        float2 q_f = __half22float2(q);
        float2 c_f = __half22float2(c);
        
        dot_product += q_f.x * c_f.x + q_f.y * c_f.y;
        query_norm += q_f.x * q_f.x + q_f.y * q_f.y;
        corpus_norm += c_f.x * c_f.x + c_f.y * c_f.y;
    }
    
    similarities[tid] = __float2half(dot_product / (sqrtf(query_norm) * sqrtf(corpus_norm)));
}

// Batch cosine similarity with matrix multiplication (uses Tensor Cores)
__global__ void batch_cosine_similarity_gemm(
    const float* __restrict__ queries,
    const float* __restrict__ corpus,
    float* __restrict__ similarities,
    int num_queries,
    int query_dim,
    int corpus_size
) {
    // This kernel assumes queries and corpus are pre-normalized
    // Uses GEMM to compute Q * C^T efficiently
    
    int query_idx = blockIdx.y;
    int corpus_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (query_idx >= num_queries || corpus_idx >= corpus_size) return;
    
    float dot_product = 0.0f;
    
    // Compute dot product
    for (int i = 0; i < query_dim; i++) {
        dot_product += queries[query_idx * query_dim + i] * 
                      corpus[corpus_idx * query_dim + i];
    }
    
    similarities[query_idx * corpus_size + corpus_idx] = dot_product;
}

// Python interface functions
extern "C" {
    
    void compute_similarities(
        float* query,
        float* corpus,
        float* similarities,
        int query_dim,
        int corpus_size
    ) {
        // Pad dimension to multiple of VECTOR_SIZE
        int dim_padded = ((query_dim + VECTOR_SIZE - 1) / VECTOR_SIZE) * VECTOR_SIZE;
        
        // Calculate grid and block dimensions
        int threads_per_block = 256;
        int blocks = corpus_size;
        
        // Shared memory size
        size_t shared_mem_size = 2 * dim_padded * sizeof(float);
        
        // Launch kernel
        fast_cosine_similarity<<<blocks, threads_per_block, shared_mem_size>>>(
            query, corpus, similarities, query_dim, corpus_size, dim_padded
        );
        
        cudaDeviceSynchronize();
    }
    
    void compute_similarities_fp16(
        __half* query,
        __half* corpus,
        __half* similarities,
        int query_dim,
        int corpus_size
    ) {
        int threads_per_block = 256;
        int blocks = (corpus_size + threads_per_block - 1) / threads_per_block;
        
        fast_cosine_similarity_fp16<<<blocks, threads_per_block>>>(
            query, corpus, similarities, query_dim, corpus_size
        );
        
        cudaDeviceSynchronize();
    }
    
    void compute_batch_similarities(
        float* queries,
        float* corpus,
        float* similarities,
        int num_queries,
        int query_dim,
        int corpus_size
    ) {
        dim3 threads_per_block(32, 1);
        dim3 blocks((corpus_size + 31) / 32, num_queries);
        
        batch_cosine_similarity_gemm<<<blocks, threads_per_block>>>(
            queries, corpus, similarities, num_queries, query_dim, corpus_size
        );
        
        cudaDeviceSynchronize();
    }
}
# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Advanced Neural Architectures for AIQToolkit
Implements state-of-the-art neural models with custom optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
from dataclasses import dataclass
from einops import rearrange, repeat
from torch.cuda.amp import autocast

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer
from aiq.cuda_kernels.cuda_similarity import get_cuda_similarity


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    dim: int
    num_heads: int
    dim_head: int = 64
    dropout: float = 0.1
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True
    use_xformers: bool = False
    max_seq_length: int = 8192


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE) for improved position encoding"""
    
    def __init__(self, dim: int, max_seq_length: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_seq_length).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        self.register_buffer('sin', freqs.sin())
        self.register_buffer('cos', freqs.cos())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, device = x.shape[1], x.device
        sin, cos = self.sin[:n].to(device), self.cos[:n].to(device)
        rot_dim = sin.shape[1] * 2
        x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
        
        x1, x2 = x_rot.chunk(2, dim=-1)
        x_rot = torch.cat((-x2, x1), dim=-1)
        
        x_out = (x_rot * sin.unsqueeze(0)) + (x[..., :rot_dim] * cos.unsqueeze(0))
        return torch.cat((x_out, x_pass), dim=-1)


class FlashAttention(nn.Module):
    """Flash Attention implementation for memory-efficient attention"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.scale = config.dim_head ** -0.5
        
        inner_dim = config.dim_head * config.num_heads
        self.to_qkv = nn.Linear(config.dim, inner_dim * 3, bias=False)
        self.to_out = nn.Linear(inner_dim, config.dim)
        
        if config.use_rotary_embeddings:
            self.rotary_emb = RotaryEmbedding(config.dim_head)
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Optimize for Tensor Cores
        self.optimizer = TensorCoreOptimizer()
        self._optimize_layers()
    
    def _optimize_layers(self):
        """Optimize layers for Tensor Core acceleration"""
        self.to_qkv = self.optimizer.optimize_model(self.to_qkv)
        self.to_out = self.optimizer.optimize_model(self.to_out)
    
    @autocast()
    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        b, n, d = x.shape
        h = self.config.num_heads
        
        # Linear projections
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        
        # Cross-attention if context provided
        if context is not None:
            context_qkv = self.to_qkv(context).chunk(3, dim=-1)
            _, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), context_qkv)
        
        # Apply rotary embeddings
        if self.config.use_rotary_embeddings:
            q = self.rotary_emb(q)
            k = self.rotary_emb(k)
        
        # Flash attention computation
        if self.config.use_flash_attention and x.is_cuda:
            # Use optimized flash attention kernel
            attn_output = self._flash_attention(q, k, v, mask)
        else:
            # Standard attention
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            
            if mask is not None:
                dots = dots.masked_fill(mask == 0, -1e9)
            
            attn = F.softmax(dots, dim=-1)
            attn = self.dropout(attn)
            attn_output = torch.matmul(attn, v)
        
        # Reshape and project output
        out = rearrange(attn_output, 'b h n d -> b n (h d)')
        return self.to_out(out)
    
    def _flash_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Optimized flash attention implementation"""
        # This would call custom CUDA kernels in production
        # For now, using standard PyTorch with memory optimizations
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=True
        ):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer for sparse computation"""
    
    def __init__(
        self,
        dim: int,
        num_experts: int = 8,
        expert_capacity: float = 1.25,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.expert_capacity = expert_capacity
        
        # Router network
        self.router = nn.Linear(dim, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])
        
        # Load balancing loss
        self.load_balancing_loss = 0.0
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        batch_size, seq_len, dim = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, dim)
        
        # Get routing scores
        router_logits = self.router(x_flat)
        routing_weights = F.softmax(router_logits, dim=1)
        
        # Select top experts
        expert_weights, expert_indices = torch.topk(
            routing_weights, k=2, dim=1
        )
        expert_weights = F.softmax(expert_weights, dim=1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get samples for this expert
            expert_mask = (expert_indices == expert_idx).any(dim=1)
            expert_input = x_flat[expert_mask]
            
            if expert_input.shape[0] > 0:
                # Apply expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Weight by routing scores
                weights = expert_weights[expert_mask]
                weights = weights[expert_indices[expert_mask] == expert_idx]
                
                # Add to output
                output[expert_mask] += expert_output * weights.unsqueeze(1)
        
        # Calculate load balancing loss
        expert_load = torch.zeros(self.num_experts, device=x.device)
        for idx in expert_indices.view(-1):
            expert_load[idx] += 1.0
        expert_load = expert_load / expert_load.sum()
        
        ideal_load = 1.0 / self.num_experts
        self.load_balancing_loss = ((expert_load - ideal_load) ** 2).sum()
        
        # Reshape output
        output = output.view(batch_size, seq_len, dim)
        
        return output, {
            "load_balancing_loss": self.load_balancing_loss.item(),
            "expert_distribution": expert_load.cpu().numpy()
        }


class NeuralMemoryBank(nn.Module):
    """Neural memory bank with differentiable read/write operations"""
    
    def __init__(
        self,
        memory_size: int = 1024,
        key_dim: int = 512,
        value_dim: int = 512,
        num_heads: int = 8,
        temperature: float = 1.0
    ):
        super().__init__()
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.temperature = temperature
        
        # Initialize memory
        self.keys = nn.Parameter(torch.randn(memory_size, key_dim))
        self.values = nn.Parameter(torch.randn(memory_size, value_dim))
        
        # Query projections
        self.query_proj = nn.Linear(key_dim, key_dim * num_heads)
        self.value_proj = nn.Linear(value_dim, value_dim)
        
        # Memory update network
        self.update_gate = nn.Sequential(
            nn.Linear(key_dim + value_dim, key_dim),
            nn.Sigmoid()
        )
    
    def read(self, queries: torch.Tensor) -> torch.Tensor:
        """Read from memory using attention mechanism"""
        batch_size, seq_len, _ = queries.shape
        
        # Project queries
        queries = self.query_proj(queries)
        queries = rearrange(
            queries, 'b n (h d) -> b h n d',
            h=self.num_heads
        )
        
        # Compute attention scores
        keys_expanded = repeat(
            self.keys, 'm d -> b h m d',
            b=batch_size, h=self.num_heads
        )
        
        scores = torch.einsum('bhnd,bhmd->bhnm', queries, keys_expanded)
        scores = scores / (self.key_dim ** 0.5) / self.temperature
        
        # Apply softmax and read values
        attn_weights = F.softmax(scores, dim=-1)
        values_expanded = repeat(
            self.values, 'm d -> b h m d',
            b=batch_size, h=self.num_heads
        )
        
        read_vectors = torch.einsum('bhnm,bhmd->bhnd', attn_weights, values_expanded)
        read_vectors = rearrange(read_vectors, 'b h n d -> b n (h d)')
        
        return self.value_proj(read_vectors)
    
    def write(self, keys: torch.Tensor, values: torch.Tensor):
        """Write to memory with gating mechanism"""
        # Compute similarity to existing keys
        similarity = F.cosine_similarity(
            keys.unsqueeze(1),
            self.keys.unsqueeze(0),
            dim=-1
        )
        
        # Find most similar slots
        _, indices = similarity.max(dim=1)
        
        # Update memory
        for i, (key, value, idx) in enumerate(zip(keys, values, indices)):
            # Compute update gate
            combined = torch.cat([key, self.values[idx]], dim=-1)
            gate = self.update_gate(combined)
            
            # Update key and value
            self.keys.data[idx] = gate * key + (1 - gate) * self.keys[idx]
            self.values.data[idx] = gate * value + (1 - gate) * self.values[idx]


class HybridNeuralSymbolicLayer(nn.Module):
    """Hybrid layer combining neural and symbolic processing"""
    
    def __init__(
        self,
        neural_dim: int,
        symbolic_dim: int,
        num_rules: int = 100,
        dropout: float = 0.1
    ):
        super().__init__()
        self.neural_dim = neural_dim
        self.symbolic_dim = symbolic_dim
        self.num_rules = num_rules
        
        # Neural processing
        self.neural_encoder = nn.Sequential(
            nn.Linear(neural_dim, neural_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(neural_dim * 2, neural_dim)
        )
        
        # Symbolic processing
        self.rule_embeddings = nn.Parameter(
            torch.randn(num_rules, symbolic_dim)
        )
        self.rule_selector = nn.Linear(neural_dim, num_rules)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(neural_dim + symbolic_dim, neural_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        symbolic_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Neural processing
        neural_out = self.neural_encoder(x)
        
        # Select relevant rules
        rule_scores = self.rule_selector(neural_out)
        rule_weights = F.softmax(rule_scores, dim=-1)
        
        # Aggregate rule embeddings
        symbolic_out = torch.matmul(rule_weights, self.rule_embeddings)
        
        # Combine neural and symbolic
        if symbolic_context is not None:
            symbolic_out = symbolic_out + symbolic_context
        
        combined = torch.cat([neural_out, symbolic_out], dim=-1)
        return self.fusion(combined)


class AdaptiveComputeLayer(nn.Module):
    """Layer with adaptive computation time"""
    
    def __init__(
        self,
        dim: int,
        max_steps: int = 10,
        threshold: float = 0.95
    ):
        super().__init__()
        self.dim = dim
        self.max_steps = max_steps
        self.threshold = threshold
        
        # Step function
        self.step_fn = nn.GRUCell(dim, dim)
        
        # Halting probability
        self.halt_prob = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize
        state = x
        halting_prob = torch.zeros(batch_size, device=device)
        remainders = torch.ones(batch_size, device=device)
        n_steps = torch.zeros(batch_size, device=device)
        
        # Adaptive computation
        for step in range(self.max_steps):
            # Compute step
            state = self.step_fn(x, state)
            
            # Compute halting probability
            p = self.halt_prob(state).squeeze(-1)
            
            # Update halting probability
            still_running = (halting_prob < self.threshold).float()
            new_halted = (halting_prob + p * remainders > self.threshold).float()
            
            # Update accumulators
            halting_prob += p * remainders * still_running
            remainders *= (1 - p) * still_running
            n_steps += still_running
            
            # Check if all samples have halted
            if (halting_prob >= self.threshold).all():
                break
        
        return state, {
            "average_steps": n_steps.mean().item(),
            "halting_distribution": halting_prob.cpu().numpy()
        }
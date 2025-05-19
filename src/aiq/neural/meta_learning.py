# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Meta-Learning and Few-Shot Learning Implementations
Includes MAML, Prototypical Networks, and Reptile
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import copy
from collections import OrderedDict
import logging

from aiq.neural.advanced_architectures import (
    FlashAttention, AttentionConfig, NeuralMemoryBank
)
from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


logger = logging.getLogger(__name__)


@dataclass
class MetaLearningConfig:
    """Configuration for meta-learning algorithms"""
    algorithm: str = "maml"  # maml, reptile, protonet
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    num_inner_steps: int = 5
    num_tasks_per_batch: int = 8
    k_shot: int = 5
    n_way: int = 5
    query_size: int = 15
    meta_batch_size: int = 4
    use_cuda: bool = True
    first_order: bool = False


class MetaLearner(nn.Module):
    """Base class for meta-learning algorithms"""
    
    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__()
        self.base_model = base_model
        self.config = config
        self.device = torch.device('cuda' if config.use_cuda and torch.cuda.is_available() else 'cpu')
        
        # Optimize for GPU
        self.tensor_optimizer = TensorCoreOptimizer()
        self.base_model = self.tensor_optimizer.optimize_model(self.base_model)
        self.base_model = self.base_model.to(self.device)
    
    def clone_model(self) -> nn.Module:
        """Create a copy of the base model"""
        return copy.deepcopy(self.base_model)
    
    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute task loss"""
        return F.cross_entropy(logits, targets)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through base model"""
        return self.base_model(x)


class MAML(MetaLearner):
    """Model-Agnostic Meta-Learning"""
    
    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(base_model, config)
        
        # Meta-optimizer
        self.meta_optimizer = torch.optim.Adam(
            self.base_model.parameters(),
            lr=config.outer_lr
        )
    
    def inner_loop(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int
    ) -> OrderedDict:
        """Inner loop adaptation"""
        # Clone model for task-specific adaptation
        adapted_params = OrderedDict(self.base_model.named_parameters())
        
        for step in range(num_steps):
            # Forward pass with current parameters
            logits = self.functional_forward(support_x, adapted_params)
            loss = self.compute_loss(logits, support_y)
            
            # Compute gradients
            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.config.first_order
            )
            
            # Update parameters
            adapted_params = OrderedDict()
            for (name, param), grad in zip(self.base_model.named_parameters(), grads):
                adapted_params[name] = param - self.config.inner_lr * grad
        
        return adapted_params
    
    def functional_forward(
        self,
        x: torch.Tensor,
        params: OrderedDict
    ) -> torch.Tensor:
        """Forward pass with given parameters"""
        # This is a simplified version - in practice, you'd need to handle
        # different layer types appropriately
        x = x.view(x.size(0), -1)
        
        for name, param in params.items():
            if 'weight' in name and 'bn' not in name:
                if 'fc' in name or 'linear' in name:
                    x = F.linear(x, param, params.get(name.replace('weight', 'bias')))
                    x = F.relu(x)
            elif name == list(params.keys())[-2]:  # Last linear layer
                x = F.linear(x, param, params.get(name.replace('weight', 'bias')))
        
        return x
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Single meta-training step"""
        meta_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Inner loop adaptation
            adapted_params = self.inner_loop(support_x, support_y, self.config.num_inner_steps)
            
            # Compute loss on query set
            query_logits = self.functional_forward(query_x, adapted_params)
            query_loss = self.compute_loss(query_logits, query_y)
            
            meta_loss += query_loss
        
        # Meta-optimization step
        meta_loss = meta_loss / len(tasks)
        
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()


class Reptile(MetaLearner):
    """Reptile meta-learning algorithm"""
    
    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(base_model, config)
        
        # Reptile doesn't need a meta-optimizer
        self.meta_lr = config.outer_lr
    
    def reptile_update(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        num_steps: int
    ) -> OrderedDict:
        """Reptile inner loop"""
        # Clone model for task adaptation
        task_model = self.clone_model()
        task_optimizer = torch.optim.SGD(task_model.parameters(), lr=self.config.inner_lr)
        
        # Train on support set
        for _ in range(num_steps):
            logits = task_model(support_x)
            loss = self.compute_loss(logits, support_y)
            
            task_optimizer.zero_grad()
            loss.backward()
            task_optimizer.step()
        
        return OrderedDict(task_model.named_parameters())
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Single meta-training step"""
        meta_grads = []
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Get adapted parameters
            adapted_params = self.reptile_update(support_x, support_y, self.config.num_inner_steps)
            
            # Compute meta-gradient (difference between adapted and original params)
            task_grads = []
            for (name, param), adapted_param in zip(self.base_model.named_parameters(), adapted_params.values()):
                task_grads.append(param - adapted_param)
            
            meta_grads.append(task_grads)
            
            # Evaluate on query set for logging
            with torch.no_grad():
                query_logits = self.base_model(query_x)
                query_loss = self.compute_loss(query_logits, query_y)
                total_loss += query_loss.item()
        
        # Average meta-gradients and update model
        avg_grads = []
        for i in range(len(task_grads)):
            avg_grad = sum(grads[i] for grads in meta_grads) / len(meta_grads)
            avg_grads.append(avg_grad)
        
        # Update model parameters
        with torch.no_grad():
            for param, grad in zip(self.base_model.parameters(), avg_grads):
                param.add_(grad, alpha=-self.meta_lr)
        
        return total_loss / len(tasks)


class PrototypicalNetwork(MetaLearner):
    """Prototypical Networks for few-shot learning"""
    
    def __init__(self, encoder: nn.Module, config: MetaLearningConfig):
        super().__init__(encoder, config)
        
        # Add projection head
        self.projection = nn.Sequential(
            nn.Linear(self.get_feature_dim(), 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        self.optimizer = torch.optim.Adam(
            list(self.base_model.parameters()) + list(self.projection.parameters()),
            lr=config.outer_lr
        )
    
    def get_feature_dim(self) -> int:
        """Get output dimension of encoder"""
        # This is a placeholder - implement based on your encoder architecture
        return 512
    
    def compute_prototypes(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """Compute class prototypes from support set"""
        # Encode support examples
        support_features = self.encode(support_x)
        
        # Compute prototypes for each class
        prototypes = []
        for class_idx in range(n_way):
            class_mask = support_y == class_idx
            class_features = support_features[class_mask]
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        return torch.stack(prototypes)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature space"""
        features = self.base_model(x)
        return self.projection(features)
    
    def forward(
        self,
        query_x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """Forward pass for prototypical networks"""
        # Compute prototypes
        prototypes = self.compute_prototypes(support_x, support_y, n_way)
        
        # Encode queries
        query_features = self.encode(query_x)
        
        # Compute distances to prototypes
        distances = torch.cdist(query_features, prototypes)
        
        # Return negative distances as logits
        return -distances
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Single meta-training step"""
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Forward pass
            logits = self.forward(query_x, support_x, support_y, self.config.n_way)
            loss = self.compute_loss(logits, query_y)
            
            total_loss += loss
        
        # Optimize
        avg_loss = total_loss / len(tasks)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()


class MatchingNetwork(MetaLearner):
    """Matching Networks with attention mechanism"""
    
    def __init__(self, encoder: nn.Module, config: MetaLearningConfig):
        super().__init__(encoder, config)
        
        feature_dim = self.get_feature_dim()
        
        # Attention components
        self.attention_config = AttentionConfig(
            dim=feature_dim,
            num_heads=8,
            dim_head=64
        )
        self.attention = FlashAttention(self.attention_config)
        
        # Bi-directional LSTM for full context embeddings
        self.context_lstm = nn.LSTM(
            feature_dim,
            feature_dim // 2,
            bidirectional=True,
            batch_first=True
        )
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.outer_lr
        )
    
    def get_feature_dim(self) -> int:
        """Get output dimension of encoder"""
        return 512
    
    def encode_support(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode support set with full context"""
        # Basic encoding
        support_features = self.base_model(support_x)
        
        # Full context encoding with LSTM
        lstm_out, _ = self.context_lstm(support_features.unsqueeze(0))
        support_features_context = lstm_out.squeeze(0)
        
        return support_features_context, support_y
    
    def attend(
        self,
        query_features: torch.Tensor,
        support_features: torch.Tensor,
        support_labels: torch.Tensor
    ) -> torch.Tensor:
        """Attention-based matching"""
        # Compute attention weights
        attention_weights = self.attention(
            query_features.unsqueeze(1),
            context=support_features.unsqueeze(0).expand(query_features.size(0), -1, -1)
        ).squeeze(1)
        
        # Weighted sum over support labels
        n_way = support_labels.max() + 1
        logits = []
        
        for query_idx in range(query_features.size(0)):
            class_scores = []
            for class_idx in range(n_way):
                class_mask = support_labels == class_idx
                class_attention = attention_weights[query_idx, class_mask]
                class_score = class_attention.sum()
                class_scores.append(class_score)
            
            logits.append(torch.stack(class_scores))
        
        return torch.stack(logits)
    
    def forward(
        self,
        query_x: torch.Tensor,
        support_x: torch.Tensor,
        support_y: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass for matching networks"""
        # Encode support and query
        support_features, _ = self.encode_support(support_x, support_y)
        query_features = self.base_model(query_x)
        
        # Attention-based matching
        logits = self.attend(query_features, support_features, support_y)
        
        return logits
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Single meta-training step"""
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Forward pass
            logits = self.forward(query_x, support_x, support_y)
            loss = self.compute_loss(logits, query_y)
            
            total_loss += loss
        
        # Optimize
        avg_loss = total_loss / len(tasks)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()


class MetaLearningWithMemory(MetaLearner):
    """Meta-learning with neural memory augmentation"""
    
    def __init__(self, base_model: nn.Module, config: MetaLearningConfig):
        super().__init__(base_model, config)
        
        feature_dim = self.get_feature_dim()
        
        # Neural memory bank
        self.memory_bank = NeuralMemoryBank(
            memory_size=256,
            key_dim=feature_dim,
            value_dim=feature_dim,
            num_heads=8
        )
        
        # Projection for memory queries
        self.memory_query_proj = nn.Linear(feature_dim, feature_dim)
        
        # Final classifier
        self.classifier = nn.Linear(feature_dim * 2, self.config.n_way)
        
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.outer_lr
        )
    
    def get_feature_dim(self) -> int:
        """Get output dimension of encoder"""
        return 512
    
    def forward(
        self,
        x: torch.Tensor,
        support_set: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Forward pass with memory augmentation"""
        # Encode input
        features = self.base_model(x)
        
        # Update memory with support set if provided
        if support_set is not None:
            support_x, support_y = support_set
            support_features = self.base_model(support_x)
            
            # Write support examples to memory
            for i in range(len(support_x)):
                key = support_features[i]
                value = F.one_hot(support_y[i], self.config.n_way).float()
                self.memory_bank.write(key.unsqueeze(0), value.unsqueeze(0))
        
        # Query memory
        query_features = self.memory_query_proj(features)
        memory_output = self.memory_bank.read(query_features)
        
        # Combine features and memory output
        combined = torch.cat([features, memory_output], dim=-1)
        
        # Classification
        logits = self.classifier(combined)
        
        return logits
    
    def meta_train_step(
        self,
        tasks: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
    ) -> float:
        """Single meta-training step"""
        total_loss = 0.0
        
        for support_x, support_y, query_x, query_y in tasks:
            # Move to device
            support_x = support_x.to(self.device)
            support_y = support_y.to(self.device)
            query_x = query_x.to(self.device)
            query_y = query_y.to(self.device)
            
            # Clear memory for new task
            self.memory_bank.keys.data.normal_(0, 0.01)
            self.memory_bank.values.data.normal_(0, 0.01)
            
            # Forward pass
            logits = self.forward(query_x, support_set=(support_x, support_y))
            loss = self.compute_loss(logits, query_y)
            
            total_loss += loss
        
        # Optimize
        avg_loss = total_loss / len(tasks)
        
        self.optimizer.zero_grad()
        avg_loss.backward()
        self.optimizer.step()
        
        return avg_loss.item()
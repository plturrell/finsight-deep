# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Neural Architecture Search (NAS) with Differentiable and Evolutionary Methods
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import random
from collections import defaultdict
import logging
from tqdm import tqdm
import time

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer
from aiq.neural.advanced_architectures import FlashAttention, AttentionConfig


logger = logging.getLogger(__name__)


@dataclass
class SearchSpace:
    """Define the search space for NAS"""
    layer_types: List[str] = None
    hidden_dims: List[int] = None
    num_heads_options: List[int] = None
    activation_functions: List[str] = None
    dropout_rates: List[float] = None
    kernel_sizes: List[int] = None
    expansion_ratios: List[float] = None
    
    def __post_init__(self):
        if self.layer_types is None:
            self.layer_types = ['linear', 'conv', 'attention', 'moe', 'residual']
        if self.hidden_dims is None:
            self.hidden_dims = [64, 128, 256, 512, 768, 1024]
        if self.num_heads_options is None:
            self.num_heads_options = [4, 8, 12, 16]
        if self.activation_functions is None:
            self.activation_functions = ['relu', 'gelu', 'swish', 'elu']
        if self.dropout_rates is None:
            self.dropout_rates = [0.0, 0.1, 0.2, 0.3]
        if self.kernel_sizes is None:
            self.kernel_sizes = [1, 3, 5, 7]
        if self.expansion_ratios is None:
            self.expansion_ratios = [1.0, 2.0, 4.0]


class SearchableLayer(nn.Module):
    """Base class for searchable layers in NAS"""
    
    def __init__(self, in_dim: int, out_dim: int, search_space: SearchSpace):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.search_space = search_space
        
        # Architecture parameters (alpha)
        self.alpha = nn.Parameter(torch.zeros(len(search_space.layer_types)))
        
        # Candidate operations
        self.ops = nn.ModuleDict()
        self._build_operations()
    
    def _build_operations(self):
        """Build candidate operations"""
        for op_type in self.search_space.layer_types:
            if op_type == 'linear':
                self.ops[op_type] = nn.Linear(self.in_dim, self.out_dim)
            elif op_type == 'conv':
                # 1D convolution for sequential data
                self.ops[op_type] = nn.Conv1d(
                    self.in_dim, self.out_dim, kernel_size=3, padding=1
                )
            elif op_type == 'attention':
                config = AttentionConfig(
                    dim=self.out_dim,
                    num_heads=8,
                    dropout=0.1
                )
                self.ops[op_type] = FlashAttention(config)
            elif op_type == 'residual':
                self.ops[op_type] = nn.Sequential(
                    nn.Linear(self.in_dim, self.out_dim),
                    nn.ReLU(),
                    nn.Linear(self.out_dim, self.out_dim)
                )
            else:
                self.ops[op_type] = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute operation weights
        weights = F.softmax(self.alpha, dim=0)
        
        # Mixed operation
        output = 0
        for i, (op_name, op) in enumerate(self.ops.items()):
            if weights[i] > 0.01:  # Skip near-zero weights
                op_out = op(x)
                # Handle dimension mismatch
                if op_out.shape != x.shape and self.in_dim == self.out_dim:
                    op_out = op_out + x  # Residual connection
                output = output + weights[i] * op_out
        
        return output
    
    def get_selected_op(self) -> str:
        """Get the operation with highest weight"""
        weights = F.softmax(self.alpha, dim=0)
        idx = weights.argmax().item()
        return self.search_space.layer_types[idx]


class SuperNet(nn.Module):
    """Supernet for weight sharing NAS"""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int,
        search_space: SearchSpace
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.search_space = search_space
        
        # Build layers
        self.layers = nn.ModuleList()
        current_dim = input_dim
        
        for i in range(num_layers):
            # Determine layer dimensions
            if i == num_layers - 1:
                next_dim = output_dim
            else:
                next_dim = random.choice(search_space.hidden_dims)
            
            layer = SearchableLayer(current_dim, next_dim, search_space)
            self.layers.append(layer)
            current_dim = next_dim
        
        # Global pooling and final projection
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.final_proj = nn.Linear(current_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling if needed
        if x.dim() == 3:  # [batch, seq, dim]
            x = x.transpose(1, 2)  # [batch, dim, seq]
            x = self.global_pool(x).squeeze(-1)  # [batch, dim]
        
        return self.final_proj(x)
    
    def get_architecture(self) -> List[str]:
        """Get the current architecture"""
        return [layer.get_selected_op() for layer in self.layers]
    
    def architecture_parameters(self) -> List[nn.Parameter]:
        """Get architecture parameters"""
        return [layer.alpha for layer in self.layers]


class DARTS:
    """Differentiable Architecture Search"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        input_dim: int,
        output_dim: int,
        num_layers: int = 8
    ):
        self.search_space = search_space
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # Create supernet
        self.supernet = SuperNet(
            input_dim, output_dim, num_layers, search_space
        )
        
        # Optimizers
        self.model_optimizer = None
        self.arch_optimizer = None
        
        # Best architecture
        self.best_architecture = None
        self.best_performance = float('-inf')
    
    def setup_optimizers(self, lr_model: float = 0.001, lr_arch: float = 0.001):
        """Setup optimizers for model and architecture parameters"""
        # Model parameters
        model_params = []
        for name, param in self.supernet.named_parameters():
            if 'alpha' not in name:
                model_params.append(param)
        
        self.model_optimizer = torch.optim.AdamW(model_params, lr=lr_model)
        
        # Architecture parameters
        self.arch_optimizer = torch.optim.Adam(
            self.supernet.architecture_parameters(), lr=lr_arch
        )
    
    def train_step(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        criterion: nn.Module
    ):
        """Single training step for DARTS"""
        device = next(self.supernet.parameters()).device
        
        # Phase 1: Update architecture parameters
        self.arch_optimizer.zero_grad()
        
        # Get validation batch
        val_batch = next(iter(val_loader))
        val_inputs, val_targets = val_batch
        val_inputs = val_inputs.to(device)
        val_targets = val_targets.to(device)
        
        # Forward pass on validation data
        val_outputs = self.supernet(val_inputs)
        val_loss = criterion(val_outputs, val_targets)
        
        # Backward pass for architecture parameters
        val_loss.backward()
        self.arch_optimizer.step()
        
        # Phase 2: Update model parameters
        self.model_optimizer.zero_grad()
        
        # Get training batch
        train_batch = next(iter(train_loader))
        train_inputs, train_targets = train_batch
        train_inputs = train_inputs.to(device)
        train_targets = train_targets.to(device)
        
        # Forward pass on training data
        train_outputs = self.supernet(train_inputs)
        train_loss = criterion(train_outputs, train_targets)
        
        # Backward pass for model parameters
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.supernet.parameters(), 5.0)
        self.model_optimizer.step()
        
        return train_loss.item(), val_loss.item()
    
    def search(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 50,
        criterion: nn.Module = None
    ):
        """Run architecture search"""
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supernet = self.supernet.to(device)
        
        logger.info("Starting DARTS architecture search...")
        
        for epoch in range(num_epochs):
            train_losses = []
            val_losses = []
            
            # Training epoch
            self.supernet.train()
            for _ in tqdm(range(len(train_loader)), desc=f"Epoch {epoch+1}/{num_epochs}"):
                train_loss, val_loss = self.train_step(
                    train_loader, val_loader, criterion
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
            
            # Evaluate current architecture
            current_arch = self.supernet.get_architecture()
            avg_val_loss = np.mean(val_losses)
            
            if -avg_val_loss > self.best_performance:
                self.best_performance = -avg_val_loss
                self.best_architecture = current_arch
            
            logger.info(
                f"Epoch {epoch+1}: Train Loss: {np.mean(train_losses):.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Architecture: {current_arch}"
            )
        
        return self.best_architecture


class EvolutionaryNAS:
    """Evolutionary Neural Architecture Search"""
    
    def __init__(
        self,
        search_space: SearchSpace,
        population_size: int = 50,
        tournament_size: int = 5,
        mutation_rate: float = 0.1
    ):
        self.search_space = search_space
        self.population_size = population_size
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        
        # Population
        self.population = []
        self.fitness_history = defaultdict(list)
    
    def create_random_architecture(self, num_layers: int) -> List[Dict[str, Any]]:
        """Create a random architecture"""
        architecture = []
        
        for _ in range(num_layers):
            layer = {
                'type': random.choice(self.search_space.layer_types),
                'hidden_dim': random.choice(self.search_space.hidden_dims),
                'activation': random.choice(self.search_space.activation_functions),
                'dropout': random.choice(self.search_space.dropout_rates)
            }
            
            if layer['type'] == 'attention':
                layer['num_heads'] = random.choice(self.search_space.num_heads_options)
            elif layer['type'] == 'conv':
                layer['kernel_size'] = random.choice(self.search_space.kernel_sizes)
            
            architecture.append(layer)
        
        return architecture
    
    def mutate(self, architecture: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Mutate an architecture"""
        mutated = copy.deepcopy(architecture)
        
        # Random mutations
        for layer in mutated:
            if random.random() < self.mutation_rate:
                # Mutate layer type
                layer['type'] = random.choice(self.search_space.layer_types)
            
            if random.random() < self.mutation_rate:
                # Mutate hyperparameters
                layer['hidden_dim'] = random.choice(self.search_space.hidden_dims)
                layer['activation'] = random.choice(self.search_space.activation_functions)
                layer['dropout'] = random.choice(self.search_space.dropout_rates)
        
        # Add/remove layers
        if random.random() < self.mutation_rate and len(mutated) > 2:
            # Remove a random layer
            idx = random.randint(0, len(mutated) - 1)
            mutated.pop(idx)
        
        if random.random() < self.mutation_rate and len(mutated) < 20:
            # Add a random layer
            idx = random.randint(0, len(mutated))
            new_layer = self.create_random_architecture(1)[0]
            mutated.insert(idx, new_layer)
        
        return mutated
    
    def crossover(
        self,
        parent1: List[Dict[str, Any]],
        parent2: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Crossover two architectures"""
        # Single-point crossover
        min_len = min(len(parent1), len(parent2))
        crossover_point = random.randint(1, min_len - 1)
        
        child = parent1[:crossover_point] + parent2[crossover_point:]
        
        return child
    
    def tournament_selection(
        self,
        population: List[Tuple[List[Dict[str, Any]], float]]
    ) -> List[Dict[str, Any]]:
        """Tournament selection"""
        tournament = random.sample(population, self.tournament_size)
        winner = max(tournament, key=lambda x: x[1])
        return winner[0]
    
    def evaluate_architecture(
        self,
        architecture: List[Dict[str, Any]],
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_epochs: int = 5
    ) -> float:
        """Evaluate an architecture"""
        # Build model from architecture
        model = self.build_model(architecture)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Train for a few epochs
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(num_epochs):
            model.train()
            for batch in train_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        # Evaluate on validation set
        model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_correct += (predicted == targets).sum().item()
                total_samples += targets.size(0)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def build_model(self, architecture: List[Dict[str, Any]]) -> nn.Module:
        """Build a model from architecture specification"""
        layers = []
        
        for i, layer_spec in enumerate(architecture):
            if layer_spec['type'] == 'linear':
                layer = nn.Linear(
                    layer_spec.get('input_dim', 512),
                    layer_spec['hidden_dim']
                )
            elif layer_spec['type'] == 'conv':
                layer = nn.Conv1d(
                    layer_spec.get('input_channels', 512),
                    layer_spec['hidden_dim'],
                    kernel_size=layer_spec.get('kernel_size', 3),
                    padding=layer_spec.get('kernel_size', 3) // 2
                )
            elif layer_spec['type'] == 'attention':
                config = AttentionConfig(
                    dim=layer_spec['hidden_dim'],
                    num_heads=layer_spec.get('num_heads', 8)
                )
                layer = FlashAttention(config)
            else:
                layer = nn.Identity()
            
            layers.append(layer)
            
            # Add activation
            if layer_spec['activation'] == 'relu':
                layers.append(nn.ReLU())
            elif layer_spec['activation'] == 'gelu':
                layers.append(nn.GELU())
            elif layer_spec['activation'] == 'swish':
                layers.append(nn.SiLU())
            
            # Add dropout
            if layer_spec['dropout'] > 0:
                layers.append(nn.Dropout(layer_spec['dropout']))
        
        return nn.Sequential(*layers)
    
    def search(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        num_generations: int = 50,
        num_layers: int = 8
    ) -> List[Dict[str, Any]]:
        """Run evolutionary search"""
        logger.info("Starting Evolutionary NAS...")
        
        # Initialize population
        self.population = []
        for _ in range(self.population_size):
            arch = self.create_random_architecture(num_layers)
            fitness = self.evaluate_architecture(arch, train_loader, val_loader)
            self.population.append((arch, fitness))
            logger.info(f"Initial architecture fitness: {fitness:.4f}")
        
        # Evolution loop
        for generation in range(num_generations):
            logger.info(f"Generation {generation + 1}/{num_generations}")
            
            # Create new population
            new_population = []
            
            # Elitism: keep best architectures
            self.population.sort(key=lambda x: x[1], reverse=True)
            elite_size = self.population_size // 10
            new_population.extend(self.population[:elite_size])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.tournament_selection(self.population)
                parent2 = self.tournament_selection(self.population)
                
                # Crossover
                child = self.crossover(parent1, parent2)
                
                # Mutation
                child = self.mutate(child)
                
                # Evaluate
                fitness = self.evaluate_architecture(child, train_loader, val_loader)
                new_population.append((child, fitness))
            
            self.population = new_population
            
            # Log best architecture
            best_arch, best_fitness = max(self.population, key=lambda x: x[1])
            logger.info(f"Best fitness: {best_fitness:.4f}")
            self.fitness_history[generation] = best_fitness
        
        # Return best architecture
        best_arch, _ = max(self.population, key=lambda x: x[1])
        return best_arch
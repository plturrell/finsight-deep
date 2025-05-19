# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Distributed training support for AIQToolkit
Implements data parallel, model parallel, and pipeline parallel training
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from dataclasses import dataclass
import asyncio
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for distributed training"""
    strategy: str = "ddp"  # "ddp", "fsdp", "pipeline", "model_parallel"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    gradient_clipping: float = 1.0
    checkpoint_interval: int = 1000
    logging_interval: int = 100
    enable_profiling: bool = False
    # DDP specific
    find_unused_parameters: bool = False
    bucket_cap_mb: int = 25
    # FSDP specific
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    # Pipeline specific
    pipeline_stages: int = 2
    micro_batch_size: int = 4


@dataclass
class TrainingMetrics:
    """Metrics collected during training"""
    step: int
    loss: float
    learning_rate: float
    throughput: float  # samples/second
    gpu_memory_used: float
    gradient_norm: float
    timestamp: datetime


class DistributedTrainer:
    """Distributed trainer for AIQToolkit models"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: TrainingConfig,
                 gpu_direct_manager=None):
        self.model = model
        self.config = config
        self.gpu_direct = gpu_direct_manager
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.device = torch.device(f'cuda:{self.rank}' if torch.cuda.is_available() else 'cpu')
        
        # Metrics tracking
        self.metrics_history: List[TrainingMetrics] = []
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Initialize distributed model
        self.distributed_model = self._setup_distributed_model()
    
    def _setup_distributed_model(self) -> nn.Module:
        """Setup distributed model based on strategy"""
        if self.config.strategy == "ddp":
            return self._setup_ddp()
        elif self.config.strategy == "fsdp":
            return self._setup_fsdp()
        elif self.config.strategy == "pipeline":
            return self._setup_pipeline()
        elif self.config.strategy == "model_parallel":
            return self._setup_model_parallel()
        else:
            raise ValueError(f"Unknown strategy: {self.config.strategy}")
    
    def _setup_ddp(self) -> DDP:
        """Setup DistributedDataParallel"""
        model = self.model.to(self.device)
        
        ddp_model = DDP(
            model,
            device_ids=[self.rank],
            output_device=self.rank,
            find_unused_parameters=self.config.find_unused_parameters,
            bucket_cap_mb=self.config.bucket_cap_mb
        )
        
        logger.info(f"DDP setup complete on rank {self.rank}")
        return ddp_model
    
    def _setup_fsdp(self) -> FSDP:
        """Setup FullyShardedDataParallel"""
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            CPUOffload,
            ShardingStrategy,
        )
        
        # Configure sharding strategy
        sharding_strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }
        
        sharding_strategy = sharding_strategies.get(
            self.config.fsdp_sharding_strategy,
            ShardingStrategy.FULL_SHARD
        )
        
        # Configure CPU offload
        cpu_offload = CPUOffload(offload_params=True) if self.config.fsdp_cpu_offload else None
        
        fsdp_model = FSDP(
            self.model,
            sharding_strategy=sharding_strategy,
            cpu_offload=cpu_offload,
            device_id=self.rank,
        )
        
        logger.info(f"FSDP setup complete on rank {self.rank}")
        return fsdp_model
    
    def _setup_pipeline(self) -> nn.Module:
        """Setup pipeline parallel training"""
        from torch.distributed.pipeline.sync import Pipe
        
        # Split model into stages
        stages = self._split_model_stages(self.config.pipeline_stages)
        
        # Create pipeline
        model = Pipe(
            nn.Sequential(*stages),
            balance=[len(stage) for stage in stages],
            devices=[f'cuda:{i}' for i in range(len(stages))],
            chunks=self.config.micro_batch_size
        )
        
        logger.info(f"Pipeline parallel setup complete with {len(stages)} stages")
        return model
    
    def _setup_model_parallel(self) -> nn.Module:
        """Setup model parallel training"""
        # This is a simplified example - real implementation would be more complex
        # and model-specific
        
        # Example: split linear layers across GPUs
        if hasattr(self.model, 'layers'):
            layers_per_gpu = len(self.model.layers) // self.world_size
            
            for i, layer in enumerate(self.model.layers):
                device = i // layers_per_gpu
                layer.to(f'cuda:{device}')
        
        logger.info(f"Model parallel setup complete across {self.world_size} GPUs")
        return self.model
    
    def _split_model_stages(self, num_stages: int) -> List[nn.Sequential]:
        """Split model into pipeline stages"""
        if not hasattr(self.model, 'layers'):
            raise ValueError("Model must have 'layers' attribute for pipeline parallel")
        
        layers = list(self.model.layers)
        layers_per_stage = len(layers) // num_stages
        
        stages = []
        for i in range(num_stages):
            start = i * layers_per_stage
            end = start + layers_per_stage if i < num_stages - 1 else len(layers)
            stages.append(nn.Sequential(*layers[start:end]))
        
        return stages
    
    async def train_step(self,
                        batch: Dict[str, torch.Tensor],
                        optimizer: torch.optim.Optimizer,
                        criterion: nn.Module,
                        step: int) -> TrainingMetrics:
        """Execute a single training step"""
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        
        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Mixed precision context
        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            # Forward pass
            outputs = self.distributed_model(batch['input'])
            loss = criterion(outputs, batch['target'])
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.config.mixed_precision and self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Gradient accumulation
        if (step + 1) % self.config.gradient_accumulation_steps == 0:
            # Gradient clipping
            if self.config.gradient_clipping > 0:
                if self.config.mixed_precision and self.scaler:
                    self.scaler.unscale_(optimizer)
                
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.distributed_model.parameters(),
                    self.config.gradient_clipping
                )
            else:
                grad_norm = 0.0
            
            # Optimizer step
            if self.config.mixed_precision and self.scaler:
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                optimizer.step()
            
            optimizer.zero_grad()
        else:
            grad_norm = 0.0
        
        end_time.record()
        torch.cuda.synchronize()
        
        # Calculate metrics
        elapsed_time = start_time.elapsed_time(end_time) / 1000  # Convert to seconds
        batch_size = batch['input'].size(0)
        throughput = batch_size * self.world_size / elapsed_time
        
        # GPU memory usage
        gpu_memory = torch.cuda.memory_allocated(self.device) / (1024**3)  # GB
        
        metrics = TrainingMetrics(
            step=step,
            loss=loss.item() * self.config.gradient_accumulation_steps,
            learning_rate=optimizer.param_groups[0]['lr'],
            throughput=throughput,
            gpu_memory_used=gpu_memory,
            gradient_norm=grad_norm,
            timestamp=datetime.now()
        )
        
        self.metrics_history.append(metrics)
        
        return metrics
    
    async def train_epoch(self,
                         dataloader: DataLoader,
                         optimizer: torch.optim.Optimizer,
                         criterion: nn.Module,
                         epoch: int) -> List[TrainingMetrics]:
        """Train for one epoch"""
        self.distributed_model.train()
        epoch_metrics = []
        
        # Create distributed sampler
        if dist.is_initialized():
            sampler = DistributedSampler(
                dataloader.dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=True
            )
            sampler.set_epoch(epoch)
            dataloader.sampler = sampler
        
        # Training loop
        for step, batch in enumerate(dataloader):
            global_step = epoch * len(dataloader) + step
            
            # Train step
            metrics = await self.train_step(batch, optimizer, criterion, global_step)
            epoch_metrics.append(metrics)
            
            # Logging
            if step % self.config.logging_interval == 0 and self.rank == 0:
                logger.info(
                    f"Epoch {epoch}, Step {step}/{len(dataloader)}, "
                    f"Loss: {metrics.loss:.4f}, "
                    f"Throughput: {metrics.throughput:.1f} samples/s, "
                    f"GPU Memory: {metrics.gpu_memory_used:.2f} GB"
                )
            
            # Checkpointing
            if global_step % self.config.checkpoint_interval == 0 and self.rank == 0:
                await self.save_checkpoint(f"checkpoint_step_{global_step}.pt", optimizer, epoch, step)
        
        return epoch_metrics
    
    async def save_checkpoint(self, 
                            filename: str,
                            optimizer: torch.optim.Optimizer,
                            epoch: int,
                            step: int):
        """Save training checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.distributed_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': self.config,
            'metrics_history': self.metrics_history[-1000:],  # Save recent metrics
        }
        
        if self.config.mixed_precision and self.scaler:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, filename)
        logger.info(f"Checkpoint saved: {filename}")
    
    async def load_checkpoint(self, 
                            filename: str,
                            optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
        """Load training checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        
        self.distributed_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.config.mixed_precision and self.scaler and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.metrics_history = checkpoint.get('metrics_history', [])
        
        logger.info(f"Checkpoint loaded: {filename}")
        return checkpoint['epoch'], checkpoint['step']
    
    def profile_training(self, 
                        dataloader: DataLoader,
                        num_steps: int = 10) -> Dict[str, Any]:
        """Profile training performance"""
        if not self.config.enable_profiling:
            return {}
        
        import torch.profiler
        
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        
        schedule = torch.profiler.schedule(
            wait=1,
            warmup=3,
            active=5,
            repeat=1
        )
        
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for step, batch in enumerate(dataloader):
                if step >= num_steps:
                    break
                
                # Dummy optimizer for profiling
                optimizer = torch.optim.SGD(self.distributed_model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                asyncio.run(self.train_step(batch, optimizer, criterion, step))
                prof.step()
        
        # Export Chrome trace
        prof.export_chrome_trace("trace.json")
        
        return {
            "key_averages": prof.key_averages(),
            "total_average": prof.total_average(),
        }


class GradientCompression:
    """Gradient compression for reduced communication"""
    
    def __init__(self, compression_ratio: float = 0.1):
        self.compression_ratio = compression_ratio
        self.residuals = {}
    
    def compress_gradient(self, 
                         gradient: torch.Tensor,
                         name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress gradient using top-k sparsification
        
        Args:
            gradient: Gradient tensor
            name: Parameter name for residual tracking
            
        Returns:
            Compressed gradient and indices
        """
        # Flatten gradient
        original_shape = gradient.shape
        gradient_flat = gradient.flatten()
        
        # Add residual from previous iteration
        if name in self.residuals:
            gradient_flat += self.residuals[name]
        
        # Top-k selection
        k = int(gradient_flat.numel() * self.compression_ratio)
        values, indices = torch.topk(gradient_flat.abs(), k)
        
        # Get actual values (with sign)
        compressed_values = gradient_flat[indices]
        
        # Store residual
        residual = gradient_flat.clone()
        residual[indices] = 0
        self.residuals[name] = residual
        
        return compressed_values, indices
    
    def decompress_gradient(self,
                          values: torch.Tensor,
                          indices: torch.Tensor,
                          shape: Tuple[int, ...]) -> torch.Tensor:
        """Decompress gradient from sparse representation"""
        # Create zero tensor
        gradient = torch.zeros(np.prod(shape), device=values.device, dtype=values.dtype)
        
        # Fill in compressed values
        gradient[indices] = values
        
        # Reshape to original shape
        return gradient.reshape(shape)


# Example usage
if __name__ == "__main__":
    # Example model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.Linear(784, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 10)
            ])
        
        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x
    
    # Training configuration
    config = TrainingConfig(
        strategy="ddp",
        mixed_precision=True,
        gradient_accumulation_steps=4,
        gradient_clipping=1.0
    )
    
    # This would run in a distributed environment
    # trainer = DistributedTrainer(model, config)
    # metrics = await trainer.train_epoch(dataloader, optimizer, criterion, epoch=0)
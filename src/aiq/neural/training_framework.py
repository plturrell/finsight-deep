# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Advanced Training Framework for Neural Models
Includes distributed training, mixed precision, and advanced optimization
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import wandb
import numpy as np
from pathlib import Path
import logging
from tqdm import tqdm
import time

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training"""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    gradient_clip: float = 1.0
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    mixed_precision: bool = True
    distributed: bool = False
    gradient_accumulation_steps: int = 1
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 10
    use_wandb: bool = True
    checkpoint_dir: str = "./checkpoints"
    resume_from_checkpoint: Optional[str] = None


class CosineAnnealingWarmup(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup"""
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        eta_min: float = 0
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        super().__init__(optimizer)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            return [
                base_lr * self.last_epoch / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_steps) / (
                self.total_steps - self.warmup_steps
            )
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + np.cos(np.pi * progress)) / 2
                for base_lr in self.base_lrs
            ]


class AdamW(torch.optim.AdamW):
    """AdamW with decoupled weight decay and gradient centralization"""
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, amsgrad=False, gradient_centralization=True):
        self.gradient_centralization = gradient_centralization
        super().__init__(params, lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
    
    @torch.no_grad()
    def step(self, closure=None):
        if self.gradient_centralization:
            # Centralize gradients
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    grad = p.grad.data
                    if len(grad.shape) > 1:
                        grad.add_(-grad.mean(dim=list(range(1, len(grad.shape))), 
                                keepdim=True))
        
        return super().step(closure)


class NeuralTrainer:
    """Advanced neural network trainer with modern techniques"""
    
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: torch.utils.data.Dataset,
        eval_dataset: Optional[torch.utils.data.Dataset] = None,
        collate_fn: Optional[Callable] = None
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        
        # Setup device and optimization
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tensor_optimizer = TensorCoreOptimizer()
        
        # Initialize training components
        self._setup_distributed()
        self._setup_model()
        self._setup_optimizer()
        self._setup_dataloaders()
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_metric = float('inf')
        
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(project="aiq-neural", config=config.__dict__)
    
    def _setup_distributed(self):
        """Setup distributed training if enabled"""
        if self.config.distributed:
            dist.init_process_group(backend='nccl')
            self.local_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device(f'cuda:{self.local_rank}')
        else:
            self.local_rank = 0
            self.world_size = 1
    
    def _setup_model(self):
        """Setup and optimize model"""
        # Optimize for Tensor Cores
        self.model = self.tensor_optimizer.optimize_model(self.model)
        self.model = self.model.to(self.device)
        
        # Distributed data parallel
        if self.config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=True
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        # Group parameters
        no_decay = ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay,
            },
            {
                'params': [p for n, p in self.model.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0,
            }
        ]
        
        # Create optimizer
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Create scheduler
        total_steps = (
            len(self.train_dataset) // self.config.batch_size // 
            self.config.gradient_accumulation_steps * self.config.num_epochs
        )
        self.scheduler = CosineAnnealingWarmup(
            self.optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=total_steps
        )
    
    def _setup_dataloaders(self):
        """Setup data loaders"""
        # Distributed sampler
        if self.config.distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.local_rank
            )
        else:
            train_sampler = None
        
        # Create dataloaders
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            collate_fn=self.collate_fn,
            num_workers=4,
            pin_memory=True
        )
        
        if self.eval_dataset:
            self.eval_loader = torch.utils.data.DataLoader(
                self.eval_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
                num_workers=4,
                pin_memory=True
            )
    
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        # Resume from checkpoint if specified
        if self.config.resume_from_checkpoint:
            self.load_checkpoint(self.config.resume_from_checkpoint)
        
        # Training loop
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Set epoch for distributed sampler
            if self.config.distributed:
                self.train_loader.sampler.set_epoch(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Evaluate
            if self.eval_dataset and epoch % 5 == 0:
                eval_metrics = self.evaluate()
                
                # Save best model
                if eval_metrics['loss'] < self.best_metric:
                    self.best_metric = eval_metrics['loss']
                    self.save_checkpoint('best')
            
            # Save checkpoint
            if epoch % 10 == 0:
                self.save_checkpoint(f'epoch_{epoch}')
        
        logger.info("Training completed!")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train single epoch"""
        self.model.train()
        epoch_loss = 0.0
        epoch_metrics = {}
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch}",
            disable=self.local_rank != 0
        )
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = self._move_to_device(batch)
            
            # Forward pass with mixed precision
            with autocast(enabled=self.config.mixed_precision):
                outputs = self.model(**batch)
                loss = outputs['loss'] / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.gradient_clip > 0:
                    if self.config.mixed_precision:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )
                
                # Optimizer step
                if self.config.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    metrics = {
                        'train/loss': loss.item() * self.config.gradient_accumulation_steps,
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'train/global_step': self.global_step
                    }
                    
                    # Update metrics
                    if 'metrics' in outputs:
                        for k, v in outputs['metrics'].items():
                            metrics[f'train/{k}'] = v
                    
                    # Log to wandb
                    if self.config.use_wandb and self.local_rank == 0:
                        wandb.log(metrics, step=self.global_step)
                    
                    # Update progress bar
                    progress_bar.set_postfix(loss=loss.item())
            
            epoch_loss += loss.item()
        
        return {'loss': epoch_loss / len(self.train_loader)}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model"""
        self.model.eval()
        eval_loss = 0.0
        eval_metrics = {}
        
        with torch.no_grad():
            for batch in tqdm(self.eval_loader, desc="Evaluating", 
                            disable=self.local_rank != 0):
                batch = self._move_to_device(batch)
                
                with autocast(enabled=self.config.mixed_precision):
                    outputs = self.model(**batch)
                    loss = outputs['loss']
                
                eval_loss += loss.item()
                
                # Accumulate metrics
                if 'metrics' in outputs:
                    for k, v in outputs['metrics'].items():
                        if k not in eval_metrics:
                            eval_metrics[k] = 0.0
                        eval_metrics[k] += v
        
        # Average metrics
        num_batches = len(self.eval_loader)
        eval_loss /= num_batches
        for k in eval_metrics:
            eval_metrics[k] /= num_batches
        
        # Log metrics
        if self.config.use_wandb and self.local_rank == 0:
            wandb.log({
                'eval/loss': eval_loss,
                **{f'eval/{k}': v for k, v in eval_metrics.items()}
            }, step=self.global_step)
        
        return {'loss': eval_loss, **eval_metrics}
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        if self.local_rank != 0:
            return
        
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'global_step': self.global_step,
            'current_epoch': self.current_epoch,
            'best_metric': self.best_metric
        }
        
        if self.config.mixed_precision:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        torch.save(checkpoint, checkpoint_dir / f'{name}.pt')
        logger.info(f"Saved checkpoint: {name}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['current_epoch']
        self.best_metric = checkpoint['best_metric']
        
        if self.config.mixed_precision and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        logger.info(f"Loaded checkpoint from {path}")
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch to device"""
        return {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }
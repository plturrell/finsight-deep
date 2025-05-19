# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Advanced Reinforcement Learning Components
Includes PPO, SAC, and Multi-Agent RL
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import deque
import gym
from torch.distributions import Categorical, Normal
import copy

from aiq.neural.advanced_architectures import (
    FlashAttention, AttentionConfig, MixtureOfExperts, NeuralMemoryBank
)
from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


@dataclass
class RLConfig:
    """Configuration for RL algorithms"""
    algo: str = "ppo"  # ppo, sac, dqn, a3c
    gamma: float = 0.99
    tau: float = 0.005
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 1000000
    update_freq: int = 1
    n_steps: int = 2048
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_update_freq: int = 1000
    use_gpu: bool = True


class ReplayBuffer:
    """Experience replay buffer with prioritization"""
    
    def __init__(self, capacity: int, prioritized: bool = False, alpha: float = 0.6):
        self.capacity = capacity
        self.prioritized = prioritized
        self.alpha = alpha
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.position = 0
    
    def push(self, state, action, reward, next_state, done, priority=None):
        """Add experience to buffer"""
        experience = (state, action, reward, next_state, done)
        
        if self.prioritized:
            priority = priority or max(self.priorities, default=1.0)
            self.priorities.append(priority)
        
        self.buffer.append(experience)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch from buffer"""
        if self.prioritized:
            # Prioritized sampling
            probs = np.array(self.priorities) ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            weights = (len(self.buffer) * probs[indices]) ** -beta
            weights /= weights.max()
            
            batch = [self.buffer[idx] for idx in indices]
            return batch, weights, indices
        else:
            # Uniform sampling
            indices = np.random.choice(len(self.buffer), batch_size)
            batch = [self.buffer[idx] for idx in indices]
            return batch, None, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for prioritized replay"""
        if self.prioritized:
            for idx, priority in zip(indices, priorities):
                self.priorities[idx] = priority
    
    def __len__(self):
        return len(self.buffer)


class ActorCriticNetwork(nn.Module):
    """Shared actor-critic network with attention"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_attention: bool = True,
        continuous: bool = True
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.continuous = continuous
        
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Attention layers
        if use_attention:
            attention_config = AttentionConfig(
                dim=hidden_dim,
                num_heads=8,
                dim_head=32
            )
            self.attention_layers = nn.ModuleList([
                FlashAttention(attention_config)
                for _ in range(num_layers)
            ])
        else:
            self.attention_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim)
                )
                for _ in range(num_layers)
            ])
        
        # Actor head
        if continuous:
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            self.actor = nn.Linear(hidden_dim, action_dim)
        
        # Critic head
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Optimize for GPU
        self.optimizer = TensorCoreOptimizer()
        self._optimize_layers()
    
    def _optimize_layers(self):
        """Optimize layers for Tensor Core acceleration"""
        optimized_modules = []
        for module in self.modules():
            if isinstance(module, nn.Linear):
                optimized_modules.append(self.optimizer.optimize_model(module))
        
        # Replace modules
        for i, module in enumerate(self.modules()):
            if isinstance(module, nn.Linear) and i < len(optimized_modules):
                module = optimized_modules[i]
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action distribution and value"""
        # Feature extraction
        features = self.feature_extractor(state)
        
        # Apply attention layers
        for layer in self.attention_layers:
            if isinstance(layer, FlashAttention):
                features = features + layer(features.unsqueeze(1)).squeeze(1)
            else:
                features = features + layer(features)
        
        # Critic output
        value = self.critic(features)
        
        # Actor output
        if self.continuous:
            action_mean = self.actor_mean(features)
            action_std = self.actor_log_std.exp()
            return action_mean, action_std, value
        else:
            action_logits = self.actor(features)
            return action_logits, value


class PPO:
    """Proximal Policy Optimization with advanced features"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RLConfig,
        continuous: bool = True
    ):
        self.config = config
        self.continuous = continuous
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor_critic = ActorCriticNetwork(
            state_dim, action_dim, continuous=continuous
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.actor_critic.parameters(),
            lr=config.lr_actor,
            weight_decay=0.01
        )
        
        # Memory
        self.memory = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
    
    def select_action(self, state: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            if self.continuous:
                action_mean, action_std, value = self.actor_critic(state)
                dist = Normal(action_mean, action_std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(-1)
            else:
                action_logits, value = self.actor_critic(state)
                dist = Categorical(logits=action_logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)
        
        return (
            action.cpu().numpy().squeeze(),
            log_prob.item(),
            value.item()
        )
    
    def store(self, state, action, reward, value, log_prob, done):
        """Store experience in memory"""
        self.memory['states'].append(state)
        self.memory['actions'].append(action)
        self.memory['rewards'].append(reward)
        self.memory['values'].append(value)
        self.memory['log_probs'].append(log_prob)
        self.memory['dones'].append(done)
    
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        """Compute Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        
        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[i + 1]
            
            delta = rewards[i] + gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + gamma * lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
        
        return torch.FloatTensor(advantages)
    
    def update(self):
        """Update policy using collected experiences"""
        # Convert memory to tensors
        states = torch.FloatTensor(self.memory['states']).to(self.device)
        actions = torch.FloatTensor(self.memory['actions']).to(self.device)
        rewards = torch.FloatTensor(self.memory['rewards']).to(self.device)
        values = torch.FloatTensor(self.memory['values']).to(self.device)
        log_probs = torch.FloatTensor(self.memory['log_probs']).to(self.device)
        dones = torch.FloatTensor(self.memory['dones']).to(self.device)
        
        # Compute advantages
        advantages = self.compute_gae(
            self.memory['rewards'],
            self.memory['values'],
            self.memory['dones']
        ).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute returns
        returns = advantages + values
        
        # PPO update
        for epoch in range(self.config.n_epochs):
            # Sample mini-batches
            indices = torch.randperm(len(states))
            
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                batch_indices = indices[start:end]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_log_probs = log_probs[batch_indices]
                batch_returns = returns[batch_indices]
                batch_advantages = advantages[batch_indices]
                
                # Get current policy outputs
                if self.continuous:
                    action_mean, action_std, values = self.actor_critic(batch_states)
                    dist = Normal(action_mean, action_std)
                    new_log_probs = dist.log_prob(batch_actions).sum(-1)
                    entropy = dist.entropy().sum(-1)
                else:
                    action_logits, values = self.actor_critic(batch_states)
                    dist = Categorical(logits=action_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy()
                
                # Compute ratio
                ratio = torch.exp(new_log_probs - batch_log_probs)
                
                # Compute surrogate loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                values = values.squeeze()
                value_loss = F.mse_loss(values, batch_returns)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = (
                    actor_loss + 
                    self.config.vf_coef * value_loss + 
                    self.config.ent_coef * entropy_loss
                )
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor_critic.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
        
        # Clear memory
        self.memory = {k: [] for k in self.memory.keys()}


class SAC:
    """Soft Actor-Critic with automatic temperature tuning"""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: RLConfig
    ):
        self.config = config
        self.device = torch.device('cuda' if config.use_gpu and torch.cuda.is_available() else 'cpu')
        
        # Networks
        self.actor = self._build_actor(state_dim, action_dim).to(self.device)
        self.critic1 = self._build_critic(state_dim + action_dim).to(self.device)
        self.critic2 = self._build_critic(state_dim + action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        
        # Optimizers
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=config.lr_actor)
        self.critic1_optimizer = torch.optim.AdamW(self.critic1.parameters(), lr=config.lr_critic)
        self.critic2_optimizer = torch.optim.AdamW(self.critic2.parameters(), lr=config.lr_critic)
        
        # Automatic entropy tuning
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr_actor)
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size, prioritized=True)
    
    def _build_actor(self, state_dim: int, action_dim: int) -> nn.Module:
        """Build actor network"""
        return nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)  # Mean and log_std
        )
    
    def _build_critic(self, input_dim: int) -> nn.Module:
        """Build critic network"""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """Select action using current policy"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.actor(state)
            mean, log_std = output.chunk(2, dim=-1)
            std = log_std.exp()
            
            if evaluate:
                action = mean
            else:
                dist = Normal(mean, std)
                action = dist.rsample()
            
            action = torch.tanh(action)
        
        return action.cpu().numpy().squeeze()
    
    def update(self):
        """Update SAC networks"""
        if len(self.replay_buffer) < self.config.batch_size:
            return
        
        # Sample batch
        batch, weights, indices = self.replay_buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Update critics
        with torch.no_grad():
            next_output = self.actor(next_states)
            next_mean, next_log_std = next_output.chunk(2, dim=-1)
            next_std = next_log_std.exp()
            next_dist = Normal(next_mean, next_std)
            next_action = next_dist.rsample()
            next_action = torch.tanh(next_action)
            
            next_q1 = self.critic1_target(torch.cat([next_states, next_action], dim=1))
            next_q2 = self.critic2_target(torch.cat([next_states, next_action], dim=1))
            next_q = torch.min(next_q1, next_q2)
            
            # Entropy term
            next_log_prob = next_dist.log_prob(next_action).sum(-1, keepdim=True)
            next_log_prob -= torch.log(1 - next_action.pow(2) + 1e-6).sum(-1, keepdim=True)
            
            target_q = rewards + self.config.gamma * (1 - dones) * (next_q - self.log_alpha.exp() * next_log_prob)
        
        # Critic losses
        q1 = self.critic1(torch.cat([states, actions], dim=1))
        q2 = self.critic2(torch.cat([states, actions], dim=1))
        
        critic1_loss = (weights * F.mse_loss(q1, target_q, reduction='none')).mean()
        critic2_loss = (weights * F.mse_loss(q2, target_q, reduction='none')).mean()
        
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        
        # Update actor
        output = self.actor(states)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.rsample()
        action = torch.tanh(action)
        
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(-1, keepdim=True)
        
        q1 = self.critic1(torch.cat([states, action], dim=1))
        q2 = self.critic2(torch.cat([states, action], dim=1))
        q = torch.min(q1, q2)
        
        actor_loss = (self.log_alpha.exp() * log_prob - q).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update priorities
        td_errors = torch.abs(target_q - q1).squeeze().cpu().numpy()
        self.replay_buffer.update_priorities(indices, td_errors + 1e-6)
        
        # Soft update target networks
        for param, target_param in zip(self.critic1.parameters(), self.critic1_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
        
        for param, target_param in zip(self.critic2.parameters(), self.critic2_target.parameters()):
            target_param.data.copy_(self.config.tau * param.data + (1 - self.config.tau) * target_param.data)
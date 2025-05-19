"""Federated learning implementation for AIQToolkit"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import grpc
from concurrent import futures

from aiq.distributed.edge.edge_node import EdgeNode, EdgeNodeConfig
from aiq.distributed.node_manager import NodeInfo
from aiq.distributed.security.auth import AuthManager, AuthConfig
from aiq.distributed.security.privacy import PrivacyManager, DifferentialPrivacyConfig

logger = logging.getLogger(__name__)


class AggregationStrategy(Enum):
    """Federated learning aggregation strategies"""
    FEDAVG = "fedavg"          # Federated Averaging
    FEDPROX = "fedprox"        # Federated Proximal
    FEDYOGI = "fedyogi"        # Federated Yogi (adaptive optimizer)
    SCAFFOLD = "scaffold"       # Scaffold algorithm
    PERSONALIZED = "personalized"  # Personalized federated learning


class ClientSelection(Enum):
    """Client selection strategies"""
    RANDOM = "random"
    WEIGHTED = "weighted"      # Based on data size
    QUALITY = "quality"        # Based on contribution quality
    RESOURCE = "resource"      # Based on available resources


@dataclass
class FederatedConfig:
    """Configuration for federated learning"""
    rounds: int = 100
    clients_per_round: int = 10
    min_clients: int = 5
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.01
    aggregation_strategy: AggregationStrategy = AggregationStrategy.FEDAVG
    client_selection: ClientSelection = ClientSelection.RANDOM
    differential_privacy: bool = True
    privacy_budget: float = 10.0
    adaptive_aggregation: bool = True
    compression_enabled: bool = True
    secure_aggregation: bool = True
    checkpoint_interval: int = 10
    convergence_threshold: float = 0.001


@dataclass
class ClientUpdate:
    """Update from a federated learning client"""
    client_id: str
    round_number: int
    model_updates: Dict[str, torch.Tensor]
    metrics: Dict[str, float]
    sample_count: int
    computation_time: float
    privacy_noise_added: bool = False
    timestamp: datetime = None


class FederatedLearningServer:
    """Central server for federated learning coordination"""
    
    def __init__(self, config: FederatedConfig):
        self.config = config
        self.current_round = 0
        self.global_model: Optional[nn.Module] = None
        self.client_registry: Dict[str, ClientInfo] = {}
        self.round_updates: List[ClientUpdate] = []
        self.metrics_history: List[Dict[str, Any]] = []
        self.privacy_manager = PrivacyManager(
            DifferentialPrivacyConfig(
                epsilon=config.privacy_budget,
                delta=1e-5
            )
        ) if config.differential_privacy else None
        self.auth_manager = AuthManager(AuthConfig(secret_key="dummy_key"))
        self._running = False
        
    async def initialize_global_model(self, model: nn.Module):
        """Initialize the global model"""
        self.global_model = model
        self.initial_params = {
            name: param.clone() for name, param in model.state_dict().items()
        }
        logger.info("Initialized global model for federated learning")
        
    async def register_client(self, client_info: Dict[str, Any]) -> str:
        """Register a new federated learning client"""
        client_id = f"client_{len(self.client_registry)}"
        
        self.client_registry[client_id] = ClientInfo(
            client_id=client_id,
            node_info=NodeInfo(**client_info["node_info"]),
            data_size=client_info.get("data_size", 0),
            capabilities=client_info.get("capabilities", {}),
            trust_score=1.0
        )
        
        logger.info(f"Registered client {client_id}")
        return client_id
        
    async def run_training(self):
        """Run federated learning training rounds"""
        self._running = True
        
        while self.current_round < self.config.rounds and self._running:
            try:
                # Select clients for this round
                selected_clients = await self._select_clients()
                
                if len(selected_clients) < self.config.min_clients:
                    logger.warning(f"Not enough clients available ({len(selected_clients)})")
                    await asyncio.sleep(30)
                    continue
                
                # Distribute model to selected clients
                await self._distribute_model(selected_clients)
                
                # Wait for client updates
                updates = await self._collect_updates(selected_clients)
                
                # Aggregate updates
                await self._aggregate_updates(updates)
                
                # Evaluate global model
                metrics = await self._evaluate_global_model()
                
                # Save checkpoint if needed
                if self.current_round % self.config.checkpoint_interval == 0:
                    await self._save_checkpoint()
                
                # Check convergence
                if self._check_convergence(metrics):
                    logger.info("Federated learning converged")
                    break
                
                self.current_round += 1
                
            except Exception as e:
                logger.error(f"Error in round {self.current_round}: {e}")
                
        self._running = False
        
    async def _select_clients(self) -> List[str]:
        """Select clients for current round"""
        available_clients = [
            cid for cid, info in self.client_registry.items()
            if info.node_info.status == "active"
        ]
        
        if self.config.client_selection == ClientSelection.RANDOM:
            selected = np.random.choice(
                available_clients,
                min(self.config.clients_per_round, len(available_clients)),
                replace=False
            ).tolist()
            
        elif self.config.client_selection == ClientSelection.WEIGHTED:
            # Weight by data size
            weights = [self.client_registry[cid].data_size for cid in available_clients]
            total_weight = sum(weights)
            probs = [w / total_weight for w in weights]
            
            selected = np.random.choice(
                available_clients,
                min(self.config.clients_per_round, len(available_clients)),
                replace=False,
                p=probs
            ).tolist()
            
        elif self.config.client_selection == ClientSelection.QUALITY:
            # Select based on trust scores
            sorted_clients = sorted(
                available_clients,
                key=lambda cid: self.client_registry[cid].trust_score,
                reverse=True
            )
            selected = sorted_clients[:self.config.clients_per_round]
            
        elif self.config.client_selection == ClientSelection.RESOURCE:
            # Select based on available resources
            sorted_clients = sorted(
                available_clients,
                key=lambda cid: (
                    self.client_registry[cid].node_info.available_gpus +
                    self.client_registry[cid].node_info.cpu_available / 100
                ),
                reverse=True
            )
            selected = sorted_clients[:self.config.clients_per_round]
            
        logger.info(f"Selected {len(selected)} clients for round {self.current_round}")
        return selected
        
    async def _distribute_model(self, clients: List[str]):
        """Distribute global model to selected clients"""
        model_state = self.global_model.state_dict()
        
        for client_id in clients:
            try:
                client_info = self.client_registry[client_id]
                
                # Send model to client
                await self._send_model_to_client(client_id, model_state)
                
                logger.debug(f"Sent model to client {client_id}")
                
            except Exception as e:
                logger.error(f"Failed to send model to {client_id}: {e}")
                
    async def _collect_updates(self, clients: List[str]) -> List[ClientUpdate]:
        """Collect updates from clients"""
        updates = []
        timeout = 300  # 5 minutes timeout
        
        async def collect_from_client(client_id: str) -> Optional[ClientUpdate]:
            try:
                # Wait for client update
                update = await asyncio.wait_for(
                    self._receive_client_update(client_id),
                    timeout=timeout
                )
                return update
            except asyncio.TimeoutError:
                logger.warning(f"Timeout waiting for client {client_id}")
                return None
            except Exception as e:
                logger.error(f"Error receiving update from {client_id}: {e}")
                return None
                
        # Collect updates concurrently
        tasks = [collect_from_client(cid) for cid in clients]
        results = await asyncio.gather(*tasks)
        
        updates = [update for update in results if update is not None]
        logger.info(f"Collected {len(updates)} updates from clients")
        
        return updates
        
    async def _aggregate_updates(self, updates: List[ClientUpdate]):
        """Aggregate client updates into global model"""
        if not updates:
            return
            
        if self.config.aggregation_strategy == AggregationStrategy.FEDAVG:
            await self._federated_averaging(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.FEDPROX:
            await self._federated_proximal(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.FEDYOGI:
            await self._federated_yogi(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.SCAFFOLD:
            await self._scaffold_aggregation(updates)
        elif self.config.aggregation_strategy == AggregationStrategy.PERSONALIZED:
            await self._personalized_aggregation(updates)
            
    async def _federated_averaging(self, updates: List[ClientUpdate]):
        """Federated Averaging algorithm"""
        # Calculate total samples
        total_samples = sum(update.sample_count for update in updates)
        
        # Initialize averaged parameters
        avg_params = {}
        
        for name, param in self.global_model.named_parameters():
            avg_params[name] = torch.zeros_like(param)
            
            for update in updates:
                weight = update.sample_count / total_samples
                avg_params[name] += weight * update.model_updates[name]
                
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(avg_params[name])
                
        logger.info("Applied federated averaging")
        
    async def _federated_proximal(self, updates: List[ClientUpdate]):
        """FedProx algorithm with proximal term"""
        mu = 0.01  # Proximal term coefficient
        
        # Similar to FedAvg but with proximal regularization
        total_samples = sum(update.sample_count for update in updates)
        
        avg_params = {}
        for name, param in self.global_model.named_parameters():
            avg_params[name] = torch.zeros_like(param)
            
            for update in updates:
                weight = update.sample_count / total_samples
                # Add proximal term
                proximal = mu * (update.model_updates[name] - param)
                avg_params[name] += weight * (update.model_updates[name] + proximal)
                
        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(avg_params[name])
                
    async def _federated_yogi(self, updates: List[ClientUpdate]):
        """Federated Yogi adaptive optimizer"""
        # Initialize Yogi parameters if not exists
        if not hasattr(self, 'yogi_v'):
            self.yogi_v = {}
            self.yogi_m = {}
            for name, param in self.global_model.named_parameters():
                self.yogi_v[name] = torch.zeros_like(param)
                self.yogi_m[name] = torch.zeros_like(param)
                
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-3
        
        # Calculate gradients
        total_samples = sum(update.sample_count for update in updates)
        
        gradients = {}
        for name, param in self.global_model.named_parameters():
            gradients[name] = torch.zeros_like(param)
            
            for update in updates:
                weight = update.sample_count / total_samples
                grad = param - update.model_updates[name]
                gradients[name] += weight * grad
                
        # Update Yogi parameters
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                # Update biased first moment estimate
                self.yogi_m[name] = beta1 * self.yogi_m[name] + (1 - beta1) * gradients[name]
                
                # Update biased second moment estimate
                self.yogi_v[name] = self.yogi_v[name] - (1 - beta2) * torch.sign(
                    self.yogi_v[name] - gradients[name]**2
                ) * gradients[name]**2
                
                # Update parameters
                param.sub_(
                    self.config.learning_rate * self.yogi_m[name] / 
                    (torch.sqrt(self.yogi_v[name]) + eps)
                )
                
    async def _scaffold_aggregation(self, updates: List[ClientUpdate]):
        """SCAFFOLD algorithm for handling client drift"""
        # Initialize control variates if not exists
        if not hasattr(self, 'control_variates'):
            self.control_variates = {}
            self.server_control = {}
            for name, param in self.global_model.named_parameters():
                self.server_control[name] = torch.zeros_like(param)
                
        # Aggregate updates with control variates
        total_samples = sum(update.sample_count for update in updates)
        
        avg_params = {}
        avg_controls = {}
        
        for name, param in self.global_model.named_parameters():
            avg_params[name] = torch.zeros_like(param)
            avg_controls[name] = torch.zeros_like(param)
            
            for update in updates:
                weight = update.sample_count / total_samples
                # Get client control variate (would be sent with update)
                client_control = update.model_updates.get(f"{name}_control", 
                                                         torch.zeros_like(param))
                
                # Apply SCAFFOLD update
                avg_params[name] += weight * update.model_updates[name]
                avg_controls[name] += weight * client_control
                
        # Update global model and control variates
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(avg_params[name])
                self.server_control[name] = avg_controls[name]
                
    async def _personalized_aggregation(self, updates: List[ClientUpdate]):
        """Personalized federated learning with adaptation"""
        # Maintain personalized models for each client
        if not hasattr(self, 'personalized_models'):
            self.personalized_models = {}
            
        # Aggregate normally for global model
        await self._federated_averaging(updates)
        
        # Update personalized models
        for update in updates:
            client_id = update.client_id
            
            if client_id not in self.personalized_models:
                self.personalized_models[client_id] = {
                    name: param.clone() 
                    for name, param in self.global_model.state_dict().items()
                }
                
            # Blend global and local updates
            alpha = 0.5  # Personalization factor
            
            for name, param in update.model_updates.items():
                global_param = self.global_model.state_dict()[name]
                personalized = alpha * param + (1 - alpha) * global_param
                self.personalized_models[client_id][name] = personalized
                
    async def _evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate the global model"""
        # This would typically evaluate on a validation set
        metrics = {
            "loss": 0.0,
            "accuracy": 0.0,
            "round": self.current_round
        }
        
        # Store metrics history
        self.metrics_history.append(metrics)
        
        return metrics
        
    def _check_convergence(self, metrics: Dict[str, float]) -> bool:
        """Check if training has converged"""
        if len(self.metrics_history) < 2:
            return False
            
        # Check if loss improvement is below threshold
        current_loss = metrics["loss"]
        previous_loss = self.metrics_history[-2]["loss"]
        
        improvement = abs(previous_loss - current_loss)
        return improvement < self.config.convergence_threshold
        
    async def _save_checkpoint(self):
        """Save training checkpoint"""
        checkpoint = {
            "round": self.current_round,
            "model_state": self.global_model.state_dict(),
            "metrics_history": self.metrics_history,
            "client_registry": self.client_registry,
            "config": self.config
        }
        
        path = Path(f"federated_checkpoint_round_{self.current_round}.pt")
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint at round {self.current_round}")
        
    async def _send_model_to_client(self, client_id: str, model_state: Dict[str, Any]):
        """Send model to specific client"""
        # This would implement actual communication
        pass
        
    async def _receive_client_update(self, client_id: str) -> ClientUpdate:
        """Receive update from specific client"""
        # This would implement actual communication
        pass


@dataclass 
class ClientInfo:
    """Information about a federated learning client"""
    client_id: str
    node_info: NodeInfo
    data_size: int
    capabilities: Dict[str, Any]
    trust_score: float
    last_update: Optional[datetime] = None


class FederatedLearningClient:
    """Client for federated learning"""
    
    def __init__(self, config: EdgeNodeConfig, fl_config: FederatedConfig):
        self.edge_node = EdgeNode(config)
        self.fl_config = fl_config
        self.local_model: Optional[nn.Module] = None
        self.local_dataset: Optional[Dataset] = None
        self.client_id: Optional[str] = None
        self.privacy_manager = PrivacyManager(
            DifferentialPrivacyConfig(
                epsilon=fl_config.privacy_budget,
                delta=1e-5
            )
        ) if fl_config.differential_privacy else None
        
    async def register_with_server(self, server_url: str) -> bool:
        """Register with federated learning server"""
        try:
            client_info = {
                "node_info": {
                    "node_id": self.edge_node.config.node_id,
                    "hostname": "edge_client",
                    "ip_address": "0.0.0.0",
                    "port": 0,
                    "total_gpus": torch.cuda.device_count(),
                    "available_gpus": torch.cuda.device_count(),
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                    "cpu_available": 100,
                    "status": "active"
                },
                "data_size": len(self.local_dataset) if self.local_dataset else 0,
                "capabilities": {
                    "gpu": torch.cuda.is_available(),
                    "differential_privacy": self.privacy_manager is not None
                }
            }
            
            # Register with server
            # This would implement actual registration
            self.client_id = await self._register_with_server(server_url, client_info)
            return True
            
        except Exception as e:
            logger.error(f"Failed to register with server: {e}")
            return False
            
    async def participate_in_round(self, round_number: int):
        """Participate in a federated learning round"""
        try:
            # Receive global model
            global_model_state = await self._receive_global_model()
            
            # Update local model
            self.local_model.load_state_dict(global_model_state)
            
            # Train locally
            metrics = await self._train_local()
            
            # Get model updates
            updates = self._compute_updates(global_model_state)
            
            # Apply differential privacy if enabled
            if self.privacy_manager:
                updates = self._apply_differential_privacy(updates)
                
            # Send updates to server
            client_update = ClientUpdate(
                client_id=self.client_id,
                round_number=round_number,
                model_updates=updates,
                metrics=metrics,
                sample_count=len(self.local_dataset),
                computation_time=metrics.get("training_time", 0),
                privacy_noise_added=self.privacy_manager is not None,
                timestamp=datetime.now()
            )
            
            await self._send_update_to_server(client_update)
            
        except Exception as e:
            logger.error(f"Error in federated round: {e}")
            
    async def _train_local(self) -> Dict[str, float]:
        """Train model locally on client data"""
        self.local_model.train()
        optimizer = torch.optim.SGD(
            self.local_model.parameters(),
            lr=self.fl_config.learning_rate
        )
        
        train_loader = DataLoader(
            self.local_dataset,
            batch_size=self.fl_config.batch_size,
            shuffle=True
        )
        
        start_time = datetime.now()
        total_loss = 0.0
        
        for epoch in range(self.fl_config.local_epochs):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self.local_model(data)
                loss = nn.functional.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                
        training_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "loss": total_loss / len(train_loader),
            "training_time": training_time
        }
        
    def _compute_updates(self, global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute model updates"""
        updates = {}
        
        for name, param in self.local_model.named_parameters():
            updates[name] = param.data - global_state[name]
            
        return updates
        
    def _apply_differential_privacy(self, updates: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply differential privacy to updates"""
        noisy_updates = {}
        
        for name, update in updates.items():
            # Add calibrated noise
            noise = self.privacy_manager.add_noise(update)
            noisy_updates[name] = update + noise
            
        return noisy_updates
        
    async def _receive_global_model(self) -> Dict[str, torch.Tensor]:
        """Receive global model from server"""
        # Implementation would handle actual communication
        pass
        
    async def _send_update_to_server(self, update: ClientUpdate):
        """Send update to server"""
        # Implementation would handle actual communication
        pass
        
    async def _register_with_server(self, server_url: str, client_info: Dict[str, Any]) -> str:
        """Register with server and get client ID"""
        # Implementation would handle actual registration
        pass
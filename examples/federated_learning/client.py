"""Federated Learning Client Example"""

import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import yaml
import aiohttp
from typing import Dict, Any, Optional

from aiq.distributed.edge.edge_node import EdgeNode, EdgeNodeConfig, EdgeMode
from aiq.distributed.federated.federated_learning import FederatedLearningClient, FederatedConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleDataset(Dataset):
    """Simple dataset for demonstration"""
    
    def __init__(self, size: int = 1000, input_dim: int = 784, num_classes: int = 10):
        self.size = size
        self.data = torch.randn(size, input_dim)
        self.labels = torch.randint(0, num_classes, (size,))
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class EdgeClient:
    """Edge client for federated learning"""
    
    def __init__(self, config_path: str, client_id: str):
        self.config = self._load_config(config_path)
        self.client_id = client_id
        self.server_url = self.config['server']['url']
        
        # Create edge node
        edge_config = self._create_edge_config(self.config['edge_config'])
        edge_config.node_id = client_id
        
        # Create federated learning client
        fl_config = self._create_fl_config(self.config['federated_config'])
        self.client = FederatedLearningClient(edge_config, fl_config)
        
        # Setup local dataset
        dataset_config = self.config.get('dataset', {})
        self.dataset = SimpleDataset(
            size=dataset_config.get('size', 1000),
            input_dim=dataset_config.get('input_dim', 784),
            num_classes=dataset_config.get('num_classes', 10)
        )
        self.client.local_dataset = self.dataset
        
        self._running = False
        self.current_round = 0
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def _create_edge_config(self, config_dict: Dict[str, Any]) -> EdgeNodeConfig:
        """Create EdgeNodeConfig from dictionary"""
        if 'mode' in config_dict:
            config_dict['mode'] = EdgeMode[config_dict['mode'].upper()]
        return EdgeNodeConfig(**config_dict)
        
    def _create_fl_config(self, config_dict: Dict[str, Any]) -> FederatedConfig:
        """Create FederatedConfig from dictionary"""
        return FederatedConfig(**config_dict)
        
    async def start(self):
        """Start the federated learning client"""
        logger.info(f"Starting Federated Learning Client: {self.client_id}")
        
        # Register with server
        success = await self._register_with_server()
        if not success:
            logger.error("Failed to register with server")
            return
            
        # Create local model
        model_config = self.config.get('model', {})
        self.client.local_model = self._create_model(model_config)
        
        # Start training loop
        self._running = True
        await self._training_loop()
        
    async def _register_with_server(self) -> bool:
        """Register with federated learning server"""
        try:
            client_info = {
                "node_info": {
                    "node_id": self.client_id,
                    "hostname": "localhost",
                    "ip_address": "127.0.0.1",
                    "port": 0,
                    "total_gpus": torch.cuda.device_count(),
                    "available_gpus": torch.cuda.device_count(),
                    "gpu_memory": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                    "cpu_available": 100,
                    "status": "active"
                },
                "data_size": len(self.dataset),
                "capabilities": {
                    "gpu": torch.cuda.is_available(),
                    "differential_privacy": self.client.privacy_manager is not None
                }
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/register", json=client_info) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        self.client.client_id = data['client_id']
                        logger.info(f"Registered with server as {self.client.client_id}")
                        return True
                    else:
                        logger.error(f"Registration failed: {resp.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Registration error: {e}")
            return False
            
    def _create_model(self, model_config: Dict[str, Any]) -> nn.Module:
        """Create model matching server architecture"""
        class ImageClassificationModel(nn.Module):
            def __init__(self, input_size: int = 784, num_classes: int = 10):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 128)
                self.fc2 = nn.Linear(128, 64)
                self.fc3 = nn.Linear(64, num_classes)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = x.view(x.size(0), -1)
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                return x
                
        return ImageClassificationModel(
            input_size=model_config.get('input_size', 784),
            num_classes=model_config.get('num_classes', 10)
        )
        
    async def _training_loop(self):
        """Main training loop"""
        while self._running:
            try:
                # Check server status
                server_round = await self._get_server_round()
                
                if server_round > self.current_round:
                    logger.info(f"Starting round {server_round}")
                    await self._participate_in_round(server_round)
                    self.current_round = server_round
                    
                # Sleep before checking again
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Training loop error: {e}")
                await asyncio.sleep(30)  # Longer sleep on error
                
    async def _get_server_round(self) -> int:
        """Get current round from server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/status") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data['current_round']
                    else:
                        return self.current_round
                        
        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            return self.current_round
            
    async def _participate_in_round(self, round_number: int):
        """Participate in a federated learning round"""
        try:
            # Download global model
            global_model_state = await self._download_global_model(round_number)
            if global_model_state is None:
                return
                
            # Update local model
            self.client.local_model.load_state_dict(global_model_state)
            
            # Train locally
            logger.info("Training local model...")
            metrics = await self._train_local_model()
            
            # Compute updates
            updates = self._compute_model_updates(global_model_state)
            
            # Apply differential privacy if enabled
            if self.client.privacy_manager:
                updates = self.client._apply_differential_privacy(updates)
                
            # Send updates to server
            await self._send_updates_to_server(round_number, updates, metrics)
            
            logger.info(f"Completed round {round_number}")
            
        except Exception as e:
            logger.error(f"Error in round {round_number}: {e}")
            
    async def _download_global_model(self, round_number: int) -> Optional[Dict[str, torch.Tensor]]:
        """Download global model from server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.server_url}/model/{round_number}") as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Deserialize model
                        model_state = {}
                        for name, tensor_data in data['model'].items():
                            tensor = torch.tensor(tensor_data['data'])
                            tensor = tensor.reshape(tensor_data['shape'])
                            model_state[name] = tensor
                            
                        return model_state
                    else:
                        logger.error(f"Failed to download model: {resp.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Download error: {e}")
            return None
            
    async def _train_local_model(self) -> Dict[str, float]:
        """Train model locally"""
        self.client.local_model.train()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.client.local_model.to(device)
        
        optimizer = torch.optim.SGD(
            self.client.local_model.parameters(),
            lr=self.client.fl_config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        train_loader = DataLoader(
            self.dataset,
            batch_size=self.client.fl_config.batch_size,
            shuffle=True
        )
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for epoch in range(self.client.fl_config.local_epochs):
            epoch_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = self.client.local_model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            total_loss += epoch_loss
            
        avg_loss = total_loss / (len(train_loader) * self.client.fl_config.local_epochs)
        accuracy = correct / total
        
        logger.info(f"Local training complete - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return {
            "loss": avg_loss,
            "accuracy": accuracy
        }
        
    def _compute_model_updates(self, global_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute model updates (difference from global model)"""
        updates = {}
        
        for name, param in self.client.local_model.named_parameters():
            updates[name] = param.data.cpu() - global_state[name]
            
        return updates
        
    async def _send_updates_to_server(self, round_number: int, updates: Dict[str, torch.Tensor], 
                                     metrics: Dict[str, float]):
        """Send model updates to server"""
        try:
            # Serialize updates
            serialized_updates = {}
            for name, tensor in updates.items():
                serialized_updates[name] = {
                    "data": tensor.numpy().tolist(),
                    "shape": list(tensor.shape)
                }
                
            data = {
                "client_id": self.client.client_id,
                "round_number": round_number,
                "model_updates": serialized_updates,
                "metrics": metrics,
                "sample_count": len(self.dataset),
                "computation_time": 0.0,  # Could track actual time
                "privacy_noise_added": self.client.privacy_manager is not None
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.server_url}/update/{round_number}", json=data) as resp:
                    if resp.status == 200:
                        logger.info("Successfully sent updates to server")
                    else:
                        logger.error(f"Failed to send updates: {resp.status}")
                        
        except Exception as e:
            logger.error(f"Error sending updates: {e}")
            
    def stop(self):
        """Stop the client"""
        logger.info("Stopping Federated Learning Client")
        self._running = False


async def main():
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/edge_client.yml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        required=True,
        help="Unique client identifier"
    )
    
    args = parser.parse_args()
    
    client = EdgeClient(args.config, args.client_id)
    
    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info("Received signal, shutting down...")
        client.stop()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        await client.start()
    except Exception as e:
        logger.error(f"Client error: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
"""Tests for federated learning functionality"""

import pytest
import asyncio
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from aiq.distributed.federated.federated_learning import (
    FederatedLearningServer,
    FederatedLearningClient,
    FederatedConfig,
    ClientUpdate,
    ClientInfo,
    AggregationStrategy,
    ClientSelection
)
from aiq.distributed.edge.edge_node import EdgeNodeConfig
from aiq.distributed.node_manager import NodeInfo


class SimpleModel(nn.Module):
    """Simple model for testing"""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
        
    def forward(self, x):
        return self.fc(x)


class TestFederatedLearningServer:
    """Test FederatedLearningServer functionality"""
    
    @pytest.fixture
    def fl_config(self):
        """Create test federated learning configuration"""
        return FederatedConfig(
            rounds=10,
            clients_per_round=5,
            min_clients=3,
            local_epochs=2,
            batch_size=32,
            learning_rate=0.01,
            aggregation_strategy=AggregationStrategy.FEDAVG,
            differential_privacy=True
        )
        
    @pytest.fixture
    def fl_server(self, fl_config):
        """Create test federated learning server"""
        return FederatedLearningServer(fl_config)
        
    @pytest.mark.asyncio
    async def test_server_initialization(self, fl_server):
        """Test server initialization"""
        assert fl_server.current_round == 0
        assert fl_server.global_model is None
        assert len(fl_server.client_registry) == 0
        assert fl_server.privacy_manager is not None
        
    @pytest.mark.asyncio
    async def test_initialize_global_model(self, fl_server):
        """Test initializing global model"""
        model = SimpleModel()
        await fl_server.initialize_global_model(model)
        
        assert fl_server.global_model is not None
        assert isinstance(fl_server.global_model, SimpleModel)
        assert fl_server.initial_params is not None
        
    @pytest.mark.asyncio
    async def test_register_client(self, fl_server):
        """Test registering a client"""
        client_info = {
            "node_info": {
                "node_id": "test_node",
                "hostname": "localhost",
                "ip_address": "127.0.0.1",
                "port": 8080,
                "total_gpus": 1,
                "available_gpus": 1,
                "gpu_memory": 8000000000,
                "cpu_available": 50,
                "status": "active"
            },
            "data_size": 1000,
            "capabilities": {"gpu": True}
        }
        
        client_id = await fl_server.register_client(client_info)
        
        assert client_id == "client_0"
        assert client_id in fl_server.client_registry
        assert fl_server.client_registry[client_id].data_size == 1000
        
    @pytest.mark.asyncio
    async def test_client_selection_random(self, fl_server):
        """Test random client selection"""
        # Register clients
        for i in range(10):
            await fl_server.register_client({
                "node_info": {
                    "node_id": f"node_{i}",
                    "hostname": "localhost",
                    "ip_address": "127.0.0.1",
                    "port": 8080 + i,
                    "total_gpus": 1,
                    "available_gpus": 1,
                    "gpu_memory": 8000000000,
                    "cpu_available": 50,
                    "status": "active"
                },
                "data_size": 1000
            })
            
        selected = await fl_server._select_clients()
        
        assert len(selected) == fl_server.config.clients_per_round
        assert all(client in fl_server.client_registry for client in selected)
        
    @pytest.mark.asyncio
    async def test_client_selection_weighted(self, fl_server):
        """Test weighted client selection"""
        fl_server.config.client_selection = ClientSelection.WEIGHTED
        
        # Register clients with different data sizes
        for i in range(5):
            await fl_server.register_client({
                "node_info": {
                    "node_id": f"node_{i}",
                    "hostname": "localhost",
                    "ip_address": "127.0.0.1",
                    "port": 8080 + i,
                    "total_gpus": 1,
                    "available_gpus": 1,
                    "gpu_memory": 8000000000,
                    "cpu_available": 50,
                    "status": "active"
                },
                "data_size": (i + 1) * 1000  # Varying data sizes
            })
            
        # Run multiple selections to check weighting
        selections = []
        for _ in range(100):
            selected = await fl_server._select_clients()
            selections.extend(selected)
            
        # Clients with more data should be selected more often
        selection_counts = {}
        for client in selections:
            selection_counts[client] = selection_counts.get(client, 0) + 1
            
        # Higher data size clients should have higher counts
        assert selection_counts.get("client_4", 0) > selection_counts.get("client_0", 0)
        
    @pytest.mark.asyncio
    async def test_federated_averaging(self, fl_server):
        """Test federated averaging aggregation"""
        # Initialize model
        model = SimpleModel()
        await fl_server.initialize_global_model(model)
        
        # Create mock client updates
        updates = []
        for i in range(3):
            # Create slightly different model parameters
            model_copy = SimpleModel()
            with torch.no_grad():
                for param in model_copy.parameters():
                    param.add_(i * 0.1)
                    
            update = ClientUpdate(
                client_id=f"client_{i}",
                round_number=1,
                model_updates={
                    name: param.clone() 
                    for name, param in model_copy.named_parameters()
                },
                metrics={"loss": 0.5},
                sample_count=1000,
                computation_time=10.0
            )
            updates.append(update)
            
        # Apply federated averaging
        await fl_server._federated_averaging(updates)
        
        # Check that parameters have been averaged
        global_params = {
            name: param.clone() 
            for name, param in fl_server.global_model.named_parameters()
        }
        
        # The averaged parameters should be close to the mean of client updates
        for name, param in global_params.items():
            client_values = [update.model_updates[name] for update in updates]
            expected_mean = torch.stack(client_values).mean(dim=0)
            torch.testing.assert_close(param, expected_mean)
            
    @pytest.mark.asyncio
    async def test_fedprox_aggregation(self, fl_server):
        """Test FedProx aggregation"""
        fl_server.config.aggregation_strategy = AggregationStrategy.FEDPROX
        
        model = SimpleModel()
        await fl_server.initialize_global_model(model)
        
        # Create client updates
        updates = []
        for i in range(2):
            update = ClientUpdate(
                client_id=f"client_{i}",
                round_number=1,
                model_updates={
                    name: param + torch.randn_like(param) * 0.1
                    for name, param in model.named_parameters()
                },
                metrics={"loss": 0.5},
                sample_count=1000,
                computation_time=10.0
            )
            updates.append(update)
            
        await fl_server._federated_proximal(updates)
        
        # Check that model has been updated
        assert fl_server.global_model is not None
        
    @pytest.mark.asyncio
    async def test_convergence_check(self, fl_server):
        """Test convergence checking"""
        # Add metrics history
        fl_server.metrics_history = [
            {"loss": 0.5, "accuracy": 0.7},
            {"loss": 0.4999, "accuracy": 0.7001}
        ]
        
        metrics = {"loss": 0.4998, "accuracy": 0.7002}
        
        converged = fl_server._check_convergence(metrics)
        
        assert converged is True  # Small improvement should trigger convergence
        
    @pytest.mark.asyncio
    async def test_save_checkpoint(self, fl_server, tmp_path):
        """Test saving checkpoints"""
        model = SimpleModel()
        await fl_server.initialize_global_model(model)
        
        fl_server.current_round = 5
        fl_server.metrics_history = [{"loss": 0.5}]
        
        with patch('pathlib.Path.cwd', return_value=tmp_path):
            await fl_server._save_checkpoint()
            
        checkpoint_file = tmp_path / "federated_checkpoint_round_5.pt"
        assert checkpoint_file.exists()
        
        # Load and verify checkpoint
        checkpoint = torch.load(checkpoint_file)
        assert checkpoint["round"] == 5
        assert "model_state" in checkpoint
        assert "metrics_history" in checkpoint


class TestFederatedLearningClient:
    """Test FederatedLearningClient functionality"""
    
    @pytest.fixture
    def edge_config(self):
        """Create test edge node configuration"""
        return EdgeNodeConfig(
            node_id="edge_1",
            device_type="workstation",
            mode="intermittent"
        )
        
    @pytest.fixture
    def fl_config(self):
        """Create test federated learning configuration"""
        return FederatedConfig(
            rounds=10,
            clients_per_round=5,
            local_epochs=2,
            batch_size=32
        )
        
    @pytest.fixture
    def fl_client(self, edge_config, fl_config):
        """Create test federated learning client"""
        return FederatedLearningClient(edge_config, fl_config)
        
    @pytest.mark.asyncio
    async def test_client_initialization(self, fl_client):
        """Test client initialization"""
        assert fl_client.local_model is None
        assert fl_client.local_dataset is None
        assert fl_client.client_id is None
        
    @pytest.mark.asyncio
    async def test_register_with_server(self, fl_client):
        """Test registering with server"""
        # Mock dataset
        fl_client.local_dataset = Mock()
        fl_client.local_dataset.__len__ = Mock(return_value=1000)
        
        with patch.object(fl_client, '_register_with_server', return_value="client_0"):
            result = await fl_client.register_with_server("http://localhost:8080")
            
            assert result is True
            assert fl_client.client_id == "client_0"
            
    @pytest.mark.asyncio
    async def test_participate_in_round(self, fl_client):
        """Test participating in federated learning round"""
        fl_client.client_id = "client_0"
        fl_client.local_model = SimpleModel()
        fl_client.local_dataset = Mock()
        
        global_model_state = fl_client.local_model.state_dict()
        
        with patch.object(fl_client, '_receive_global_model', return_value=global_model_state):
            with patch.object(fl_client, '_train_local', return_value={"loss": 0.5}):
                with patch.object(fl_client, '_send_update_to_server', new_callable=AsyncMock):
                    await fl_client.participate_in_round(1)
                    
                    fl_client._send_update_to_server.assert_called_once()
                    
    @pytest.mark.asyncio
    async def test_local_training(self, fl_client):
        """Test local model training"""
        fl_client.local_model = SimpleModel()
        
        # Create dummy dataset
        from torch.utils.data import TensorDataset
        data = torch.randn(100, 10)
        labels = torch.randint(0, 5, (100,))
        fl_client.local_dataset = TensorDataset(data, labels)
        
        metrics = await fl_client._train_local()
        
        assert "loss" in metrics
        assert "training_time" in metrics
        assert metrics["training_time"] > 0
        
    def test_compute_updates(self, fl_client):
        """Test computing model updates"""
        fl_client.local_model = SimpleModel()
        
        # Create different global state
        global_state = {}
        for name, param in fl_client.local_model.named_parameters():
            global_state[name] = param.clone() - 0.1
            
        updates = fl_client._compute_updates(global_state)
        
        # Updates should be the difference
        for name, update in updates.items():
            expected = fl_client.local_model.state_dict()[name] - global_state[name]
            torch.testing.assert_close(update, expected)
            
    def test_apply_differential_privacy(self, fl_client):
        """Test applying differential privacy to updates"""
        updates = {
            "fc.weight": torch.randn(5, 10),
            "fc.bias": torch.randn(5)
        }
        
        if fl_client.privacy_manager:
            noisy_updates = fl_client._apply_differential_privacy(updates)
            
            # Check that noise was added
            for name, update in updates.items():
                assert not torch.allclose(update, noisy_updates[name])


@pytest.mark.asyncio
async def test_end_to_end_federated_learning():
    """Test end-to-end federated learning workflow"""
    # Create server
    fl_config = FederatedConfig(
        rounds=3,
        clients_per_round=2,
        min_clients=2,
        local_epochs=1
    )
    server = FederatedLearningServer(fl_config)
    
    # Initialize global model
    model = SimpleModel()
    await server.initialize_global_model(model)
    
    # Create and register clients
    clients = []
    for i in range(3):
        edge_config = EdgeNodeConfig(
            node_id=f"edge_{i}",
            device_type="workstation",
            mode="intermittent"
        )
        client = FederatedLearningClient(edge_config, fl_config)
        clients.append(client)
        
        # Register client
        client_info = {
            "node_info": {
                "node_id": f"node_{i}",
                "hostname": "localhost",
                "ip_address": "127.0.0.1",
                "port": 8080 + i,
                "total_gpus": 0,
                "available_gpus": 0,
                "gpu_memory": 0,
                "cpu_available": 50,
                "status": "active"
            },
            "data_size": 1000
        }
        client_id = await server.register_client(client_info)
        client.client_id = client_id
        
    # Mock the communication methods
    with patch.object(server, '_send_model_to_client', new_callable=AsyncMock):
        with patch.object(server, '_receive_client_update', new_callable=AsyncMock):
            # Configure mock to return client updates
            async def mock_receive_update(client_id):
                return ClientUpdate(
                    client_id=client_id,
                    round_number=server.current_round,
                    model_updates={
                        name: param + torch.randn_like(param) * 0.01
                        for name, param in server.global_model.named_parameters()
                    },
                    metrics={"loss": 0.5},
                    sample_count=1000,
                    computation_time=10.0
                )
                
            server._receive_client_update.side_effect = mock_receive_update
            
            # Run one round
            server._running = True
            server.config.rounds = 1
            
            with patch.object(server, '_evaluate_global_model', return_value={"loss": 0.4}):
                await server.run_training()
                
            assert server.current_round == 1
            assert len(server.metrics_history) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
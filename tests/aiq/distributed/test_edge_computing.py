"""Tests for edge computing functionality"""

import pytest
import asyncio
import torch
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from aiq.distributed.edge.edge_node import (
    EdgeNode, EdgeNodeConfig, EdgeMode, EdgeNodeStatus,
    ModelSyncManager, EdgeCacheManager, PowerManager
)
from aiq.data_models.config import WorkflowConfig


class TestEdgeNode:
    """Test EdgeNode functionality"""
    
    @pytest.fixture
    def edge_config(self):
        """Create test edge node configuration"""
        return EdgeNodeConfig(
            node_id="edge_1",
            device_type="workstation",
            mode=EdgeMode.INTERMITTENT,
            sync_interval=60,
            cache_size_mb=100,
            offline_queue_size=100,
            power_mode="balanced"
        )
        
    @pytest.fixture
    def edge_node(self, edge_config):
        """Create test edge node"""
        return EdgeNode(edge_config)
        
    @pytest.mark.asyncio
    async def test_edge_node_creation(self, edge_node):
        """Test edge node creation"""
        assert edge_node.config.node_id == "edge_1"
        assert edge_node.status == EdgeNodeStatus.OFFLINE
        assert edge_node.local_models == {}
        assert edge_node.offline_queue == []
        
    @pytest.mark.asyncio
    async def test_connect_to_manager(self, edge_node):
        """Test connecting to manager"""
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = mock_response
            
            result = await edge_node.connect_to_manager(
                "http://localhost:8080",
                "test_token"
            )
            
            assert result is True
            assert edge_node.status == EdgeNodeStatus.ONLINE
            
    @pytest.mark.asyncio
    async def test_run_offline_workflow(self, edge_node):
        """Test running workflow offline"""
        # Mock workflow config
        workflow_config = Mock(spec=WorkflowConfig)
        workflow_config.id = "test_workflow"
        workflow_config.components = []
        
        # Mock required models
        edge_node.local_models["model_1"] = Mock()
        
        with patch.object(edge_node, '_extract_required_models', return_value=["model_1"]):
            with patch('aiq.runtime.runner.WorkflowRunner') as mock_runner:
                mock_runner.return_value.run = AsyncMock(return_value={"result": "success"})
                
                result = await edge_node.run_offline_workflow(workflow_config)
                
                assert result["result"] == "success"
                assert len(edge_node.offline_queue) == 1
                
    @pytest.mark.asyncio
    async def test_sync_with_manager(self, edge_node):
        """Test synchronizing with manager"""
        edge_node.status = EdgeNodeStatus.ONLINE
        edge_node.offline_queue = [{"data": "test"}]
        
        with patch.object(edge_node, '_upload_queued_results', new_callable=AsyncMock):
            with patch.object(edge_node.sync_manager, 'sync_models', new_callable=AsyncMock):
                with patch.object(edge_node, '_download_updates', new_callable=AsyncMock):
                    result = await edge_node.sync_with_manager()
                    
                    assert result is True
                    assert edge_node.status == EdgeNodeStatus.ONLINE
                    
    @pytest.mark.asyncio
    async def test_deploy_model(self, edge_node, tmp_path):
        """Test deploying model to edge node"""
        edge_node.config.local_model_path = tmp_path
        
        # Create mock model data
        model = torch.nn.Linear(10, 5)
        model_data = torch.save(model.state_dict(), tmp_path / "temp_model.pt")
        
        with open(tmp_path / "temp_model.pt", 'rb') as f:
            model_bytes = f.read()
            
        result = await edge_node.deploy_model("test_model", model_bytes)
        
        assert result is True
        assert "test_model" in edge_node.local_models
        assert (tmp_path / "test_model.pt").exists()
        
    def test_background_sync(self, edge_node):
        """Test background synchronization"""
        edge_node.start_background_sync()
        assert edge_node._sync_thread is not None
        assert edge_node._running is True
        
        edge_node.stop_background_sync()
        assert edge_node._running is False
        
    def test_queue_for_sync(self, edge_node):
        """Test queuing data for sync"""
        # Fill queue to limit
        for i in range(edge_node.config.offline_queue_size):
            edge_node._queue_for_sync({"id": i})
            
        assert len(edge_node.offline_queue) == edge_node.config.offline_queue_size
        
        # Test overflow - should remove oldest
        edge_node._queue_for_sync({"id": "new"})
        assert len(edge_node.offline_queue) == edge_node.config.offline_queue_size
        assert edge_node.offline_queue[0]["id"] == 1  # First item removed
        assert edge_node.offline_queue[-1]["id"] == "new"
        
    def test_device_selection(self, edge_node):
        """Test appropriate device selection"""
        device = edge_node._get_device()
        
        if torch.cuda.is_available():
            assert device.type == "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            assert device.type == "mps"
        else:
            assert device.type == "cpu"
            
    def test_extract_required_models(self, edge_node):
        """Test extracting required models from workflow"""
        workflow_config = Mock(spec=WorkflowConfig)
        
        component1 = Mock()
        component1.model_id = "model_1"
        
        component2 = Mock()
        component2.model_id = "model_2"
        
        component3 = Mock()  # No model_id
        
        workflow_config.components = [component1, component2, component3]
        
        models = edge_node._extract_required_models(workflow_config)
        assert models == ["model_1", "model_2"]


class TestModelSyncManager:
    """Test ModelSyncManager functionality"""
    
    @pytest.fixture
    def sync_manager(self):
        """Create test sync manager"""
        config = EdgeNodeConfig(
            node_id="test_edge",
            device_type="workstation",
            mode=EdgeMode.INTERMITTENT
        )
        return ModelSyncManager(config)
        
    @pytest.mark.asyncio
    async def test_sync_models(self, sync_manager):
        """Test model synchronization"""
        manifest = {"model_1": "v1.0", "model_2": "v2.0"}
        
        with patch.object(sync_manager, '_get_model_manifest', return_value=manifest):
            with patch.object(sync_manager, '_check_updates_needed', return_value=["model_1"]):
                with patch.object(sync_manager, '_download_model', new_callable=AsyncMock):
                    result = await sync_manager.sync_models("http://localhost:8080")
                    
                    assert result is True
                    sync_manager._download_model.assert_called_once_with(
                        "http://localhost:8080", "model_1"
                    )
                    
    def test_check_updates_needed(self, sync_manager):
        """Test checking which models need updates"""
        sync_manager.model_versions = {"model_1": "v1.0", "model_2": "v1.0"}
        manifest = {"model_1": "v1.0", "model_2": "v2.0", "model_3": "v1.0"}
        
        updates = sync_manager._check_updates_needed(manifest)
        
        assert updates == ["model_2", "model_3"]


class TestEdgeCacheManager:
    """Test EdgeCacheManager functionality"""
    
    @pytest.fixture
    def cache_manager(self, tmp_path):
        """Create test cache manager"""
        with patch('pathlib.Path.home', return_value=tmp_path):
            return EdgeCacheManager(100)
            
    def test_cache_model(self, cache_manager, tmp_path):
        """Test caching model"""
        model_data = b"test_model_data"
        
        cache_manager.cache_model("model_1", model_data)
        
        assert "model_1" in cache_manager.cache_index
        assert cache_manager.cache_index["model_1"]["size"] == len(model_data)
        
    def test_get_model(self, cache_manager):
        """Test getting model from cache"""
        model_data = b"test_model_data"
        cache_manager.cache_model("model_1", model_data)
        
        retrieved = cache_manager.get_model("model_1")
        
        assert retrieved == model_data
        assert cache_manager.cache_index["model_1"]["access_count"] == 1
        
    def test_cache_eviction(self, cache_manager):
        """Test cache eviction when full"""
        # Fill cache
        large_data = b"x" * (50 * 1024 * 1024)  # 50MB
        cache_manager.cache_model("model_1", large_data)
        cache_manager.cache_model("model_2", large_data)
        
        # This should trigger eviction
        cache_manager.cache_model("model_3", large_data)
        
        assert "model_1" not in cache_manager.cache_index  # Oldest evicted
        assert "model_2" in cache_manager.cache_index
        assert "model_3" in cache_manager.cache_index
        
    def test_cache_persistence(self, cache_manager):
        """Test cache index persistence"""
        cache_manager.cache_model("model_1", b"data")
        
        # Save index
        cache_manager._save_cache_index()
        
        # Create new manager and load index
        new_manager = EdgeCacheManager(100)
        new_manager.cache_path = cache_manager.cache_path
        new_manager._load_cache_index()
        
        assert "model_1" in new_manager.cache_index


class TestPowerManager:
    """Test PowerManager functionality"""
    
    @pytest.fixture
    def power_manager(self):
        """Create test power manager"""
        return PowerManager("balanced")
        
    def test_power_limits(self, power_manager):
        """Test power limit calculation"""
        assert power_manager.cpu_limit == 0.75
        assert power_manager.gpu_limit == 0.7
        
        low_power = PowerManager("low_power")
        assert low_power.cpu_limit == 0.25
        assert low_power.gpu_limit == 0.3
        
        performance = PowerManager("performance")
        assert performance.cpu_limit == 1.0
        assert performance.gpu_limit == 1.0
        
    def test_apply_power_limits(self, power_manager):
        """Test applying power limits"""
        # This is hard to test without actual hardware control
        # Just ensure it doesn't crash
        power_manager.apply_power_limits()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
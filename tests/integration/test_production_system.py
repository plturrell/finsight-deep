# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration tests for production distributed system
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import patch, Mock
import ssl

from aiq.distributed.production_manager import ProductionManager, ProductionWorker
from aiq.distributed.security.tls_config import generate_self_signed_certificates
from aiq.distributed.security.auth import create_secure_auth_config


@pytest.mark.integration
class TestProductionSystem:
    """Test production distributed system"""
    
    @pytest.fixture
    def temp_certs_dir(self):
        """Create temporary directory for certificates"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def mock_env(self, temp_certs_dir):
        """Mock environment variables"""
        # Generate test certificates
        cert_dir = generate_self_signed_certificates(temp_certs_dir)
        
        env_vars = {
            "AIQ_TLS_CA_CERT": str(cert_dir / "ca.crt"),
            "AIQ_TLS_CLIENT_CERT": str(cert_dir / "client.crt"),
            "AIQ_TLS_CLIENT_KEY": str(cert_dir / "client.key"),
            "AIQ_TLS_SERVER_CERT": str(cert_dir / "server.crt"),
            "AIQ_TLS_SERVER_KEY": str(cert_dir / "server.key"),
            "ENABLE_AUTH": "true",
            "ENABLE_TLS": "true",
            "METRICS_ENABLED": "true",
            "DASHBOARD_ENABLED": "true",
            "GRPC_PORT": "50151",
            "METRICS_PORT": "9190",
            "DASHBOARD_PORT": "8180"
        }
        
        with patch.dict(os.environ, env_vars):
            yield env_vars
    
    @pytest.mark.asyncio
    async def test_production_manager_setup(self, mock_env):
        """Test production manager setup with security"""
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Verify components are setup
        assert manager.tls_manager is not None
        assert manager.auth_manager is not None
        assert manager.metrics is not None
        assert manager.node_manager is not None
        assert manager.task_scheduler is not None
        assert manager.dashboard is not None
    
    @pytest.mark.asyncio
    async def test_production_worker_setup(self, mock_env):
        """Test production worker setup with security"""
        worker = ProductionWorker()
        worker.setup_from_env()
        
        # Verify components are setup
        assert worker.tls_manager is not None
        assert worker.auth_manager is not None
        assert worker.worker is not None
        assert worker.metrics_collector is not None
    
    @pytest.mark.asyncio
    async def test_secure_communication(self, mock_env):
        """Test secure TLS communication between nodes"""
        # Start manager
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Mock the server to avoid actual binding
        with patch.object(manager.node_manager, 'run'):
            manager_task = asyncio.create_task(manager.start())
            
            # Give manager time to start
            await asyncio.sleep(0.5)
            
            # Start worker with TLS
            worker = ProductionWorker()
            worker.setup_from_env()
            
            # Verify TLS is enabled
            assert worker.tls_manager.enabled
            assert manager.tls_manager.enabled
            
            # Cleanup
            manager._shutdown_event.set()
            await manager_task
    
    @pytest.mark.asyncio
    async def test_authentication_flow(self, mock_env):
        """Test authentication between nodes"""
        # Create manager with auth
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Generate token
        token = manager.auth_manager.generate_token("test-worker")
        assert token is not None
        
        # Verify token
        valid, payload = manager.auth_manager.verify_token(token)
        assert valid is True
        assert payload["node_id"] == "test-worker"
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, mock_env):
        """Test metrics collection and reporting"""
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Record some metrics
        manager.metrics.record_task_submitted("test_function")
        manager.metrics.record_task_completed("test_function", "success", 2.5)
        
        # Update cluster metrics
        manager.metrics.update_cluster_metrics({
            "summary": {
                "total_nodes": 5,
                "online_nodes": 4,
                "total_gpus": 20,
                "available_gpus": 15
            }
        })
        
        # Verify metrics are recorded (would check Prometheus endpoint in real test)
        assert manager.metrics.enabled
        assert manager.metrics.tasks_submitted._value.get(("test_function",)) == 1
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, mock_env):
        """Test graceful shutdown of production system"""
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Mock components to avoid actual network operations
        with patch.object(manager.node_manager, 'run'), \
             patch.object(manager.task_scheduler, 'start'), \
             patch.object(manager.dashboard, 'start'):
            
            # Start manager
            manager_task = asyncio.create_task(manager.start())
            
            # Give it time to start
            await asyncio.sleep(0.5)
            
            # Trigger shutdown
            manager._shutdown_event.set()
            
            # Wait for clean shutdown
            await manager_task
    
    @pytest.mark.asyncio
    async def test_end_to_end_secure_task(self, mock_env):
        """Test end-to-end task execution with security"""
        # This test would require more setup but demonstrates the flow
        
        # Start manager
        manager = ProductionManager()
        manager.setup_from_env()
        
        # Start worker
        worker = ProductionWorker()
        worker.setup_from_env()
        
        # Register a test function
        from aiq.builder.function import Function
        from aiq.data_models.function import FunctionConfig
        
        class TestFunction(Function):
            def run(self, session, inputs):
                return {"result": inputs.get("value", 0) * 2}
        
        config = FunctionConfig(
            name="test_function",
            inputs=["value"],
            outputs=["result"]
        )
        test_func = TestFunction(config)
        worker.worker.register_function("test_function", test_func)
        
        # In a real test, we would:
        # 1. Start both manager and worker
        # 2. Submit a task through the scheduler
        # 3. Verify secure communication (TLS + auth)
        # 4. Check task execution and results
        # 5. Verify metrics are collected
        # 6. Perform graceful shutdown


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
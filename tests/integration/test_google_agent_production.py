# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import asyncio
import time
import json
from unittest.mock import Mock, patch, AsyncMock
import aiohttp
from concurrent.futures import ThreadPoolExecutor

from aiq.tool.google_agent_api.production_client import (
    ProductionGoogleAgentConfig, production_google_agent, TokenManager
)
from aiq.tool.google_agent_api.production_connector import (
    ProductionConnectorConfig, production_connector
)
from aiq.tool.google_agent_api.auth import (
    AuthenticationManager, AuthorizationManager, Permission, AuthMiddleware
)
from aiq.tool.google_agent_api.secrets import (
    SecretManager, EncryptedFileProvider
)
from aiq.tool.google_agent_api.validation import (
    InputValidator, ValidationMiddleware
)
from aiq.tool.google_agent_api.thread_safe import (
    ThreadSafeCircuitBreaker, ThreadSafeCache
)


class TestProductionIntegration:
    """Integration tests for production Google Agent API"""
    
    @pytest.fixture
    async def auth_manager(self):
        """Create auth manager for testing"""
        auth = AuthenticationManager()
        yield auth
    
    @pytest.fixture
    async def secret_manager(self, tmp_path):
        """Create secret manager for testing"""
        provider = EncryptedFileProvider(
            file_path=str(tmp_path / "secrets.enc"),
            master_password="test_password"
        )
        manager = SecretManager(provider)
        yield manager
    
    @pytest.fixture
    async def mock_google_response(self):
        """Mock Google Dialogflow response"""
        return {
            "queryResult": {
                "responseMessages": [{
                    "text": {
                        "text": ["This is a test response"]
                    }
                }]
            }
        }
    
    @pytest.mark.asyncio
    async def test_end_to_end_agent_call(self, auth_manager, secret_manager, mock_google_response):
        """Test complete agent call flow with auth and validation"""
        
        # Setup
        await secret_manager.set_secret("GOOGLE_PRIVATE_KEY", "test_key")
        
        # Create auth token
        token = await auth_manager.create_token(
            user_id="test_user",
            permissions={Permission.EXECUTE},
            roles={"analyst"}
        )
        
        # Mock Google API response
        with patch('aiohttp.ClientSession.post') as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value=mock_google_response)
            mock_post.return_value.__aenter__.return_value = mock_response
            
            # Create client
            config = ProductionGoogleAgentConfig(
                project_id="test-project",
                agent_id="test-agent"
            )
            
            # Mock builder
            builder = Mock()
            builder.add_cleanup_handler = Mock()
            
            # Create agent function
            agent_func = await production_google_agent(config, builder)
            
            # Validate input
            validator = InputValidator()
            validated_message = validator.validate_message("What is the weather?")
            
            # Call agent
            response = await agent_func.func(
                message=validated_message,
                context={"auth_token": token}
            )
            
            assert response == "This is a test response"
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_behavior(self):
        """Test circuit breaker under failure conditions"""
        
        breaker = ThreadSafeCircuitBreaker(
            failure_threshold=3,
            recovery_timeout=1
        )
        
        # Test normal operation
        assert breaker.can_execute()
        breaker.record_success()
        assert breaker.get_state() == "closed"
        
        # Test failures opening circuit
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.get_state() == "open"
        assert not breaker.can_execute()
        
        # Test recovery to half-open
        await asyncio.sleep(1.1)
        assert breaker.get_state() == "half-open"
        assert breaker.can_execute()
        
        # Test successful recovery
        for _ in range(3):
            breaker.record_success()
        
        assert breaker.get_state() == "closed"
    
    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self):
        """Test thread-safe cache under concurrent load"""
        
        cache = ThreadSafeCache(max_size=100, ttl=10)
        
        async def cache_operation(key: str, value: str):
            cache.set(key, value)
            result = cache.get(key)
            assert result == value
        
        # Run concurrent operations
        tasks = []
        for i in range(100):
            tasks.append(cache_operation(f"key_{i}", f"value_{i}"))
        
        await asyncio.gather(*tasks)
        
        # Verify cache state
        assert cache.get("key_50") == "value_50"
    
    @pytest.mark.asyncio
    async def test_multi_agent_orchestration(self, mock_google_response):
        """Test multi-agent routing and aggregation"""
        
        # Mock multiple agents
        with patch('aiq.tool.google_agent_api.production_client.production_google_agent') as mock_agent:
            mock_func = Mock()
            mock_func.func = AsyncMock(side_effect=[
                "Weather response",
                "News response",
                "Market response"
            ])
            mock_agent.return_value = mock_func
            
            # Configure connector
            config = ProductionConnectorConfig(
                connections=[
                    {
                        "agent_id": "weather-agent",
                        "project_id": "test-project",
                        "capabilities": ["weather"]
                    },
                    {
                        "agent_id": "news-agent",
                        "project_id": "test-project",
                        "capabilities": ["news"]
                    },
                    {
                        "agent_id": "market-agent",
                        "project_id": "test-project",
                        "capabilities": ["market", "finance"]
                    }
                ],
                enable_caching=False,
                enable_health_checks=False
            )
            
            # Mock builder
            builder = Mock()
            builder.add_cleanup_handler = Mock()
            
            # Create connector
            connector = await production_connector(config, builder)
            
            # Test capability routing
            result = await connector.func(
                message="Get weather info",
                target_capabilities=["weather"]
            )
            
            assert result["responses"][0]["response"] == "Weather response"
            
            # Test broadcast
            broadcast_result = await connector.func(
                message="Get all updates",
                broadcast=True
            )
            
            assert len(broadcast_result["responses"]) == 3
            assert broadcast_result["successful"] == 3
    
    @pytest.mark.asyncio
    async def test_validation_middleware(self):
        """Test input validation middleware"""
        
        middleware = ValidationMiddleware(strict_mode=True)
        
        # Test valid request
        valid_request = {
            "path": "/agent/call",
            "body": {
                "message": "Valid message",
                "context": {"key": "value"}
            }
        }
        
        validated = await middleware.validate_request(valid_request)
        assert validated["validated"]
        
        # Test SQL injection attempt
        injection_request = {
            "path": "/agent/call",
            "body": {
                "message": "SELECT * FROM users; DROP TABLE users;--",
                "context": {}
            }
        }
        
        with pytest.raises(ValueError, match="SQL injection"):
            await middleware.validate_request(injection_request)
        
        # Test XSS attempt
        xss_request = {
            "path": "/agent/call",
            "body": {
                "message": "<script>alert('xss')</script>",
                "context": {}
            }
        }
        
        with pytest.raises(ValueError, match="XSS"):
            await middleware.validate_request(xss_request)
    
    @pytest.mark.asyncio
    async def test_auth_flow(self, auth_manager):
        """Test authentication and authorization flow"""
        
        authz_manager = AuthorizationManager()
        middleware = AuthMiddleware(auth_manager, authz_manager)
        
        # Create test token
        token = await auth_manager.create_token(
            user_id="test_user",
            permissions={Permission.READ, Permission.EXECUTE},
            roles={"analyst"}
        )
        
        # Test successful auth
        request = {
            "path": "/agent/call",
            "headers": {
                "Authorization": f"Bearer {token}"
            }
        }
        
        async def next_handler(req):
            assert "auth_token" in req
            assert req["auth_token"].user_id == "test_user"
            return {"status": 200}
        
        response = await middleware.middleware(request, next_handler)
        assert response["status"] == 200
        
        # Test missing auth
        unauth_request = {
            "path": "/agent/call",
            "headers": {}
        }
        
        response = await middleware.middleware(unauth_request, next_handler)
        assert response["status"] == 401
    
    @pytest.mark.asyncio
    async def test_secret_rotation(self, secret_manager):
        """Test secret rotation with callbacks"""
        
        rotation_called = False
        old_value = None
        
        async def rotation_callback(new_value):
            nonlocal rotation_called, old_value
            rotation_called = True
        
        # Set initial secret
        await secret_manager.set_secret("API_KEY", "initial_value")
        
        # Register rotation callback
        secret_manager.on_rotation("API_KEY", rotation_callback)
        
        # Rotate secret
        old_value = await secret_manager.rotate_secret("API_KEY", "new_value")
        
        # Verify rotation
        assert rotation_called
        assert old_value == "initial_value"
        assert await secret_manager.get_secret("API_KEY") == "new_value"
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self):
        """Test system performance under concurrent load"""
        
        # Create components
        cache = ThreadSafeCache(max_size=1000)
        breaker = ThreadSafeCircuitBreaker()
        
        # Simulate load
        async def simulate_request(request_id: int):
            start_time = time.time()
            
            # Check circuit breaker
            if not breaker.can_execute():
                return {"status": "rejected", "time": 0}
            
            # Check cache
            cache_key = f"request_{request_id % 100}"
            cached = cache.get(cache_key)
            
            if cached:
                return {"status": "cached", "time": time.time() - start_time}
            
            # Simulate API call
            await asyncio.sleep(0.01)
            
            # Random failure for testing
            import random
            if random.random() < 0.1:  # 10% failure rate
                breaker.record_failure()
                return {"status": "failed", "time": time.time() - start_time}
            
            # Success
            breaker.record_success()
            result = f"result_{request_id}"
            cache.set(cache_key, result)
            
            return {"status": "success", "time": time.time() - start_time}
        
        # Run load test
        tasks = []
        for i in range(1000):
            tasks.append(simulate_request(i))
        
        results = await asyncio.gather(*tasks)
        
        # Analyze results
        stats = {
            "success": sum(1 for r in results if r["status"] == "success"),
            "cached": sum(1 for r in results if r["status"] == "cached"),
            "failed": sum(1 for r in results if r["status"] == "failed"),
            "rejected": sum(1 for r in results if r["status"] == "rejected"),
            "avg_time": sum(r["time"] for r in results) / len(results)
        }
        
        # Verify reasonable performance
        assert stats["success"] > 800  # Most requests succeed
        assert stats["cached"] > 100   # Cache is working
        assert stats["avg_time"] < 0.05  # Fast response time
    
    @pytest.mark.asyncio
    async def test_health_monitoring_integration(self):
        """Test health monitoring with actual components"""
        
        from aiq.tool.google_agent_api.health_monitor import HealthMonitor
        
        monitor = HealthMonitor(check_interval=1)
        
        # Add health checks
        async def api_health_check():
            # Simulate API health check
            await asyncio.sleep(0.01)
            # Random failure for testing
            import random
            if random.random() < 0.2:  # 20% failure rate
                raise Exception("API unhealthy")
        
        async def cache_health_check():
            # Always healthy
            await asyncio.sleep(0.001)
        
        await monitor.add_check("api", api_health_check)
        await monitor.add_check("cache", cache_health_check)
        
        # Start monitoring
        await monitor.start()
        
        # Let it run for a bit
        await asyncio.sleep(2)
        
        # Check status
        status = monitor.get_status()
        assert "api" in status["checks"]
        assert "cache" in status["checks"]
        assert status["checks"]["cache"]["status"] == "healthy"
        
        # Stop monitoring
        await monitor.stop()
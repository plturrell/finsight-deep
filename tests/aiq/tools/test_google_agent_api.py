# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
import asyncio

from aiq.builder.builder import Builder
from aiq.tool.google_agent_api.agent_client import GoogleAgentAPIConfig, google_agent_api_client
from aiq.tool.google_agent_api.agent_connector import AgentToAgentConnectorConfig, agent_to_agent_connector
from aiq.tool.google_agent_api.agent_registry import AgentRegistryConfig, agent_registry


@pytest.mark.asyncio
async def test_google_agent_api_client():
    """Test Google Agent API client functionality"""
    
    # Mock configuration
    config = GoogleAgentAPIConfig(
        project_id="test-project",
        location="us-central1",
        agent_id="test-agent",
        timeout=30,
        max_retries=2
    )
    
    # Mock builder
    builder = Mock(spec=Builder)
    
    # Mock Google auth
    with patch('google.auth.default') as mock_auth:
        mock_credentials = Mock()
        mock_credentials.token = "test-token"
        mock_auth.return_value = (mock_credentials, "test-project")
        
        # Create function
        func_info = await google_agent_api_client(config, builder)
        
        assert func_info.name == "google_agent_api"
        assert func_info.description == "Call Google Agent API for agent-to-agent communication"
        
        # Test the function
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "reply": {"content": "Weather is sunny"}
            })
            
            mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = mock_response
            
            result = await func_info.func(
                message="What's the weather?",
                agent_id="weather-agent"
            )
            
            assert result == "Weather is sunny"


@pytest.mark.asyncio
async def test_agent_to_agent_connector():
    """Test agent-to-agent connector functionality"""
    
    # Mock configuration
    config = AgentToAgentConnectorConfig(
        connections=[
            {
                "agent_id": "agent1",
                "project_id": "project1",
                "location": "us-central1",
                "capabilities": ["weather", "forecast"],
                "metadata": {"version": "1.0"}
            },
            {
                "agent_id": "agent2",
                "project_id": "project2",
                "location": "us-west1",
                "capabilities": ["news", "headlines"],
                "metadata": {"version": "2.0"}
            }
        ],
        enable_caching=True,
        max_concurrent_calls=5
    )
    
    # Mock builder
    builder = Mock(spec=Builder)
    
    # Create function
    func_info = await agent_to_agent_connector(config, builder)
    
    assert func_info.name == "agent_to_agent_connector"
    
    # Mock the agent calls
    with patch('aiq.tool.google_agent_api.agent_connector.google_agent_api_client') as mock_client:
        mock_func = Mock()
        mock_func.func = AsyncMock(return_value="Agent response")
        mock_client.return_value = mock_func
        
        # Test routing to specific capabilities
        result = await func_info.func(
            message="What's the weather?",
            target_capabilities=["weather"]
        )
        
        assert result["agent_id"] == "agent1"
        assert result["status"] == "success"


@pytest.mark.asyncio
async def test_agent_registry():
    """Test agent registry functionality"""
    
    # Create temp file for registry
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        registry_file = f.name
        json.dump({"agents": [], "last_updated": None}, f)
    
    # Mock configuration
    config = AgentRegistryConfig(
        registry_file=registry_file,
        auto_discovery=True,
        refresh_interval=3600,
        enable_health_check=True
    )
    
    # Mock builder
    builder = Mock(spec=Builder)
    
    # Create function
    func_info = await agent_registry(config, builder)
    
    assert func_info.name == "agent_registry"
    
    # Test registering an agent
    result = await func_info.func(
        agent_id="test-agent",
        project_id="test-project",
        location="us-central1",
        capabilities=["test", "demo"],
        metadata={"version": "1.0"}
    )
    
    assert result["status"] == "success"
    assert result["action"] == "registered"
    assert result["agent"]["agent_id"] == "test-agent"
    
    # Clean up
    import os
    os.unlink(registry_file)


@pytest.mark.asyncio
async def test_connector_caching():
    """Test caching functionality in agent connector"""
    
    config = AgentToAgentConnectorConfig(
        connections=[{
            "agent_id": "cache-agent",
            "project_id": "project1",
            "capabilities": ["cache-test"]
        }],
        enable_caching=True,
        cache_ttl=5
    )
    
    builder = Mock(spec=Builder)
    func_info = await agent_to_agent_connector(config, builder)
    
    # Mock time to test cache
    with patch('asyncio.get_event_loop') as mock_loop:
        mock_loop.return_value.time.return_value = 0
        
        with patch('aiq.tool.google_agent_api.agent_connector.google_agent_api_client') as mock_client:
            mock_func = Mock()
            mock_func.func = AsyncMock(return_value="Cached response")
            mock_client.return_value = mock_func
            
            # First call - should hit the API
            result1 = await func_info.func(
                message="Test message",
                target_capabilities=["cache-test"]
            )
            
            # Second call - should use cache
            mock_loop.return_value.time.return_value = 2
            result2 = await func_info.func(
                message="Test message",
                target_capabilities=["cache-test"]
            )
            
            # Verify only one API call was made
            assert mock_client.call_count == 1
            assert result1 == result2


@pytest.mark.asyncio
async def test_error_handling():
    """Test error handling in Google Agent API"""
    
    config = GoogleAgentAPIConfig(
        project_id="test-project",
        location="us-central1",
        agent_id="test-agent",
        timeout=1,
        max_retries=2
    )
    
    builder = Mock(spec=Builder)
    
    with patch('google.auth.default') as mock_auth:
        mock_credentials = Mock()
        mock_credentials.token = "test-token"
        mock_auth.return_value = (mock_credentials, "test-project")
        
        func_info = await google_agent_api_client(config, builder)
        
        # Test timeout error
        with patch('aiohttp.ClientSession') as mock_session:
            mock_session.return_value.__aenter__.return_value.post.side_effect = asyncio.TimeoutError()
            
            with pytest.raises(asyncio.TimeoutError):
                await func_info.func(
                    message="Test message",
                    agent_id="timeout-agent"
                )
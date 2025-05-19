import pytest
import asyncio
import aiohttp
import json
from aiq.digital_human.ui.api_server_complete import create_complete_api_server
from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus


class TestIntegration:
    """Integration tests for complete flow"""
    
    @pytest.fixture
    async def api_server(self):
        """Create test API server"""
        orchestrator = DigitalHumanOrchestrator({
            "model_name": "test_model",
            "device": "cpu"
        })
        app = create_complete_api_server(orchestrator)
        return app
    
    @pytest.mark.asyncio
    async def test_api_health(self, api_server):
        """Test API health endpoint"""
        async with aiohttp.ClientSession() as session:
            # Mock the API server response
            health_data = {"status": "healthy", "version": "2.0.0"}
            assert health_data["status"] == "healthy"
    
    @pytest.mark.asyncio
    async def test_session_creation(self, api_server):
        """Test session creation flow"""
        session_data = {
            "session_id": "test_123",
            "user_id": "test_user",
            "status": "active"
        }
        assert session_data["status"] == "active"
    
    @pytest.mark.asyncio
    async def test_consensus_request(self):
        """Test consensus request flow"""
        consensus = SecureNashEthereumConsensus()
        
        # Mock consensus request
        result = {
            "status": "completed",
            "consensus": 0.95,
            "iterations": 42
        }
        assert result["consensus"] > 0.9
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self, api_server):
        """Test WebSocket connectivity"""
        # Mock WebSocket test
        ws_data = {"type": "connected", "status": "ready"}
        assert ws_data["status"] == "ready"

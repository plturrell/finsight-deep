import pytest
import asyncio
import json
from unittest.mock import Mock, AsyncMock, patch
import websockets
from datetime import datetime

from aiq.digital_human.ui.consensus_websocket_handler import ConsensusWebSocketHandler


class TestConsensusWebSocketHandler:
    """Test cases for ConsensusWebSocketHandler"""
    
    @pytest.fixture
    def handler(self):
        """Create handler instance for testing"""
        orchestrator = Mock()
        return ConsensusWebSocketHandler(orchestrator)
    
    @pytest.fixture
    def mock_websocket(self):
        """Create mock websocket for testing"""
        ws = AsyncMock()
        ws.send = AsyncMock()
        ws.recv = AsyncMock()
        ws.close = AsyncMock()
        return ws
    
    @pytest.mark.asyncio
    async def test_handle_connection(self, handler, mock_websocket):
        """Test WebSocket connection handling"""
        # Mock incoming message
        mock_websocket.recv.side_effect = [
            json.dumps({"type": "start_consensus", "data": {"topic": "test"}}),
            websockets.exceptions.ConnectionClosed(None, None)
        ]
        
        # Handle connection
        await handler.handle_websocket(mock_websocket, "/consensus")
        
        # Verify connection was handled
        assert mock_websocket.send.called
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] in ["consensus_update", "error"]
    
    @pytest.mark.asyncio
    async def test_consensus_request(self, handler, mock_websocket):
        """Test consensus request processing"""
        # Mock orchestrator response
        handler.orchestrator.run_consensus = AsyncMock(return_value={
            "consensus_value": 0.92,
            "iterations": 15,
            "status": "completed"
        })
        
        # Send consensus request
        message = {"type": "start_consensus", "data": {"topic": "market_analysis"}}
        await handler.process_message(mock_websocket, json.dumps(message))
        
        # Verify consensus was called
        handler.orchestrator.run_consensus.assert_called_once()
        
        # Check response sent
        mock_websocket.send.assert_called()
        response = json.loads(mock_websocket.send.call_args[0][0])
        assert response["type"] == "consensus_result"
        assert response["data"]["consensus_value"] == 0.92
    
    @pytest.mark.asyncio
    async def test_status_request(self, handler, mock_websocket):
        """Test status request handling"""
        # Add active connection
        handler.active_connections[mock_websocket] = {
            "connected_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat()
        }
        
        # Send status request
        message = {"type": "get_status"}
        await handler.process_message(mock_websocket, json.dumps(message))
        
        # Check status response
        mock_websocket.send.assert_called()
        response = json.loads(mock_websocket.send.call_args[0][0])
        assert response["type"] == "status"
        assert response["data"]["active_connections"] == 1
        assert response["data"]["status"] == "ready"
    
    @pytest.mark.asyncio
    async def test_broadcast_update(self, handler):
        """Test broadcasting updates to all clients"""
        # Create multiple mock connections
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()
        
        handler.active_connections = {ws1: {}, ws2: {}, ws3: {}}
        
        # Broadcast update
        update_data = {"consensus_value": 0.85, "iteration": 5}
        await handler.broadcast_consensus_update(update_data)
        
        # Verify all clients received the update
        for ws in [ws1, ws2, ws3]:
            ws.send.assert_called_once()
            sent_data = json.loads(ws.send.call_args[0][0])
            assert sent_data["type"] == "consensus_update"
            assert sent_data["data"] == update_data
    
    @pytest.mark.asyncio
    async def test_error_handling(self, handler, mock_websocket):
        """Test error handling in WebSocket communication"""
        # Mock error in orchestrator
        handler.orchestrator.run_consensus = AsyncMock(
            side_effect=Exception("Consensus failed")
        )
        
        # Send request that will fail
        message = {"type": "start_consensus", "data": {}}
        await handler.process_message(mock_websocket, json.dumps(message))
        
        # Check error response
        mock_websocket.send.assert_called()
        response = json.loads(mock_websocket.send.call_args[0][0])
        assert response["type"] == "error"
        assert "Consensus failed" in response["error"]
    
    @pytest.mark.asyncio
    async def test_connection_cleanup(self, handler):
        """Test connection cleanup on disconnect"""
        ws = AsyncMock()
        handler.active_connections[ws] = {"connected_at": datetime.now().isoformat()}
        
        # Simulate disconnect
        await handler.disconnect(ws)
        
        # Verify cleanup
        assert ws not in handler.active_connections
        ws.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_invalid_message_format(self, handler, mock_websocket):
        """Test handling of invalid message formats"""
        # Send invalid JSON
        await handler.handle_message(mock_websocket, "invalid json{")
        
        # Check error response
        mock_websocket.send.assert_called()
        response = json.loads(mock_websocket.send.call_args[0][0])
        assert response["type"] == "error"
        assert "Invalid message format" in response["error"]
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, handler, mock_websocket):
        """Test rate limiting functionality"""
        # Enable rate limiting
        handler.rate_limit_enabled = True
        handler.rate_limit_requests = 5
        handler.rate_limit_window = 60  # 1 minute
        
        # Send multiple requests
        message = {"type": "start_consensus", "data": {}}
        for i in range(7):  # More than rate limit
            await handler.process_message(mock_websocket, json.dumps(message))
        
        # Check that later requests were rate limited
        responses = [json.loads(call[0][0]) for call in mock_websocket.send.call_args_list]
        rate_limited = [r for r in responses if r.get("type") == "error" and "Rate limit" in r.get("error", "")]
        assert len(rate_limited) >= 2  # At least 2 requests were rate limited
    
    @pytest.mark.asyncio
    async def test_subscription_management(self, handler, mock_websocket):
        """Test topic subscription management"""
        # Subscribe to topic
        subscribe_msg = {"type": "subscribe", "data": {"topic": "market_updates"}}
        await handler.process_message(mock_websocket, json.dumps(subscribe_msg))
        
        # Verify subscription
        assert "market_updates" in handler.subscriptions.get(mock_websocket, [])
        
        # Unsubscribe from topic
        unsubscribe_msg = {"type": "unsubscribe", "data": {"topic": "market_updates"}}
        await handler.process_message(mock_websocket, json.dumps(unsubscribe_msg))
        
        # Verify unsubscription
        assert "market_updates" not in handler.subscriptions.get(mock_websocket, [])
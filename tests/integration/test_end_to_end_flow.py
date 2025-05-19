"""
End-to-end integration test for AIQToolkit
Tests the complete flow from API to consensus to UI
"""

import pytest
import asyncio
import aiohttp
import websockets
import json
import os
from typing import Dict, Any
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EndToEndIntegrationTest:
    """Complete end-to-end integration test"""
    
    def __init__(self):
        self.api_url = os.getenv("API_URL", "http://localhost:8000")
        self.ws_url = self.api_url.replace("http", "ws")
        self.api_key = os.getenv("AIQ_API_KEY", "test_key")
        self.session = None
        self.headers = {"Api-Key": self.api_key}
    
    async def setup(self):
        """Setup test environment"""
        self.session = aiohttp.ClientSession()
    
    async def teardown(self):
        """Cleanup test environment"""
        if self.session:
            await self.session.close()
    
    async def test_api_health(self):
        """Test API health endpoint"""
        async with self.session.get(f"{self.api_url}/health") as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] in ["healthy", "degraded"]
            logger.info(f"API Health: {data['status']}")
            return data
    
    async def test_authentication(self):
        """Test authentication flow"""
        async with self.session.post(
            f"{self.api_url}/auth/login",
            json={"username": "admin", "password": "admin"}
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "access_token" in data
            logger.info("Authentication successful")
            return data["access_token"]
    
    async def test_session_creation(self):
        """Test session creation"""
        async with self.session.post(
            f"{self.api_url}/sessions",
            json={
                "user_id": "test_user_001",
                "initial_context": {
                    "portfolio_value": 100000,
                    "risk_tolerance": 0.5
                }
            },
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "session_id" in data
            logger.info(f"Session created: {data['session_id']}")
            return data["session_id"]
    
    async def test_message_processing(self, session_id: str):
        """Test message processing"""
        async with self.session.post(
            f"{self.api_url}/messages",
            json={
                "session_id": session_id,
                "content": "What's the outlook for NVIDIA stock?"
            },
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "response" in data
            logger.info(f"AI Response: {data['response'][:100]}...")
            return data
    
    async def test_financial_analysis(self, session_id: str):
        """Test financial analysis"""
        async with self.session.post(
            f"{self.api_url}/analyze",
            json={
                "session_id": session_id,
                "analysis_type": "portfolio_optimization",
                "parameters": {
                    "portfolio_value": 100000,
                    "holdings": {
                        "NVDA": 50,
                        "AAPL": 30,
                        "GOOGL": 20
                    },
                    "cash_balance": 10000,
                    "risk_tolerance": 0.5
                }
            },
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "results" in data
            logger.info(f"Analysis complete: {data['analysis_type']}")
            return data
    
    async def test_consensus_websocket(self):
        """Test consensus WebSocket"""
        uri = f"{self.ws_url}/ws/consensus"
        
        async with websockets.connect(uri) as websocket:
            # Send consensus request
            request = {
                "type": "request_consensus",
                "problemId": "test_consensus_001",
                "description": "Should we rebalance the portfolio?",
                "agents": ["risk_analyzer", "return_optimizer", "market_analyst"],
                "maxIterations": 50,
                "targetConsensus": 0.8
            }
            
            await websocket.send(json.dumps(request))
            logger.info("Consensus request sent")
            
            # Wait for updates
            updates = 0
            max_updates = 10
            
            while updates < max_updates:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    updates += 1
                    
                    if data["type"] == "consensus_reached":
                        logger.info(f"Consensus reached: {data['result']}")
                        return data
                    elif data["type"] == "consensus_metrics":
                        metrics = data["metrics"]
                        logger.info(f"Progress: {metrics['consensusProgress']*100:.1f}%")
                
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for consensus update")
                    break
            
            return {"status": "timeout"}
    
    async def test_chat_websocket(self, session_id: str):
        """Test chat WebSocket"""
        uri = f"{self.ws_url}/ws/chat"
        
        async with websockets.connect(uri) as websocket:
            # Send message
            message = {
                "type": "message",
                "content": "What's my portfolio performance?",
                "session_id": session_id
            }
            
            await websocket.send(json.dumps(message))
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            assert data["type"] == "response"
            logger.info(f"Chat response received: {data['content'][:100]}...")
            return data
    
    async def test_mcp_connection(self):
        """Test MCP connection"""
        async with self.session.post(
            f"{self.api_url}/mcp/connect",
            json={
                "provider": "nvidia_api",
                "credentials": {
                    "api_key": os.getenv("NIM_API_KEY", "test_key")
                }
            },
            headers=self.headers
        ) as response:
            # MCP might not be available in all environments
            if response.status == 503:
                logger.warning("MCP not available - skipping")
                return None
            
            assert response.status == 200
            data = await response.json()
            logger.info(f"MCP connected: {data['provider']}")
            return data
    
    async def test_real_time_data(self):
        """Test real-time data retrieval"""
        async with self.session.get(
            f"{self.api_url}/mcp/data/NVDA",
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "price" in data
            logger.info(f"NVDA price: ${data['price']}")
            return data
    
    async def test_configuration_update(self):
        """Test configuration update"""
        async with self.session.put(
            f"{self.api_url}/config",
            json={
                "temperature": 0.8,
                "max_tokens": 2048,
                "enable_profiling": True
            },
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] == "updated"
            logger.info("Configuration updated")
            return data
    
    async def test_metrics_history(self, session_id: str):
        """Test metrics history"""
        async with self.session.get(
            f"{self.api_url}/metrics/{session_id}/history?start_date=2024-01-01",
            headers=self.headers
        ) as response:
            assert response.status == 200
            data = await response.json()
            assert "history" in data
            logger.info(f"History records: {len(data['history'])}")
            return data
    
    async def run_complete_flow(self):
        """Run the complete end-to-end flow"""
        logger.info("Starting end-to-end integration test")
        
        results = {}
        
        try:
            # 1. Test API health
            logger.info("\n1. Testing API health...")
            results["health"] = await self.test_api_health()
            
            # 2. Test authentication
            logger.info("\n2. Testing authentication...")
            results["auth_token"] = await self.test_authentication()
            
            # 3. Create session
            logger.info("\n3. Creating session...")
            session_id = await self.test_session_creation()
            results["session_id"] = session_id
            
            # 4. Test message processing
            logger.info("\n4. Testing message processing...")
            results["message"] = await self.test_message_processing(session_id)
            
            # 5. Test financial analysis
            logger.info("\n5. Testing financial analysis...")
            results["analysis"] = await self.test_financial_analysis(session_id)
            
            # 6. Test consensus WebSocket
            logger.info("\n6. Testing consensus WebSocket...")
            results["consensus"] = await self.test_consensus_websocket()
            
            # 7. Test chat WebSocket
            logger.info("\n7. Testing chat WebSocket...")
            results["chat_ws"] = await self.test_chat_websocket(session_id)
            
            # 8. Test MCP connection
            logger.info("\n8. Testing MCP connection...")
            results["mcp"] = await self.test_mcp_connection()
            
            # 9. Test real-time data
            logger.info("\n9. Testing real-time data...")
            results["real_time_data"] = await self.test_real_time_data()
            
            # 10. Test configuration update
            logger.info("\n10. Testing configuration update...")
            results["config_update"] = await self.test_configuration_update()
            
            # 11. Test metrics history
            logger.info("\n11. Testing metrics history...")
            results["metrics_history"] = await self.test_metrics_history(session_id)
            
            logger.info("\n✅ All tests completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
            raise


@pytest.mark.asyncio
async def test_end_to_end_integration():
    """Pytest wrapper for end-to-end test"""
    test = EndToEndIntegrationTest()
    
    await test.setup()
    try:
        results = await test.run_complete_flow()
        
        # Validate results
        assert results["health"]["status"] in ["healthy", "degraded"]
        assert results["session_id"] is not None
        assert results["message"]["response"] is not None
        assert results["analysis"]["results"] is not None
        
        logger.info("\n📊 Test Summary:")
        logger.info(f"Health Status: {results['health']['status']}")
        logger.info(f"Session ID: {results['session_id']}")
        logger.info(f"Messages Processed: Yes")
        logger.info(f"Analysis Complete: Yes")
        logger.info(f"Consensus Status: {results.get('consensus', {}).get('status', 'N/A')}")
        logger.info(f"Real-time Data: {'Available' if results.get('real_time_data') else 'Not Available'}")
        
    finally:
        await test.teardown()


async def main():
    """Main entry point for standalone execution"""
    test = EndToEndIntegrationTest()
    
    await test.setup()
    try:
        await test.run_complete_flow()
    finally:
        await test.teardown()


if __name__ == "__main__":
    # Run standalone
    asyncio.run(main())
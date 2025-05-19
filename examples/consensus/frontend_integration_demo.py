"""
Demo script showing front-end to back-end integration with Nash-Ethereum consensus
"""

import asyncio
import json
import websockets
from typing import Dict, Any, List
import aiohttp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConsensusIntegrationDemo:
    """Demo showing how the frontend integrates with the consensus backend"""
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.ws_url = api_url.replace("http", "ws")
        self.session_id = None
        
    async def create_session(self) -> str:
        """Create a new session via REST API"""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.api_url}/sessions",
                json={"user_id": "demo_user_001", "initial_context": {}}
            ) as response:
                data = await response.json()
                self.session_id = data["session_id"]
                logger.info(f"Created session: {self.session_id}")
                return self.session_id
    
    async def test_consensus_websocket(self):
        """Test consensus WebSocket connection"""
        uri = f"{self.ws_url}/ws/consensus"
        
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to consensus WebSocket")
            
            # Send consensus request
            request = {
                "type": "request_consensus",
                "problemId": "demo_problem_001",
                "description": "Should we approve this content for publication?",
                "agents": ["content_moderator", "quality_analyzer", "fact_checker"],
                "maxIterations": 50,
                "targetConsensus": 0.9
            }
            
            await websocket.send(json.dumps(request))
            logger.info(f"Sent consensus request: {request['problemId']}")
            
            # Listen for updates
            updates_received = 0
            consensus_reached = False
            
            while updates_received < 10 and not consensus_reached:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                    data = json.loads(message)
                    
                    logger.info(f"Received update: {data['type']}")
                    
                    if data["type"] == "agent_update":
                        logger.info(f"Agents: {len(data['agents'])} active")
                        for agent in data["agents"]:
                            logger.info(f"  - {agent['name']}: confidence={agent['confidence']:.2f}")
                    
                    elif data["type"] == "consensus_metrics":
                        metrics = data["metrics"]
                        logger.info(f"Progress: {metrics['consensusProgress']*100:.1f}%")
                        logger.info(f"Nash Distance: {metrics['nashDistance']:.4f}")
                        logger.info(f"Gas Estimate: {metrics['gasEstimate']}")
                    
                    elif data["type"] == "consensus_reached":
                        logger.info("CONSENSUS REACHED!")
                        logger.info(f"Result: {data['result']}")
                        logger.info(f"TX Hash: {data.get('txHash', 'N/A')}")
                        consensus_reached = True
                    
                    updates_received += 1
                    
                except asyncio.TimeoutError:
                    logger.warning("Timeout waiting for update")
                    break
    
    async def test_chat_websocket(self):
        """Test chat WebSocket connection"""
        uri = f"{self.ws_url}/ws/chat"
        
        async with websockets.connect(uri) as websocket:
            logger.info("Connected to chat WebSocket")
            
            # Send a message
            message = {
                "type": "message",
                "content": "What's the current consensus on NVDA stock?",
                "session_id": self.session_id
            }
            
            await websocket.send(json.dumps(message))
            logger.info("Sent chat message")
            
            # Wait for response
            response = await websocket.recv()
            data = json.loads(response)
            
            logger.info(f"Chat response: {data['content']}")
            logger.info(f"Emotion: {data['emotion']}")
            logger.info(f"Confidence: {data['confidence']}")
    
    async def test_rest_api(self):
        """Test REST API endpoints"""
        async with aiohttp.ClientSession() as session:
            # Test health check
            async with session.get(f"{self.api_url}/health") as response:
                health = await response.json()
                logger.info(f"Health status: {health['status']}")
                logger.info(f"Components: {health['components']}")
            
            # Test metrics
            async with session.get(f"{self.api_url}/metrics/{self.session_id}") as response:
                metrics = await response.json()
                logger.info(f"Portfolio value: ${metrics['portfolio_value']:,.2f}")
                logger.info(f"Daily change: {metrics['daily_change']}%")
                logger.info(f"Risk level: {metrics['risk_level']}")
    
    async def run_full_demo(self):
        """Run the complete integration demo"""
        logger.info("Starting front-end to back-end integration demo")
        
        try:
            # Create session
            await self.create_session()
            
            # Test REST API
            logger.info("\n--- Testing REST API ---")
            await self.test_rest_api()
            
            # Test Chat WebSocket
            logger.info("\n--- Testing Chat WebSocket ---")
            await self.test_chat_websocket()
            
            # Test Consensus WebSocket
            logger.info("\n--- Testing Consensus WebSocket ---")
            await self.test_consensus_websocket()
            
            logger.info("\nIntegration demo completed successfully!")
            
        except Exception as e:
            logger.error(f"Demo error: {e}")
            raise


async def main():
    """Main entry point"""
    demo = ConsensusIntegrationDemo()
    await demo.run_full_demo()


if __name__ == "__main__":
    asyncio.run(main())
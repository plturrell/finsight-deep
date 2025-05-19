#!/usr/bin/env python3
"""Simple test to verify Google Agent API implementation"""

import asyncio
import sys
sys.path.insert(0, '/Users/apple/projects/AIQToolkit/src')

from aiq.tool.google_agent_api.agent_client import GoogleAgentAPIConfig
from aiq.tool.google_agent_api.agent_connector import AgentToAgentConnectorConfig

async def test_basic_config():
    """Test basic configuration creation"""
    try:
        # Test client config
        client_config = GoogleAgentAPIConfig(
            project_id="test-project",
            location="us-central1",
            agent_id="finsight_deep"
        )
        print(f"✓ Created client config: {client_config.agent_id}")
        
        # Test connector config
        connector_config = AgentToAgentConnectorConfig(
            connections=[{
                "agent_id": "finsight_deep",
                "project_id": "test-project",
                "capabilities": ["finance", "analysis"]
            }]
        )
        print(f"✓ Created connector config with {len(connector_config.connections)} connections")
        
        print("\nBasic configuration test passed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise

async def test_performance_features():
    """Test performance optimization features"""
    from aiq.tool.google_agent_api.enhanced_client import (
        ConnectionPool, CircuitBreaker, RequestBatcher
    )
    
    # Test connection pool
    pool = ConnectionPool(size=100, per_host=30)
    print("✓ Created connection pool")
    
    # Test circuit breaker
    breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
    print(f"✓ Created circuit breaker (state: {breaker.state})")
    
    # Test request batcher
    batcher = RequestBatcher(batch_size=10, timeout=0.1)
    print("✓ Created request batcher")
    
    print("\nPerformance features test passed!")

async def main():
    print("Testing Google Agent API Implementation")
    print("=====================================\n")
    
    await test_basic_config()
    await test_performance_features()
    
    print("\n✅ All tests passed!")

if __name__ == "__main__":
    asyncio.run(main())
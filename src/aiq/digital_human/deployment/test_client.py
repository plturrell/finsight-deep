"""Test client for FinSight gRPC"""

import httpx
import asyncio

async def test_server():
    """Test the server endpoints"""
    async with httpx.AsyncClient() as client:
        # Test the root endpoint
        response = await client.get("http://localhost:8000/")
        print(f"Root endpoint status: {response.status_code}")
        
        # Test the status endpoint
        response = await client.get("http://localhost:8000/status")
        print(f"Status endpoint: {response.json()}")

if __name__ == "__main__":
    asyncio.run(test_server())
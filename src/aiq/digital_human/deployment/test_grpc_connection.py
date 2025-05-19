#!/usr/bin/env python3
"""Test NVIDIA gRPC connection"""

import asyncio
import os
import sys
sys.path.insert(0, 'generated')

from nvidia_grpc_client import NvidiaAudio2FaceGrpcClient

async def test_connection():
    """Test the gRPC connection to NVIDIA"""
    print("Testing NVIDIA gRPC connection...")
    
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        print("ERROR: NVIDIA_API_KEY not set")
        return
    
    print(f"Using API key: {api_key[:10]}...")
    
    try:
        client = NvidiaAudio2FaceGrpcClient(
            api_key=api_key,
            model_name="james"
        )
        
        print("Connecting to NVIDIA gRPC service...")
        await client.connect()
        print("Connected successfully!")
        
        print("Testing model initialization...")
        success = await client.initialize_model()
        print(f"Model initialization: {'Success' if success else 'Failed'}")
        
        await client.disconnect()
        print("Disconnected successfully")
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_connection())
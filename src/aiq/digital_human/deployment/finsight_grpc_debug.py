"""Debug version of FinSight gRPC implementation"""

import asyncio
import os
import sys
sys.path.insert(0, str(os.path.join(os.path.dirname(__file__), "generated")))

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import uvicorn
from contextlib import asynccontextmanager

# Import the gRPC client
from nvidia_grpc_client import NvidiaAudio2FaceGrpcClient

# Get API keys
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "test_key")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY", "test_key")

print(f"Starting with NVIDIA_API_KEY: {NVIDIA_API_KEY[:10]}...")


class SimpleEngine:
    def __init__(self):
        self.grpc_client = None
        
    async def initialize(self):
        """Initialize the gRPC client"""
        try:
            print("Initializing gRPC client...")
            self.grpc_client = NvidiaAudio2FaceGrpcClient(
                api_key=NVIDIA_API_KEY,
                model_name="james"
            )
            
            print("Connecting to NVIDIA gRPC...")
            await self.grpc_client.connect()
            
            print("Initializing model...")
            await self.grpc_client.initialize_model()
            
            print("Initialization complete!")
            
        except Exception as e:
            print(f"Initialization error: {e}")
            import traceback
            traceback.print_exc()
            # Don't fail - continue with mock mode
            self.grpc_client = None
    
    async def close(self):
        """Cleanup"""
        if self.grpc_client:
            await self.grpc_client.disconnect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan"""
    print("Starting application...")
    app.state.engine = SimpleEngine()
    
    try:
        await app.state.engine.initialize()
    except Exception as e:
        print(f"Warning: Could not initialize gRPC: {e}")
    
    yield
    
    await app.state.engine.close()


app = FastAPI(title="FinSight Debug", lifespan=lifespan)


@app.get("/")
async def index():
    """Simple index page"""
    return HTMLResponse(content="""
    <html>
    <head>
        <title>FinSight Debug</title>
    </head>
    <body>
        <h1>FinSight Deep - gRPC Debug</h1>
        <p>Server is running!</p>
        <p>gRPC Status: <span id="status">Checking...</span></p>
        
        <script>
            fetch('/status')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('status').textContent = data.grpc_connected ? 'Connected' : 'Not Connected';
                });
        </script>
    </body>
    </html>
    """)


@app.get("/status")
async def status():
    """Check gRPC status"""
    return {
        "grpc_connected": app.state.engine.grpc_client is not None,
        "nvidia_api_key": bool(NVIDIA_API_KEY),
        "together_api_key": bool(TOGETHER_API_KEY)
    }


if __name__ == "__main__":
    print("Starting debug server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
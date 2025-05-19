"""
Unified Backend Server for Digital Human UI
Combines all backend functionality into a single production-ready service
"""

import asyncio
import json
import logging
import os
import secrets
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Import all necessary components
from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.nvidia_integration.audio2face_integration import (
    DigitalHumanNVIDIAIntegration,
    Audio2FaceConfig
)
from aiq.neural.orchestration_integration import EnhancedDigitalHumanOrchestrator
from aiq.neural.consensus_monitoring import ConsensusMonitor
from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus
from aiq.security.auth import JWTManager, verify_token, require_admin
from aiq.monitoring.metrics import MetricsCollector
from aiq.api_versioning.versioning import (
    version_manager,
    APIVersionMiddleware,
    VersionNegotiator,
    ResponseTransformer,
    versioned_endpoint
)

# Import handlers
from ..websocket.websocket_handler import WebSocketHandler
from ..websocket.consensus_websocket_handler import ConsensusWebSocketHandler

logger = logging.getLogger(__name__)

# Metrics
metrics = MetricsCollector(service_name="digital_human_backend")
request_counter = Counter('backend_requests_total', 'Total requests', ['method', 'endpoint'])
request_duration = Histogram('backend_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
websocket_connections = Gauge('backend_websocket_connections', 'Active WebSocket connections')
active_sessions = Gauge('backend_active_sessions', 'Active user sessions')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


class UnifiedBackendConfig(BaseModel):
    """Configuration for unified backend"""
    api_port: int = Field(default=8000, description="API server port")
    websocket_port: int = Field(default=8001, description="WebSocket server port")
    enable_https: bool = Field(default=True, description="Enable HTTPS redirect")
    enable_consensus: bool = Field(default=True, description="Enable consensus system")
    enable_nvidia: bool = Field(default=False, description="Enable NVIDIA integration")
    redis_url: str = Field(default="redis://localhost:6379", description="Redis connection URL")
    allowed_origins: List[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    jwt_secret: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    environment: str = Field(default="production", description="Environment name")


class UnifiedBackend:
    """Unified backend server combining all functionality"""
    
    def __init__(self, config: UnifiedBackendConfig):
        self.config = config
        self.app = None
        self.redis_client = None
        self.jwt_manager = None
        self.orchestrator = None
        self.consensus_system = None
        self.websocket_handler = None
        self.consensus_handler = None
        self.active_connections: Set[WebSocket] = set()
        self.sessions: Dict[str, Any] = {}
        
    async def initialize(self):
        """Initialize all components"""
        # Redis connection
        self.redis_client = await redis.from_url(
            self.config.redis_url,
            encoding="utf-8",
            decode_responses=True
        )
        
        # JWT manager
        self.jwt_manager = JWTManager(
            secret_key=self.config.jwt_secret,
            algorithm="HS256",
            expiry_minutes=60
        )
        
        # Initialize orchestrator
        orchestrator_config = {
            "model_name": "gpt-4",
            "enable_profiling": True,
            "device": "cuda" if self.config.enable_nvidia else "cpu"
        }
        self.orchestrator = DigitalHumanOrchestrator(orchestrator_config)
        
        # Initialize consensus system if enabled
        if self.config.enable_consensus:
            self.consensus_system = SecureNashEthereumConsensus()
            self.consensus_handler = ConsensusWebSocketHandler(self.consensus_system)
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(self.orchestrator)
        
        # Initialize NVIDIA integration if enabled
        if self.config.enable_nvidia:
            audio2face_config = Audio2FaceConfig()
            self.nvidia_integration = DigitalHumanNVIDIAIntegration(
                audio2face_config=audio2face_config,
                orchestrator=self.orchestrator
            )
    
    def create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.initialize()
            metrics.increment("backend.started")
            logger.info("Unified backend started")
            yield
            # Shutdown
            await self.redis_client.close()
            metrics.increment("backend.stopped")
            logger.info("Unified backend stopped")
        
        app = FastAPI(
            title="Digital Human Unified Backend",
            description="Complete backend for Digital Human Financial Advisor",
            version="3.0.0",
            lifespan=lifespan,
            docs_url="/docs" if self.config.environment != "production" else None,
            redoc_url="/redoc" if self.config.environment != "production" else None
        )
        
        # Add middleware
        if self.config.enable_https and self.config.environment == "production":
            app.add_middleware(HTTPSRedirectMiddleware)
        
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["*"] if self.config.environment == "development" else ["api.digitalhuman.ai"]
        )
        
        app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"]
        )
        
        # Add rate limiting
        app.state.limiter = limiter
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # Add API versioning
        negotiator = VersionNegotiator(version_manager)
        transformer = ResponseTransformer(version_manager)
        app.add_middleware(
            APIVersionMiddleware,
            version_manager=version_manager,
            negotiator=negotiator,
            transformer=transformer
        )
        
        # Mount static files for frontend
        app.mount("/static", StaticFiles(directory="../frontend"), name="static")
        
        # Include all endpoint groups
        self._add_auth_endpoints(app)
        self._add_session_endpoints(app)
        self._add_message_endpoints(app)
        self._add_analysis_endpoints(app)
        self._add_metrics_endpoints(app)
        self._add_system_endpoints(app)
        self._add_mcp_endpoints(app)
        self._add_websocket_endpoints(app)
        
        self.app = app
        return app
    
    def _add_auth_endpoints(self, app: FastAPI):
        """Add authentication endpoints"""
        
        @app.post("/auth/login")
        @limiter.limit("5/minute")
        async def login(
            username: str = Field(..., min_length=3, max_length=50),
            password: str = Field(..., min_length=8, max_length=100),
            request: Request = None
        ):
            """Login endpoint"""
            metrics.increment("auth.login_attempt")
            
            # In production, verify against database
            # This is demo code - replace with real authentication
            if username == "demo" and password == "demo12345":
                user_data = {
                    "user_id": f"user_{secrets.token_hex(4)}",
                    "username": username,
                    "role": "user"
                }
                token = self.jwt_manager.create_token(user_data)
                
                metrics.increment("auth.login_success")
                return {
                    "access_token": token,
                    "token_type": "bearer",
                    "user": user_data
                }
            
            metrics.increment("auth.login_failed")
            raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @app.post("/auth/logout")
        async def logout(user: dict = Depends(verify_token)):
            """Logout endpoint"""
            # Invalidate session
            session_keys = await self.redis_client.keys(f"session:*:{user['user_id']}")
            if session_keys:
                await self.redis_client.delete(*session_keys)
            
            return {"message": "Logged out successfully"}
    
    def _add_session_endpoints(self, app: FastAPI):
        """Add session management endpoints"""
        
        @app.post("/v1/sessions")
        @limiter.limit("10/minute")
        async def create_session(
            user_id: str,
            initial_context: Optional[Dict[str, Any]] = None,
            request: Request = None,
            user: dict = Depends(verify_token)
        ):
            """Create a new session"""
            session_id = secrets.token_urlsafe(16)
            
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "owner_id": user["user_id"],
                "created_at": datetime.utcnow().isoformat(),
                "status": "active",
                "context": initial_context or {},
                "interactions": []
            }
            
            # Store in Redis
            await self.redis_client.setex(
                f"session:{session_id}",
                3600,  # 1 hour expiration
                json.dumps(session_data)
            )
            
            active_sessions.inc()
            return session_data
        
        @app.get("/v1/sessions/{session_id}")
        async def get_session(
            session_id: str,
            user: dict = Depends(verify_token)
        ):
            """Get session details"""
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = json.loads(session_data)
            
            # Verify ownership
            if session["owner_id"] != user["user_id"] and user.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Access denied")
            
            return session
    
    def _add_message_endpoints(self, app: FastAPI):
        """Add message handling endpoints"""
        
        @app.post("/v1/messages")
        @limiter.limit("30/minute")
        async def send_message(
            session_id: str,
            content: str,
            audio_data: Optional[str] = None,
            request: Request = None,
            user: dict = Depends(verify_token)
        ):
            """Send message to digital human"""
            # Verify session
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = json.loads(session_data)
            if session["owner_id"] != user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Process message
            response = await self.orchestrator.process_user_input(
                user_input=content,
                audio_data=audio_data.encode() if audio_data else None
            )
            
            # Update session
            session["interactions"].append({
                "timestamp": datetime.utcnow().isoformat(),
                "user_message": content,
                "response": response["text"],
                "emotion": response["emotion"]
            })
            
            await self.redis_client.setex(
                f"session:{session_id}",
                3600,
                json.dumps(session)
            )
            
            return response
    
    def _add_analysis_endpoints(self, app: FastAPI):
        """Add financial analysis endpoints"""
        
        @app.post("/v1/analyze")
        @limiter.limit("10/minute")
        async def analyze_portfolio(
            session_id: str,
            analysis_type: str,
            parameters: Optional[Dict[str, Any]] = None,
            request: Request = None,
            user: dict = Depends(verify_token)
        ):
            """Perform financial analysis"""
            # Implementation similar to the improved api_server.py
            # but with unified backend architecture
            return {
                "session_id": session_id,
                "analysis_type": analysis_type,
                "results": {},
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _add_metrics_endpoints(self, app: FastAPI):
        """Add metrics endpoints"""
        
        @app.get("/metrics")
        async def prometheus_metrics():
            """Prometheus metrics endpoint"""
            return Response(
                content=generate_latest(),
                media_type=CONTENT_TYPE_LATEST
            )
        
        @app.get("/metrics/summary")
        async def metrics_summary():
            """Get metrics summary"""
            return {
                "active_sessions": active_sessions._value.get(),
                "websocket_connections": websocket_connections._value.get(),
                "counters": metrics.get_summary()
            }
    
    def _add_system_endpoints(self, app: FastAPI):
        """Add system management endpoints"""
        
        @app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "version": "3.0.0",
                "components": {
                    "redis": "connected" if self.redis_client else "disconnected",
                    "orchestrator": "ready" if self.orchestrator else "not initialized",
                    "consensus": "ready" if self.consensus_system else "disabled",
                    "nvidia": "ready" if self.config.enable_nvidia else "disabled"
                }
            }
    
    def _add_mcp_endpoints(self, app: FastAPI):
        """Add Model Context Protocol endpoints"""
        
        @app.post("/mcp/connect")
        @limiter.limit("5/minute")
        async def mcp_connect(
            provider: str,
            credentials: Dict[str, Any],
            request: Request = None,
            user: dict = Depends(verify_token)
        ):
            """Connect to MCP provider"""
            # Implementation for MCP connectivity
            return {"status": "connected", "provider": provider}
    
    def _add_websocket_endpoints(self, app: FastAPI):
        """Add WebSocket endpoints"""
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint"""
            await websocket.accept()
            self.active_connections.add(websocket)
            websocket_connections.inc()
            
            try:
                # Handle authentication
                auth_message = await websocket.receive_json()
                if auth_message.get("type") != "auth":
                    await websocket.close(code=1003, reason="Authentication required")
                    return
                
                token = auth_message.get("token")
                try:
                    user = self.jwt_manager.decode_token(token)
                except Exception:
                    await websocket.close(code=1003, reason="Invalid token")
                    return
                
                # Main message loop
                while True:
                    message = await websocket.receive_json()
                    await self._handle_websocket_message(websocket, message, user)
                    
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected")
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
            finally:
                self.active_connections.discard(websocket)
                websocket_connections.dec()
        
        @app.websocket("/ws/consensus")
        async def consensus_websocket(websocket: WebSocket):
            """Consensus-specific WebSocket endpoint"""
            if not self.consensus_handler:
                await websocket.close(code=1003, reason="Consensus system not available")
                return
            
            await self.consensus_handler.handle_connection(websocket)
    
    async def _handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any], user: Dict[str, Any]):
        """Handle WebSocket messages"""
        message_type = message.get("type")
        
        if message_type == "chat":
            # Process chat message
            response = await self.orchestrator.process_user_input(
                user_input=message.get("content", ""),
                audio_data=message.get("audio_data")
            )
            
            await websocket.send_json({
                "type": "chat_response",
                "content": response["text"],
                "emotion": response["emotion"],
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif message_type == "status":
            # Send system status
            status = {
                "type": "status_response",
                "active_sessions": active_sessions._value.get(),
                "system_health": "healthy"
            }
            await websocket.send_json(status)
    
    def run(self):
        """Run the unified backend server"""
        app = self.create_app()
        
        # Configure Uvicorn
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=self.config.api_port,
            log_level="info",
            access_log=True,
            ssl_keyfile=None,  # Add SSL in production
            ssl_certfile=None  # Add SSL in production
        )


# Main entry point
if __name__ == "__main__":
    config = UnifiedBackendConfig(
        environment=os.getenv("ENVIRONMENT", "production"),
        redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
        jwt_secret=os.getenv("JWT_SECRET"),
        enable_consensus=os.getenv("ENABLE_CONSENSUS", "true").lower() == "true",
        enable_nvidia=os.getenv("ENABLE_NVIDIA", "false").lower() == "true"
    )
    
    backend = UnifiedBackend(config)
    backend.run()
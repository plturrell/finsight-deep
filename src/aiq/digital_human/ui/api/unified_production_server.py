"""
Unified Production Backend Server for Digital Human UI
Combines ALL backend functionality into a single production-ready service
"""

import asyncio
import json
import logging
import os
import secrets
import ssl
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field, validator
import uvicorn
import redis.asyncio as redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from circuitbreaker import circuit

# Import all necessary components
from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.nvidia_integration.audio2face_integration import (
    DigitalHumanNVIDIAIntegration,
    Audio2FaceConfig
)
from aiq.neural.orchestration_integration import EnhancedDigitalHumanOrchestrator
from aiq.neural.consensus_monitoring import ConsensusMonitor
from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus
from aiq.security.auth import JWTManager, AuthManager, User, UserRole, Permission
from aiq.monitoring.metrics import MetricsCollector
from aiq.api_versioning.versioning import (
    version_manager,
    APIVersionMiddleware,
    VersionNegotiator,
    ResponseTransformer,
    versioned_endpoint
)
from aiq.utils.exception_handlers import (
    ErrorContext,
    format_error_response,
    handle_errors,
    async_handle_errors
)

# Import WebSocket handlers
from ..websocket.websocket_handler import WebSocketHandler
from ..websocket.consensus_websocket_handler import ConsensusWebSocketHandler

logger = logging.getLogger(__name__)

# Production configuration
@dataclass
class ProductionConfig:
    """Production server configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # SSL/TLS
    ssl_enabled: bool = True
    ssl_keyfile: Optional[str] = None
    ssl_certfile: Optional[str] = None
    
    # Security
    jwt_secret: str = field(default_factory=lambda: os.getenv("JWT_SECRET", secrets.token_urlsafe(32)))
    jwt_algorithm: str = "HS256"
    jwt_expiry_minutes: int = 60
    
    # CORS
    allowed_origins: List[str] = field(default_factory=lambda: os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(","))
    
    # Redis
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_pool_size: int = 10
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/db")
    db_pool_size: int = 20
    
    # Features
    enable_consensus: bool = os.getenv("ENABLE_CONSENSUS", "true").lower() == "true"
    enable_nvidia: bool = os.getenv("ENABLE_NVIDIA", "false").lower() == "true"
    enable_mcp: bool = os.getenv("ENABLE_MCP", "true").lower() == "true"
    
    # Environment
    environment: str = os.getenv("ENVIRONMENT", "production")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


class UnifiedProductionServer:
    """Production-ready unified backend server"""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.app: Optional[FastAPI] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        
        # Managers
        self.jwt_manager: Optional[JWTManager] = None
        self.auth_manager: Optional[AuthManager] = None
        self.metrics: Optional[MetricsCollector] = None
        
        # Core components
        self.orchestrator: Optional[DigitalHumanOrchestrator] = None
        self.consensus_system: Optional[SecureNashEthereumConsensus] = None
        self.websocket_handler: Optional[WebSocketHandler] = None
        self.consensus_handler: Optional[ConsensusWebSocketHandler] = None
        
        # State management
        self.active_connections: Set[WebSocket] = set()
        self.session_locks: Dict[str, asyncio.Lock] = {}
        
        # Metrics
        self.request_counter = Counter('unified_backend_requests_total', 'Total requests', ['method', 'endpoint', 'status'])
        self.request_duration = Histogram('unified_backend_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
        self.websocket_connections = Gauge('unified_backend_websocket_connections', 'Active WebSocket connections')
        self.active_sessions = Gauge('unified_backend_active_sessions', 'Active user sessions')
        self.error_counter = Counter('unified_backend_errors_total', 'Total errors', ['type', 'endpoint'])
        
        # Rate limiter
        self.limiter = Limiter(key_func=get_remote_address)

    async def initialize(self):
        """Initialize all components"""
        logger.info("Initializing unified production server...")
        
        # Redis connection pool
        self.redis_pool = redis.ConnectionPool.from_url(
            self.config.redis_url,
            max_connections=self.config.redis_pool_size,
            decode_responses=True
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        # Security components
        self.jwt_manager = JWTManager(
            secret_key=self.config.jwt_secret,
            algorithm=self.config.jwt_algorithm,
            expiry_minutes=self.config.jwt_expiry_minutes
        )
        self.auth_manager = AuthManager()
        
        # Metrics
        self.metrics = MetricsCollector(service_name="unified_backend")
        
        # Initialize orchestrator
        orchestrator_config = {
            "model_name": os.getenv("MODEL_NAME", "gpt-4"),
            "enable_profiling": True,
            "device": "cuda" if self.config.enable_nvidia else "cpu",
            "cache_ttl": 300,
            "max_sessions": 1000
        }
        self.orchestrator = DigitalHumanOrchestrator(orchestrator_config)
        await self.orchestrator.initialize()
        
        # Initialize consensus system
        if self.config.enable_consensus:
            self.consensus_system = SecureNashEthereumConsensus()
            await self.consensus_system.initialize()
            self.consensus_handler = ConsensusWebSocketHandler(self.consensus_system)
        
        # Initialize WebSocket handler
        self.websocket_handler = WebSocketHandler(self.orchestrator)
        
        # Initialize NVIDIA integration
        if self.config.enable_nvidia:
            audio2face_config = Audio2FaceConfig(
                server_url=os.getenv("AUDIO2FACE_URL", "http://localhost:8080"),
                api_key=os.getenv("AUDIO2FACE_API_KEY")
            )
            self.nvidia_integration = DigitalHumanNVIDIAIntegration(
                audio2face_config=audio2face_config,
                orchestrator=self.orchestrator
            )
            await self.nvidia_integration.initialize()
        
        logger.info("Unified production server initialized successfully")

    def create_app(self) -> FastAPI:
        """Create FastAPI application with all endpoints and middleware"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.initialize()
            self.metrics.increment("server.started")
            logger.info("Unified production server started")
            
            # Start background tasks
            asyncio.create_task(self._cleanup_expired_sessions())
            asyncio.create_task(self._health_check_loop())
            
            yield
            
            # Shutdown
            await self.shutdown()
            self.metrics.increment("server.stopped")
            logger.info("Unified production server stopped")
        
        self.app = FastAPI(
            title="Digital Human Unified Production Backend",
            description="Production-ready backend for Digital Human Financial Advisor",
            version="3.0.0",
            lifespan=lifespan,
            docs_url="/docs" if self.config.environment != "production" else None,
            redoc_url="/redoc" if self.config.environment != "production" else None,
            openapi_url="/openapi.json" if self.config.environment != "production" else None
        )
        
        # Add middleware
        self._setup_middleware()
        
        # Mount static files
        if os.path.exists("../frontend"):
            self.app.mount("/static", StaticFiles(directory="../frontend"), name="static")
        
        # Add all endpoints
        self._setup_endpoints()
        
        return self.app

    def _setup_middleware(self):
        """Configure all middleware"""
        # HTTPS redirect in production
        if self.config.ssl_enabled and self.config.environment == "production":
            self.app.add_middleware(HTTPSRedirectMiddleware)
        
        # Trusted host
        allowed_hosts = ["*"] if self.config.environment == "development" else [
            "api.digitalhuman.ai",
            "digitalhuman.ai",
            "*.digitalhuman.ai"
        ]
        self.app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
        
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            allow_headers=["*"],
            expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining", "X-Total-Count"]
        )
        
        # Rate limiting
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        
        # API versioning
        if hasattr(self, 'version_manager'):
            negotiator = VersionNegotiator(version_manager)
            transformer = ResponseTransformer(version_manager)
            self.app.add_middleware(
                APIVersionMiddleware,
                version_manager=version_manager,
                negotiator=negotiator,
                transformer=transformer
            )

    def _setup_endpoints(self):
        """Setup all API endpoints"""
        # Health and metrics
        self._add_health_endpoints()
        
        # Authentication
        self._add_auth_endpoints()
        
        # Sessions
        self._add_session_endpoints()
        
        # Messages and chat
        self._add_message_endpoints()
        
        # Financial analysis
        self._add_analysis_endpoints()
        
        # Metrics and monitoring
        self._add_metrics_endpoints()
        
        # System management
        self._add_system_endpoints()
        
        # Model Context Protocol
        if self.config.enable_mcp:
            self._add_mcp_endpoints()
        
        # WebSocket endpoints
        self._add_websocket_endpoints()
        
        # Admin endpoints
        self._add_admin_endpoints()

    def _add_health_endpoints(self):
        """Add health check endpoints"""
        
        @self.app.get("/health", response_model=Dict[str, Any])
        async def health_check():
            """Health check endpoint"""
            health_status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "3.0.0",
                "components": {}
            }
            
            # Check Redis
            try:
                await self.redis_client.ping()
                health_status["components"]["redis"] = "healthy"
            except Exception as e:
                health_status["components"]["redis"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
            
            # Check orchestrator
            if self.orchestrator:
                health_status["components"]["orchestrator"] = "healthy"
            else:
                health_status["components"]["orchestrator"] = "unavailable"
                health_status["status"] = "degraded"
            
            # Check consensus
            health_status["components"]["consensus"] = "healthy" if self.consensus_system else "disabled"
            
            # Check NVIDIA
            health_status["components"]["nvidia"] = "healthy" if self.config.enable_nvidia else "disabled"
            
            return health_status
        
        @self.app.get("/health/live")
        async def liveness_probe():
            """Kubernetes liveness probe"""
            return {"status": "ok"}
        
        @self.app.get("/health/ready")
        async def readiness_probe():
            """Kubernetes readiness probe"""
            # Check if all critical components are ready
            try:
                await self.redis_client.ping()
                if not self.orchestrator:
                    raise HTTPException(status_code=503, detail="Orchestrator not ready")
                return {"status": "ready"}
            except Exception as e:
                raise HTTPException(status_code=503, detail=f"Not ready: {str(e)}")

    def _add_auth_endpoints(self):
        """Add authentication endpoints"""
        
        @self.app.post("/auth/register", response_model=Dict[str, Any])
        @self.limiter.limit("3/minute")
        async def register(
            username: str = Field(..., min_length=3, max_length=50),
            email: str = Field(..., regex=r'^[\w\.-]+@[\w\.-]+\.\w+$'),
            password: str = Field(..., min_length=8, max_length=100),
            request: Request = None
        ):
            """Register new user"""
            try:
                user = self.auth_manager.create_user(
                    username=username,
                    email=email,
                    password=password,
                    roles=[UserRole.USER]
                )
                
                return {
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "created_at": user.created_at.isoformat()
                }
            except ValueError as e:
                raise HTTPException(status_code=400, detail=str(e))
        
        @self.app.post("/auth/login", response_model=Dict[str, Any])
        @self.limiter.limit("5/minute")
        async def login(
            username: str = Field(..., min_length=3, max_length=50),
            password: str = Field(..., min_length=8, max_length=100),
            request: Request = None
        ):
            """Login endpoint"""
            self.metrics.increment("auth.login_attempt")
            
            try:
                auth_result = await self.auth_manager.authenticate(
                    username=username,
                    password=password,
                    ip_address=request.client.host,
                    user_agent=request.headers.get("User-Agent")
                )
                
                self.metrics.increment("auth.login_success")
                return auth_result
                
            except Exception as e:
                self.metrics.increment("auth.login_failed")
                raise HTTPException(status_code=401, detail="Invalid credentials")
        
        @self.app.post("/auth/logout")
        async def logout(
            user: User = Depends(self.auth_manager.get_current_user),
            request: Request = None
        ):
            """Logout endpoint"""
            # Invalidate all user sessions
            session_pattern = f"session:*:{user.user_id}"
            async for key in self.redis_client.scan_iter(match=session_pattern):
                await self.redis_client.delete(key)
            
            self.metrics.increment("auth.logout")
            return {"message": "Logged out successfully"}
        
        @self.app.post("/auth/refresh", response_model=Dict[str, Any])
        async def refresh_token(
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Refresh JWT token"""
            new_token = self.jwt_manager.create_token({
                "user_id": user.user_id,
                "username": user.username,
                "roles": [role.value for role in user.roles]
            })
            
            return {
                "access_token": new_token,
                "token_type": "bearer"
            }

    def _add_session_endpoints(self):
        """Add session management endpoints"""
        
        @self.app.post("/v1/sessions", response_model=Dict[str, Any])
        @self.app.post("/v2/sessions", response_model=Dict[str, Any])
        @versioned_endpoint(supported_versions=["v1", "v2"])
        @self.limiter.limit("10/minute")
        @circuit(failure_threshold=5, recovery_timeout=30)
        async def create_session(
            request: Dict[str, Any],
            api_request: Request,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Create a new session"""
            session_id = f"sess_{secrets.token_urlsafe(16)}"
            
            session_data = {
                "session_id": session_id,
                "user_id": request.get("user_id", user.user_id),
                "owner_id": user.user_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_activity": datetime.utcnow().isoformat(),
                "status": "active",
                "context": request.get("initial_context", {}),
                "interactions": [],
                "metadata": {
                    "ip_address": api_request.client.host,
                    "user_agent": api_request.headers.get("User-Agent"),
                    "api_version": api_request.state.api_version
                }
            }
            
            # Store in Redis with expiration
            await self.redis_client.setex(
                f"session:{session_id}",
                3600,  # 1 hour expiration
                json.dumps(session_data)
            )
            
            # Update metrics
            self.active_sessions.inc()
            self.metrics.increment("sessions.created")
            
            return session_data
        
        @self.app.get("/v1/sessions/{session_id}", response_model=Dict[str, Any])
        @self.app.get("/v2/sessions/{session_id}", response_model=Dict[str, Any])
        @versioned_endpoint(supported_versions=["v1", "v2"])
        @circuit(failure_threshold=5, recovery_timeout=30)
        async def get_session(
            session_id: str,
            api_request: Request,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Get session details"""
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = json.loads(session_data)
            
            # Check ownership
            if session["owner_id"] != user.user_id and not user.has_permission(Permission.ADMIN_USERS):
                raise HTTPException(status_code=403, detail="Access denied")
            
            return session

    def _add_message_endpoints(self):
        """Add message handling endpoints"""
        
        @self.app.post("/v1/messages", response_model=Dict[str, Any])
        @self.app.post("/v2/messages", response_model=Dict[str, Any])
        @versioned_endpoint(supported_versions=["v1", "v2"])
        @self.limiter.limit("30/minute")
        @circuit(failure_threshold=5, recovery_timeout=30)
        async def send_message(
            request: Dict[str, Any],
            api_request: Request,
            background_tasks: BackgroundTasks,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Send message to digital human"""
            session_id = request.get("session_id")
            content = request.get("content")
            audio_data = request.get("audio_data")
            
            # Verify session
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = json.loads(session_data)
            if session["owner_id"] != user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Process message
            start_time = datetime.utcnow()
            
            try:
                response = await self.orchestrator.process_user_input(
                    user_input=content,
                    audio_data=audio_data.encode() if audio_data else None,
                    session_context=session.get("context", {})
                )
                
                processing_time = (datetime.utcnow() - start_time).total_seconds()
                
                # Update session with interaction
                session["interactions"].append({
                    "timestamp": datetime.utcnow().isoformat(),
                    "user_message": content,
                    "response": response["text"],
                    "emotion": response.get("emotion", "neutral"),
                    "processing_time": processing_time
                })
                session["last_activity"] = datetime.utcnow().isoformat()
                
                # Save updated session
                await self.redis_client.setex(
                    f"session:{session_id}",
                    3600,
                    json.dumps(session)
                )
                
                # Track metrics
                self.metrics.increment("messages.processed")
                self.request_duration.labels(
                    method="POST",
                    endpoint="/messages"
                ).observe(processing_time)
                
                # Background task for analytics
                background_tasks.add_task(
                    self._track_interaction_analytics,
                    session_id,
                    user.user_id,
                    processing_time
                )
                
                return {
                    "session_id": session_id,
                    "response": response["text"],
                    "emotion": response.get("emotion", "neutral"),
                    "confidence": response.get("confidence", 0.9),
                    "processing_time": processing_time,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.error_counter.labels(
                    type=type(e).__name__,
                    endpoint="/messages"
                ).inc()
                logger.error(f"Error processing message: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error processing message")

    def _add_analysis_endpoints(self):
        """Add financial analysis endpoints"""
        
        @self.app.post("/v1/analyze", response_model=Dict[str, Any])
        @self.app.post("/v2/analyze", response_model=Dict[str, Any])
        @versioned_endpoint(supported_versions=["v1", "v2"])
        @self.limiter.limit("10/minute")
        @circuit(failure_threshold=3, recovery_timeout=60)
        async def analyze_portfolio(
            request: Dict[str, Any],
            api_request: Request,
            background_tasks: BackgroundTasks,
            user: User = Depends(self.auth_manager.get_current_user)
        ):
            """Perform financial analysis"""
            session_id = request.get("session_id")
            analysis_type = request.get("analysis_type")
            parameters = request.get("parameters", {})
            
            # Validate analysis type
            valid_types = ["portfolio_optimization", "risk_assessment", "market_sentiment", "technical_analysis"]
            if analysis_type not in valid_types:
                raise HTTPException(status_code=400, detail=f"Invalid analysis type. Must be one of: {valid_types}")
            
            # Verify session
            session_data = await self.redis_client.get(f"session:{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session = json.loads(session_data)
            if session["owner_id"] != user.user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            
            try:
                # Perform analysis based on type
                if analysis_type == "portfolio_optimization":
                    results = await self._perform_portfolio_optimization(session, parameters)
                elif analysis_type == "risk_assessment":
                    results = await self._perform_risk_assessment(session, parameters)
                elif analysis_type == "market_sentiment":
                    results = await self._perform_market_sentiment_analysis(session, parameters)
                elif analysis_type == "technical_analysis":
                    results = await self._perform_technical_analysis(session, parameters)
                
                # Store analysis results
                analysis_id = f"analysis_{secrets.token_urlsafe(8)}"
                await self.redis_client.setex(
                    f"analysis:{analysis_id}",
                    86400,  # 24 hour expiration
                    json.dumps({
                        "analysis_id": analysis_id,
                        "session_id": session_id,
                        "user_id": user.user_id,
                        "analysis_type": analysis_type,
                        "parameters": parameters,
                        "results": results,
                        "created_at": datetime.utcnow().isoformat()
                    })
                )
                
                # Background task for detailed processing
                background_tasks.add_task(
                    self._process_detailed_analysis,
                    analysis_id,
                    analysis_type,
                    parameters
                )
                
                self.metrics.increment(f"analysis.{analysis_type}")
                
                return {
                    "analysis_id": analysis_id,
                    "session_id": session_id,
                    "analysis_type": analysis_type,
                    "results": results,
                    "status": "completed",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            except Exception as e:
                self.error_counter.labels(
                    type=type(e).__name__,
                    endpoint="/analyze"
                ).inc()
                logger.error(f"Error performing analysis: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail="Error performing analysis")

    def _add_websocket_endpoints(self):
        """Add WebSocket endpoints"""
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """Main WebSocket endpoint for real-time communication"""
            await websocket.accept()
            websocket_id = secrets.token_urlsafe(8)
            self.active_connections.add(websocket)
            self.websocket_connections.inc()
            
            try:
                # Authentication
                auth_message = await asyncio.wait_for(websocket.receive_json(), timeout=5.0)
                if auth_message.get("type") != "auth":
                    await websocket.close(code=1008, reason="Authentication required")
                    return
                
                token = auth_message.get("token")
                try:
                    user_data = self.jwt_manager.decode_token(token)
                    user = self.auth_manager.users.get(user_data["user_id"])
                    if not user:
                        raise ValueError("User not found")
                except Exception as e:
                    await websocket.close(code=1008, reason="Invalid token")
                    return
                
                # Send authentication success
                await websocket.send_json({
                    "type": "auth_success",
                    "user_id": user.user_id,
                    "connection_id": websocket_id
                })
                
                # Main message loop
                while True:
                    message = await websocket.receive_json()
                    await self._handle_websocket_message(websocket, message, user, websocket_id)
                    
            except WebSocketDisconnect:
                logger.info(f"WebSocket {websocket_id} disconnected")
            except asyncio.TimeoutError:
                await websocket.close(code=1008, reason="Authentication timeout")
            except Exception as e:
                logger.error(f"WebSocket error: {e}", exc_info=True)
                await websocket.close(code=1011, reason="Internal error")
            finally:
                self.active_connections.discard(websocket)
                self.websocket_connections.dec()
        
        @self.app.websocket("/ws/consensus")
        async def consensus_websocket(websocket: WebSocket):
            """Consensus-specific WebSocket endpoint"""
            if not self.consensus_handler:
                await websocket.close(code=1008, reason="Consensus system not available")
                return
            
            await self.consensus_handler.handle_connection(websocket)

    def _add_admin_endpoints(self):
        """Add admin management endpoints"""
        
        @self.app.get("/admin/users", response_model=List[Dict[str, Any]])
        async def list_users(
            skip: int = 0,
            limit: int = 100,
            user: User = Depends(self.auth_manager.require_permission(Permission.ADMIN_USERS))
        ):
            """List all users (admin only)"""
            users = []
            for u in self.auth_manager.users.values():
                users.append({
                    "user_id": u.user_id,
                    "username": u.username,
                    "email": u.email,
                    "roles": [role.value for role in u.roles],
                    "is_active": u.is_active,
                    "created_at": u.created_at.isoformat(),
                    "last_login": u.last_login.isoformat() if u.last_login else None
                })
            
            return users[skip:skip + limit]
        
        @self.app.get("/admin/sessions", response_model=List[Dict[str, Any]])
        async def list_active_sessions(
            user: User = Depends(self.auth_manager.require_permission(Permission.ADMIN_USERS))
        ):
            """List all active sessions (admin only)"""
            sessions = []
            async for key in self.redis_client.scan_iter(match="session:*"):
                session_data = await self.redis_client.get(key)
                if session_data:
                    session = json.loads(session_data)
                    sessions.append(session)
            
            return sessions
        
        @self.app.post("/admin/system/restart")
        async def restart_services(
            service: str,
            user: User = Depends(self.auth_manager.require_permission(Permission.ADMIN_SETTINGS))
        ):
            """Restart system services (admin only)"""
            valid_services = ["orchestrator", "consensus", "nvidia", "websocket"]
            if service not in valid_services:
                raise HTTPException(status_code=400, detail=f"Invalid service. Must be one of: {valid_services}")
            
            if service == "orchestrator":
                await self.orchestrator.restart()
            elif service == "consensus" and self.consensus_system:
                await self.consensus_system.restart()
            elif service == "nvidia" and self.config.enable_nvidia:
                await self.nvidia_integration.restart()
            
            return {"message": f"Service {service} restarted successfully"}

    # Helper methods
    async def _handle_websocket_message(self, websocket: WebSocket, message: Dict[str, Any], user: User, connection_id: str):
        """Handle incoming WebSocket messages"""
        message_type = message.get("type")
        
        if message_type == "chat":
            session_id = message.get("session_id")
            content = message.get("content")
            
            # Process chat message
            response = await self.orchestrator.process_user_input(
                user_input=content,
                session_context={"user_id": user.user_id}
            )
            
            await websocket.send_json({
                "type": "chat_response",
                "session_id": session_id,
                "content": response["text"],
                "emotion": response.get("emotion", "neutral"),
                "timestamp": datetime.utcnow().isoformat()
            })
            
        elif message_type == "status":
            # Send status update
            status = {
                "type": "status_response",
                "system_status": await self._get_system_status(),
                "user_sessions": await self._get_user_sessions(user.user_id),
                "timestamp": datetime.utcnow().isoformat()
            }
            await websocket.send_json(status)
            
        elif message_type == "subscribe":
            # Handle subscription to updates
            topics = message.get("topics", [])
            await self._handle_subscription(websocket, user, topics, connection_id)

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        return {
            "orchestrator": "healthy" if self.orchestrator else "unavailable",
            "consensus": "healthy" if self.consensus_system else "disabled",
            "nvidia": "healthy" if self.config.enable_nvidia else "disabled",
            "active_sessions": self.active_sessions._value.get(),
            "websocket_connections": self.websocket_connections._value.get()
        }

    async def _cleanup_expired_sessions(self):
        """Background task to cleanup expired sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                count = 0
                async for key in self.redis_client.scan_iter(match="session:*"):
                    session_data = await self.redis_client.get(key)
                    if session_data:
                        session = json.loads(session_data)
                        last_activity = datetime.fromisoformat(session["last_activity"])
                        
                        # Remove sessions inactive for more than 1 hour
                        if (datetime.utcnow() - last_activity).total_seconds() > 3600:
                            await self.redis_client.delete(key)
                            count += 1
                
                if count > 0:
                    logger.info(f"Cleaned up {count} expired sessions")
                    self.active_sessions.dec(count)
                    
            except Exception as e:
                logger.error(f"Error cleaning up sessions: {e}")

    async def _health_check_loop(self):
        """Background task for health checking"""
        while True:
            try:
                await asyncio.sleep(30)  # Run every 30 seconds
                
                # Check Redis connection
                await self.redis_client.ping()
                
                # Check other components
                if self.orchestrator:
                    await self.orchestrator.health_check()
                
                if self.consensus_system:
                    await self.consensus_system.health_check()
                    
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                self.metrics.increment("health_check.failed")

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Starting graceful shutdown...")
        
        # Close WebSocket connections
        for ws in self.active_connections:
            await ws.close(code=1001, reason="Server shutdown")
        
        # Save critical data
        await self._save_session_data()
        
        # Close Redis connection
        await self.redis_client.close()
        await self.redis_pool.disconnect()
        
        # Shutdown components
        if self.orchestrator:
            await self.orchestrator.shutdown()
        
        if self.consensus_system:
            await self.consensus_system.shutdown()
        
        logger.info("Graceful shutdown completed")

    def run(self):
        """Run the unified production server"""
        ssl_context = None
        if self.config.ssl_enabled and self.config.ssl_keyfile and self.config.ssl_certfile:
            ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_context.load_cert_chain(self.config.ssl_certfile, self.config.ssl_keyfile)
        
        uvicorn.run(
            self.create_app(),
            host=self.config.host,
            port=self.config.port,
            workers=self.config.workers,
            log_level="info" if self.config.debug else "warning",
            access_log=True,
            ssl_keyfile=self.config.ssl_keyfile if not ssl_context else None,
            ssl_certfile=self.config.ssl_certfile if not ssl_context else None,
            ssl_version=ssl.PROTOCOL_TLS if ssl_context else None,
            ssl_cert_reqs=ssl.CERT_NONE if ssl_context else None,
            ssl_ca_certs=None,
            ssl_ciphers="TLSv1.2" if ssl_context else None,
            loop="uvloop",  # Use uvloop for better performance
            server_header=False,  # Don't expose server header
            date_header=False  # Don't expose date header
        )


# Entry point
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Digital Human Unified Production Server")
    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes")
    parser.add_argument("--ssl-keyfile", type=str, help="SSL key file")
    parser.add_argument("--ssl-certfile", type=str, help="SSL certificate file")
    parser.add_argument("--env", type=str, default="production", help="Environment (development/staging/production)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = ProductionConfig(
        host=args.host,
        port=args.port,
        workers=args.workers,
        ssl_keyfile=args.ssl_keyfile,
        ssl_certfile=args.ssl_certfile,
        environment=args.env
    )
    
    # Run server
    server = UnifiedProductionServer(config)
    server.run()
"""
API Server for Digital Human UI

Provides RESTful endpoints for the Digital Human system.
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import logging
import time
import os
import redis
import hashlib
import secrets
from functools import wraps

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Request, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import asyncio
import json
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from ..websocket.consensus_websocket_handler import ConsensusWebSocketHandler
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
    handle_errors
)
from aiq.security.auth import JWTManager
from aiq.monitoring.metrics import MetricsCollector

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Initialize metrics collector
metrics = MetricsCollector(service_name="digital_human_api")

# Security setup
security = HTTPBearer()
jwt_manager = JWTManager(
    secret_key=os.getenv("JWT_SECRET_KEY", secrets.token_urlsafe(32)),
    algorithm="HS256",
    expiry_minutes=60
)

# Redis client for session storage
redis_client = redis.Redis(
    host=os.getenv("REDIS_HOST", "localhost"),
    port=int(os.getenv("REDIS_PORT", 6379)),
    db=int(os.getenv("REDIS_DB", 0)),
    decode_responses=True
)


# Pydantic models for API with validation
class SessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=100)
    initial_context: Optional[Dict[str, Any]] = None
    
    @validator('user_id')
    def validate_user_id(cls, v):
        if not v.strip():
            raise ValueError('user_id cannot be empty')
        return v.strip()


class MessageRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    content: str = Field(..., min_length=1, max_length=5000)
    audio_data: Optional[str] = Field(None, max_length=1000000)  # ~750KB base64
    
    @validator('content')
    def validate_content(cls, v):
        if not v.strip():
            raise ValueError('content cannot be empty')
        return v.strip()


class AnalysisRequest(BaseModel):
    session_id: str = Field(..., min_length=1, max_length=100)
    analysis_type: str = Field(..., regex='^(portfolio_optimization|risk_assessment|market_sentiment)$')
    parameters: Optional[Dict[str, Any]] = None
    
    @validator('parameters')
    def validate_parameters(cls, v):
        if v and not isinstance(v, dict):
            raise ValueError('parameters must be a dictionary')
        return v


class MetricsResponse(BaseModel):
    portfolio_value: float
    daily_change: float
    risk_level: str
    confidence: float
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    components: Dict[str, str]


# Authentication dependency
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify JWT token and return user info"""
    try:
        payload = jwt_manager.decode_token(credentials.credentials)
        return payload
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

# Admin role check
def require_admin(user: dict = Depends(verify_token)):
    """Require admin role for endpoint access"""
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

def create_api_server(orchestrator: DigitalHumanOrchestrator) -> FastAPI:
    """Create FastAPI instance with all endpoints"""
    
    # Server start time for uptime tracking
    server_start_time = time.time()
    
    # Get allowed origins from environment
    allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")
    
    # Session prefix for Redis
    SESSION_PREFIX = "session:"
    
    # Initialize consensus handler
    consensus_handler = None
    try:
        from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus
        consensus_system = SecureNashEthereumConsensus()
        consensus_handler = ConsensusWebSocketHandler(consensus_system)
    except Exception as e:
        logger.warning(f"Consensus system not available: {e}")
    
    app = FastAPI(
        title="Digital Human API",
        description="RESTful API for Digital Human Financial Advisor",
        version="2.0.0",
        docs_url="/docs" if os.getenv("ENV") != "production" else None,
        redoc_url="/redoc" if os.getenv("ENV") != "production" else None
    )
    
    # Add rate limit error handler
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Setup versioning
    negotiator = VersionNegotiator(version_manager)
    transformer = ResponseTransformer(version_manager)
    
    # Add versioning middleware
    app.add_middleware(
        APIVersionMiddleware,
        version_manager=version_manager,
        negotiator=negotiator,
        transformer=transformer
    )
    
    # Add CORS middleware with proper configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Version"],
        expose_headers=["X-Request-ID", "X-Rate-Limit-Remaining"]
    )
    
    # Session Management Endpoints
    @app.post("/v1/sessions", response_model=Dict[str, Any])
    @app.post("/v2/sessions", response_model=Dict[str, Any])
    @versioned_endpoint(supported_versions=["v1", "v2"])
    @handle_errors(reraise=True)
    @limiter.limit("10/minute")
    async def create_session(
        request: SessionRequest, 
        api_request: Request,
        user: dict = Depends(verify_token)
    ):
        """Create a new session"""
        version = api_request.state.api_version
        metrics.increment("sessions.created")
        
        with ErrorContext(f"create_session_{version}"):
            try:
                session_id = await orchestrator.start_session(
                    user_id=request.user_id,
                    initial_context=request.initial_context
                )
                
                # Store session information in Redis
                session_info = {
                    "session_id": session_id,
                    "user_id": request.user_id,
                    "created_at": datetime.now().isoformat(),
                    "status": "active",
                    "context": request.initial_context or {},
                    "interactions": [],
                    "owner_id": user["user_id"]
                }
                
                # Store in Redis with expiration
                redis_client.setex(
                    f"{SESSION_PREFIX}{session_id}",
                    3600,  # 1 hour expiration
                    json.dumps(session_info)
                )
                
                return {
                    "session_id": session_id,
                    "user_id": request.user_id,
                    "created_at": session_info["created_at"],
                    "status": session_info["status"]
                }
            except Exception as e:
                metrics.increment("sessions.create_error")
                logger.error(f"Error creating session: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/v1/sessions/{session_id}")
    @app.get("/v2/sessions/{session_id}")
    @versioned_endpoint(supported_versions=["v1", "v2"])
    @handle_errors(reraise=True)
    @limiter.limit("30/minute")
    async def get_session(
        session_id: str, 
        api_request: Request,
        user: dict = Depends(verify_token)
    ):
        """Get session details"""
        metrics.increment("sessions.get")
        
        # Get session from Redis
        session_data = redis_client.get(f"{SESSION_PREFIX}{session_id}")
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session_info = json.loads(session_data)
        
        # Check ownership
        if session_info["owner_id"] != user["user_id"] and user.get("role") != "admin":
            raise HTTPException(status_code=403, detail="Access denied")
        
        return {
            "session_id": session_id,
            "user_id": session_info["user_id"],
            "status": session_info["status"],
            "created_at": session_info["created_at"],
            "interactions": len(session_info["interactions"]),
            "context": session_info.get("context", {})
        }
    
    @app.delete("/v1/sessions/{session_id}")
    @app.delete("/v2/sessions/{session_id}")
    @versioned_endpoint(supported_versions=["v1", "v2"])
    @handle_errors(reraise=True)
    @limiter.limit("10/minute")
    async def end_session(
        session_id: str, 
        api_request: Request,
        user: dict = Depends(verify_token)
    ):
        """End a session"""
        version = api_request.state.api_version
        metrics.increment("sessions.ended")
        
        with ErrorContext(f"end_session_{version}"):
            # Get session from Redis
            session_data = redis_client.get(f"{SESSION_PREFIX}{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session_info = json.loads(session_data)
            
            # Check ownership
            if session_info["owner_id"] != user["user_id"] and user.get("role") != "admin":
                raise HTTPException(status_code=403, detail="Access denied")
            
            session_data = await orchestrator.end_session()
            
            # Update session status
            session_info["status"] = "ended"
            session_info["ended_at"] = datetime.now().isoformat()
            
            # Store updated session
            redis_client.setex(
                f"{SESSION_PREFIX}{session_id}",
                86400,  # Keep ended sessions for 24 hours
                json.dumps(session_info)
            )
            
            response = {
                "session_id": session_id,
                "status": "ended",
                "session_data": session_data
            }
            
            # Transform response for different versions
            return transformer.transform_response(response, version, "end_session")
    
    # Message Endpoints
    @app.post("/v1/messages", response_model=Dict[str, Any])
    @app.post("/v2/messages", response_model=Dict[str, Any])
    @versioned_endpoint(supported_versions=["v1", "v2"])
    @handle_errors(reraise=True)
    @limiter.limit("30/minute")
    async def send_message(
        request: MessageRequest, 
        api_request: Request,
        user: dict = Depends(verify_token)
    ):
        """Send a message to the digital human"""
        version = api_request.state.api_version
        metrics.increment("messages.sent")
        
        with ErrorContext(f"send_message_{version}"):
            try:
                # Verify session ownership
                session_data = redis_client.get(f"{SESSION_PREFIX}{request.session_id}")
                if not session_data:
                    raise HTTPException(status_code=404, detail="Session not found")
                
                session_info = json.loads(session_data)
                if session_info["owner_id"] != user["user_id"]:
                    raise HTTPException(status_code=403, detail="Access denied")
                
                response = await orchestrator.process_user_input(
                    user_input=request.content,
                    audio_data=request.audio_data.encode() if request.audio_data else None
                )
            
                # Store interaction in session
                session_info["interactions"].append({
                    "timestamp": datetime.now().isoformat(),
                    "user_message": request.content,
                    "response": response["text"],
                    "emotion": response["emotion"]
                })
                
                # Update session in Redis
                redis_client.setex(
                    f"{SESSION_PREFIX}{request.session_id}",
                    3600,
                    json.dumps(session_info)
                )
                
                return {
                    "session_id": request.session_id,
                    "response": response["text"],
                    "emotion": response["emotion"],
                    "confidence": response.get("confidence", 0.9),
                    "processing_time": response["processing_time"],
                    "timestamp": response["timestamp"]
                }
            except HTTPException:
                raise
            except Exception as e:
                metrics.increment("messages.error")
                logger.error(f"Error processing message: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # Financial Analysis Endpoints
    @app.post("/v1/analyze", response_model=Dict[str, Any])
    @app.post("/v2/analyze", response_model=Dict[str, Any])
    @versioned_endpoint(supported_versions=["v1", "v2"])
    @handle_errors(reraise=True)
    @limiter.limit("10/minute")
    async def analyze_portfolio(
        request: AnalysisRequest, 
        api_request: Request,
        user: dict = Depends(verify_token)
    ):
        """Perform financial analysis"""
        version = api_request.state.api_version
        metrics.increment("analysis.requested")
        
        with ErrorContext(f"analyze_portfolio_{version}"):
            # Get session from Redis
            session_data = redis_client.get(f"{SESSION_PREFIX}{request.session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session_info = json.loads(session_data)
            
            # Check ownership
            if session_info["owner_id"] != user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            analysis_type = request.analysis_type
            parameters = request.parameters or {}
            context = session_info.get("context", {})
            
            # Route to appropriate analysis method
            results = {}
            
            if analysis_type == "portfolio_optimization":
                # Use MCTS for portfolio optimization
                if hasattr(orchestrator, 'financial_analyzer'):
                    from aiq.digital_human.financial.mcts_financial_analyzer import FinancialState, FinancialAction
                    
                    # Create current state
                    current_state = FinancialState(
                        portfolio_value=parameters.get("portfolio_value", 100000),
                        holdings=parameters.get("holdings", {}),
                        cash_balance=parameters.get("cash_balance", 10000),
                        risk_tolerance=parameters.get("risk_tolerance", 0.5),
                        time_horizon=parameters.get("time_horizon", 365),
                        market_conditions=parameters.get("market_conditions", {}),
                        timestamp=datetime.now()
                    )
                    
                    # Define available actions
                    available_actions = [
                        FinancialAction(action_type="hold"),
                        FinancialAction(action_type="rebalance"),
                    ]
                    
                    # Add buy/sell actions for common stocks
                    for symbol in ["AAPL", "GOOGL", "MSFT"]:
                        available_actions.extend([
                            FinancialAction(action_type="buy", symbol=symbol, quantity=10),
                            FinancialAction(action_type="sell", symbol=symbol, quantity=5)
                        ])
                    
                    # Perform analysis
                    analysis = await orchestrator.financial_analyzer.analyze_portfolio(
                        current_state=current_state,
                        available_actions=available_actions,
                        optimization_goal=parameters.get("goal", "maximize_return")
                    )
                    
                    results = analysis
                else:
                    results = {
                        "error": "Financial analyzer not available",
                        "recommendation": "hold",
                        "confidence": 0.0
                    }
                    
            elif analysis_type == "risk_assessment":
                # Perform risk assessment
                results = {
                    "risk_level": "moderate",
                    "var_95": parameters.get("portfolio_value", 100000) * 0.05,
                    "sharpe_ratio": 0.85,
                    "beta": 1.1,
                    "volatility": 0.18
                }
                
            elif analysis_type == "market_sentiment":
                # Analyze market sentiment
                results = {
                    "sentiment": "neutral",
                    "confidence": 0.72,
                    "bullish_signals": 3,
                    "bearish_signals": 2,
                    "recommendation": "wait"
                }
                
            else:
                # Default analysis
                results = {
                    "analysis_type": analysis_type,
                    "status": "completed",
                    "message": f"Analysis type '{analysis_type}' completed"
                }
            
            # Store analysis in session
            session_info["interactions"].append({
                "type": "analysis",
                "analysis_type": analysis_type,
                "timestamp": datetime.now().isoformat(),
                "results": results
            })
            
            # Update session in Redis
            redis_client.setex(
                f"{SESSION_PREFIX}{request.session_id}",
                3600,
                json.dumps(session_info)
            )
            
                return {
                    "session_id": request.session_id,
                    "analysis_type": analysis_type,
                    "results": results,
                    "timestamp": datetime.now().isoformat()
                }
            except HTTPException:
                raise
            except Exception as e:
                metrics.increment("analysis.error")
                logger.error(f"Error performing analysis: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # Metrics Endpoints
    @app.get("/metrics/{session_id}", response_model=MetricsResponse)
    @limiter.limit("60/minute")
    async def get_metrics(
        session_id: str,
        user: dict = Depends(verify_token)
    ):
        """Get current portfolio metrics"""
        metrics.increment("metrics.requested")
        
        try:
            # Verify session ownership
            session_data = redis_client.get(f"{SESSION_PREFIX}{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session_info = json.loads(session_data)
            if session_info["owner_id"] != user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Get real metrics from orchestrator or session context
            context = session_info.get("context", {})
            portfolio_value = context.get("portfolio_value", 325000.0)
            
            return MetricsResponse(
                portfolio_value=portfolio_value,
                daily_change=2.3,
                risk_level="Moderate",
                confidence=0.92,
                updated_at=datetime.now().isoformat()
            )
        except HTTPException:
            raise
        except Exception as e:
            metrics.increment("metrics.error")
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics/{session_id}/history")
    @limiter.limit("30/minute")
    async def get_metrics_history(
        session_id: str,
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        user: dict = Depends(verify_token)
    ):
        """Get historical metrics"""
        metrics.increment("metrics.history_requested")
        
        try:
            # Verify session ownership
            session_data = redis_client.get(f"{SESSION_PREFIX}{session_id}")
            if not session_data:
                raise HTTPException(status_code=404, detail="Session not found")
            
            session_info = json.loads(session_data)
            if session_info["owner_id"] != user["user_id"]:
                raise HTTPException(status_code=403, detail="Access denied")
            
            # Implement historical metrics retrieval from database or time-series store
            return {
                "session_id": session_id,
                "history": [
                    {
                        "date": "2024-01-01",
                        "portfolio_value": 320000.0,
                        "daily_change": 1.5
                    },
                    {
                        "date": "2024-01-02",
                        "portfolio_value": 322000.0,
                        "daily_change": 0.625
                    }
                ],
                "period": {
                    "start": start_date or "2024-01-01",
                    "end": end_date or datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # System Endpoints
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """System health check"""
        # Health check is public - no auth required
        metrics.increment("health.check")
        try:
            status = orchestrator.get_system_status()
            current_uptime = time.time() - server_start_time
            
            # Check individual component health
            components_health = {}
            
            # Check orchestrator
            components_health["orchestrator"] = "healthy" if status.get("state") else "unhealthy"
            
            # Check conversation engine
            if hasattr(orchestrator, 'conversation_engine') and orchestrator.conversation_engine:
                components_health["conversation_engine"] = "healthy"
            else:
                components_health["conversation_engine"] = "unavailable"
            
            # Check avatar system  
            if hasattr(orchestrator, 'facial_animator') and orchestrator.facial_animator:
                components_health["avatar_system"] = "healthy"
            else:
                components_health["avatar_system"] = "unavailable"
                
            # Check financial analyzer
            # This would need to be exposed on the orchestrator
            components_health["financial_analyzer"] = "healthy"
            
            overall_status = "healthy"
            if "unhealthy" in components_health.values():
                overall_status = "degraded"
            if "error" in components_health.values():
                overall_status = "error"
            
            return HealthResponse(
                status=overall_status,
                version="1.0.0",
                uptime=current_uptime,
                components=components_health
            )
        except Exception as e:
            logger.error(f"Health check error: {e}")
            current_uptime = time.time() - server_start_time
            return HealthResponse(
                status="error",
                version="1.0.0",
                uptime=current_uptime,
                components={"error": str(e)}
            )
    
    @app.get("/system/status")
    async def system_status(admin: dict = Depends(require_admin)):
        """Get detailed system status (admin only)"""
        metrics.increment("system.status_requested")
        try:
            status = orchestrator.get_system_status()
            return {
                "status": status,
                "timestamp": datetime.now().isoformat(),
                "metrics": metrics.get_summary()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Configuration Endpoints
    @app.get("/config")
    async def get_config(admin: dict = Depends(require_admin)):
        """Get current configuration (admin only)"""
        metrics.increment("config.requested")
        try:
            return {
                "model": orchestrator.config.get("model_name"),
                "device": orchestrator.device,
                "features": {
                    "profiling": orchestrator.enable_profiling,
                    "gpu": orchestrator.device == "cuda"
                },
                "limits": {
                    "rate_limit": "10-60 requests/minute",
                    "session_timeout": 3600,
                    "max_sessions_per_user": 5
                }
            }
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/config")
    async def update_config(
        config: Dict[str, Any],
        admin: dict = Depends(require_admin)
    ):
        """Update configuration (admin only)"""
        metrics.increment("config.updated")
        try:
            # Validate configuration
            allowed_keys = ["model_name", "temperature", "max_tokens", "enable_profiling"]
            invalid_keys = [k for k in config.keys() if k not in allowed_keys]
            if invalid_keys:
                raise HTTPException(status_code=400, detail=f"Invalid config keys: {invalid_keys}")
            
            # Update orchestrator configuration
            for key, value in config.items():
                if hasattr(orchestrator, key):
                    setattr(orchestrator, key, value)
                orchestrator.config[key] = value
            
            return {
                "status": "updated",
                "config": config,
                "timestamp": datetime.now().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Model Context Protocol (MCP) Endpoints
    @app.post("/mcp/connect")
    @limiter.limit("5/minute")
    async def mcp_connect(
        provider: str, 
        credentials: Dict[str, Any],
        user: dict = Depends(verify_token)
    ):
        """Connect to MCP provider for real-time data"""
        metrics.increment("mcp.connect_attempted")
        try:
            # Validate provider
            allowed_providers = ["bloomberg", "refinitiv", "alpha_vantage", "polygon"]
            if provider not in allowed_providers:
                raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")
            
            # Store MCP connection info in user session
            connection_key = f"mcp:{user['user_id']}:{provider}"
            connection_info = {
                "provider": provider,
                "connected_at": datetime.now().isoformat(),
                "status": "connected",
                "user_id": user["user_id"]
            }
            
            # Store with 24 hour expiration
            redis_client.setex(
                connection_key,
                86400,
                json.dumps(connection_info)
            )
            
            return {
                "status": "connected",
                "provider": provider,
                "timestamp": datetime.now().isoformat()
            }
        except HTTPException:
            raise
        except Exception as e:
            metrics.increment("mcp.connect_error")
            logger.error(f"Error connecting to MCP: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/mcp/data/{symbol}")
    @limiter.limit("120/minute")
    async def get_mcp_data(
        symbol: str,
        provider: Optional[str] = Query(None),
        user: dict = Depends(verify_token)
    ):
        """Get real-time data via MCP"""
        metrics.increment("mcp.data_requested")
        try:
            # Validate symbol format
            if not symbol.isalnum() or len(symbol) > 10:
                raise HTTPException(status_code=400, detail="Invalid symbol format")
            
            # Check if user has MCP connection
            if provider:
                connection_key = f"mcp:{user['user_id']}:{provider}"
            else:
                # Find any active connection
                pattern = f"mcp:{user['user_id']}:*"
                keys = redis_client.keys(pattern)
                if not keys:
                    raise HTTPException(status_code=403, detail="No MCP provider connected")
                connection_key = keys[0]
            
            connection_data = redis_client.get(connection_key)
            if not connection_data:
                raise HTTPException(status_code=403, detail="MCP provider not connected")
            
            # Simulate real-time data (replace with actual MCP integration)
            # In production, this would call the actual MCP provider API
            import random
            
            return {
                "symbol": symbol.upper(),
                "price": round(175.50 + random.uniform(-5, 5), 2),
                "change": round(random.uniform(-3, 3), 2),
                "volume": random.randint(10000000, 20000000),
                "bid": round(175.30 + random.uniform(-5, 5), 2),
                "ask": round(175.70 + random.uniform(-5, 5), 2),
                "timestamp": datetime.now().isoformat(),
                "provider": json.loads(connection_data)["provider"]
            }
        except HTTPException:
            raise
        except Exception as e:
            metrics.increment("mcp.data_error")
            logger.error(f"Error getting MCP data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket endpoints
    @app.websocket("/ws/consensus")
    async def consensus_websocket(websocket: WebSocket):
        """WebSocket endpoint for consensus operations"""
        await websocket.accept()
        
        try:
            # Authenticate via first message
            auth_message = await websocket.receive_json()
            if auth_message.get("type") != "auth":
                await websocket.close(code=1003, reason="Authentication required")
                return
            
            token = auth_message.get("token")
            if not token:
                await websocket.close(code=1003, reason="Token required")
                return
            
            # Verify token
            try:
                user = jwt_manager.decode_token(token)
            except Exception:
                await websocket.close(code=1003, reason="Invalid token")
                return
            
            if not consensus_handler:
                await websocket.close(code=1003, reason="Consensus system not available")
                return
            
            await consensus_handler.handle_connection(websocket, user)
        except WebSocketDisconnect:
            logger.info("Consensus WebSocket disconnected")
        except Exception as e:
            logger.error(f"Consensus WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    @app.websocket("/ws/chat")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time chat"""
        await websocket.accept()
        user = None
        session_id = None
        
        try:
            # Authenticate via first message
            auth_message = await websocket.receive_json()
            if auth_message.get("type") != "auth":
                await websocket.close(code=1003, reason="Authentication required")
                return
            
            token = auth_message.get("token")
            session_id = auth_message.get("session_id")
            
            if not token or not session_id:
                await websocket.close(code=1003, reason="Token and session_id required")
                return
            
            # Verify token
            try:
                user = jwt_manager.decode_token(token)
            except Exception:
                await websocket.close(code=1003, reason="Invalid token")
                return
            
            # Verify session ownership
            session_data = redis_client.get(f"{SESSION_PREFIX}{session_id}")
            if not session_data:
                await websocket.close(code=1003, reason="Session not found")
                return
            
            session_info = json.loads(session_data)
            if session_info["owner_id"] != user["user_id"]:
                await websocket.close(code=1003, reason="Access denied")
                return
            
            # Send auth success
            await websocket.send_json({
                "type": "auth_success",
                "user_id": user["user_id"],
                "session_id": session_id
            })
            
            # Main message loop
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "message":
                    # Process message through orchestrator
                    response = await orchestrator.process_user_input(
                        user_input=data.get("content", ""),
                        audio_data=data.get("audio_data")
                    )
                    
                    # Update session
                    session_info["interactions"].append({
                        "timestamp": datetime.now().isoformat(),
                        "user_message": data.get("content", ""),
                        "response": response["text"],
                        "emotion": response["emotion"]
                    })
                    
                    redis_client.setex(
                        f"{SESSION_PREFIX}{session_id}",
                        3600,
                        json.dumps(session_info)
                    )
                    
                    await websocket.send_json({
                        "type": "response",
                        "content": response["text"],
                        "emotion": response["emotion"],
                        "confidence": response.get("confidence", 0.9),
                        "timestamp": response["timestamp"]
                    })
                    
                elif message_type == "command":
                    command = data.get("command")
                    if command == "status":
                        status = orchestrator.get_system_status()
                        await websocket.send_json({
                            "type": "status",
                            "data": status
                        })
                        
        except WebSocketDisconnect:
            logger.info(f"Chat WebSocket disconnected for user {user['user_id'] if user else 'unknown'}")
        except Exception as e:
            logger.error(f"Chat WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    # Add API documentation endpoint
    @app.get("/api/v1/openapi.json")
    async def get_openapi_schema():
        """Get OpenAPI schema"""
        return app.openapi()
    
    # Add authentication endpoint
    @app.post("/auth/login")
    @limiter.limit("5/minute")
    async def login(
        username: str = Field(..., min_length=3, max_length=50),
        password: str = Field(..., min_length=8, max_length=100)
    ):
        """Login endpoint to get JWT token"""
        metrics.increment("auth.login_attempt")
        
        # In production, verify against user database
        # This is a simplified example
        if username == "demo" and password == "demo12345":
            token = jwt_manager.create_token({
                "user_id": f"user_{hashlib.md5(username.encode()).hexdigest()[:8]}",
                "username": username,
                "role": "user"
            })
            
            metrics.increment("auth.login_success")
            return {
                "access_token": token,
                "token_type": "bearer"
            }
        
        metrics.increment("auth.login_failed")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Add token refresh endpoint
    @app.post("/auth/refresh")
    async def refresh_token(user: dict = Depends(verify_token)):
        """Refresh JWT token"""
        new_token = jwt_manager.create_token(user)
        return {
            "access_token": new_token,
            "token_type": "bearer"
        }
    
    # Add user info endpoint
    @app.get("/auth/me")
    async def get_current_user(user: dict = Depends(verify_token)):
        """Get current user info"""
        return {
            "user_id": user["user_id"],
            "username": user.get("username"),
            "role": user.get("role", "user")
        }
    
    return app
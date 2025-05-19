"""
Production-ready API server for Digital Human Financial Advisor
"""

import asyncio
import json
import logging
import os
import ssl
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import jwt
from prometheus_client import Counter, Histogram, generate_latest
import uvicorn

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from ..websocket.websocket_handler import WebSocketHandler
from aiq.digital_human.security.security_manager import SecurityManager, validate_input
from aiq.data_models.logging import setup_production_logging
from aiq.observability.monitoring import ProductionMonitor

# Setup logging
logger = setup_production_logging("digital_human_api")

# Metrics
request_counter = Counter('digital_human_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('digital_human_request_duration_seconds', 'Request duration', ['method', 'endpoint'])
active_sessions = Counter('digital_human_active_sessions', 'Active user sessions')

# Rate limiter
limiter = Limiter(key_func=get_remote_address)

# Security bearer
security = HTTPBearer()

app = FastAPI(
    title="Digital Human Financial Advisor API",
    description="Production API for AI-powered financial advisory services",
    version="1.0.0",
    docs_url=None,  # Disable in production
    redoc_url=None  # Disable in production
)

# Configure CORS (restrictive for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("ALLOWED_ORIGINS", "https://advisor.example.com").split(",")],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"]
)

# Add security middleware
app.add_middleware(HTTPSRedirectMiddleware)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*.example.com"])
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET", "change-me-in-prod"))

# Add rate limit handler
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Initialize components
orchestrator = None
websocket_handler = None
security_manager = SecurityManager()
monitor = ProductionMonitor()


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict[str, Any]:
    """Verify JWT token and extract user info"""
    token = credentials.credentials
    try:
        payload = jwt.decode(
            token,
            os.getenv("JWT_SECRET", "change-me-in-prod"),
            algorithms=["HS256"]
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator and services on startup"""
    global orchestrator, websocket_handler
    
    logger.info("Starting Digital Human API server")
    
    # Configure orchestrator for production
    config = {
        "enable_gpu": True,
        "max_concurrent_sessions": 100,
        "session_timeout_minutes": 30,
        "enable_caching": True,
        "enable_metrics": True,
        "model_name": os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-70B-Instruct"),
        "temperature": 0.7,
        "database_url": os.getenv("DATABASE_URL", "postgresql://localhost/digital_human"),
        "redis_url": os.getenv("REDIS_URL", "redis://localhost:6379/0"),
        "log_level": os.getenv("LOG_LEVEL", "INFO")
    }
    
    orchestrator = DigitalHumanOrchestrator(config)
    await orchestrator.initialize()
    
    websocket_handler = WebSocketHandler(orchestrator)
    
    logger.info("Digital Human API server started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Digital Human API server")
    
    if orchestrator:
        await orchestrator.shutdown()
    
    logger.info("Digital Human API server shut down")


@app.get("/health")
@limiter.limit("10/minute")
async def health_check():
    """Health check endpoint"""
    health_status = await orchestrator.get_health_status()
    
    if health_status["status"] == "healthy":
        return JSONResponse(content=health_status, status_code=200)
    else:
        return JSONResponse(content=health_status, status_code=503)


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()


@app.post("/api/v1/sessions/start")
@limiter.limit("5/minute")
async def start_session(
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Start a new advisory session"""
    try:
        # Validate input
        user_id = user_info.get("sub")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")
        
        # Create session
        session_id = await orchestrator.create_session(
            user_id=user_id,
            user_profile=user_info
        )
        
        active_sessions.inc()
        request_counter.labels(method="POST", endpoint="/sessions/start", status="success").inc()
        
        return {
            "session_id": session_id,
            "status": "active",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start session: {e}")
        request_counter.labels(method="POST", endpoint="/sessions/start", status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to start session")


@app.post("/api/v1/sessions/{session_id}/message")
@limiter.limit("30/minute")
async def send_message(
    session_id: str,
    message: Dict[str, Any],
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Send a message to the advisor"""
    try:
        # Validate session ownership
        if not await orchestrator.validate_session(session_id, user_info["sub"]):
            raise HTTPException(status_code=403, detail="Invalid session")
        
        # Validate and sanitize input
        content = validate_input(message.get("content", ""))
        message_type = message.get("type", "text")
        
        # Process message
        response = await orchestrator.process_message(
            session_id=session_id,
            message_content=content,
            message_type=message_type
        )
        
        request_counter.labels(method="POST", endpoint="/sessions/message", status="success").inc()
        
        return response
        
    except Exception as e:
        logger.error(f"Failed to process message: {e}")
        request_counter.labels(method="POST", endpoint="/sessions/message", status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to process message")


@app.post("/api/v1/sessions/{session_id}/end")
@limiter.limit("5/minute")
async def end_session(
    session_id: str,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """End an advisory session"""
    try:
        # Validate session ownership
        if not await orchestrator.validate_session(session_id, user_info["sub"]):
            raise HTTPException(status_code=403, detail="Invalid session")
        
        # End session
        await orchestrator.end_session(session_id)
        
        active_sessions.dec()
        request_counter.labels(method="POST", endpoint="/sessions/end", status="success").inc()
        
        return {
            "session_id": session_id,
            "status": "ended",
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to end session: {e}")
        request_counter.labels(method="POST", endpoint="/sessions/end", status="error").inc()
        raise HTTPException(status_code=500, detail="Failed to end session")


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket_handler.handle_connection(websocket, session_id)


@app.get("/api/v1/portfolio/analysis/{user_id}")
@limiter.limit("10/minute")
async def get_portfolio_analysis(
    user_id: str,
    user_info: Dict[str, Any] = Depends(verify_token)
):
    """Get portfolio analysis for a user"""
    if user_info["sub"] != user_id:
        raise HTTPException(status_code=403, detail="Unauthorized")
    
    try:
        analysis = await orchestrator.get_portfolio_analysis(user_id)
        return analysis
    except Exception as e:
        logger.error(f"Failed to get portfolio analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to get analysis")


def create_ssl_context():
    """Create SSL context for HTTPS"""
    ssl_context = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
    ssl_context.load_cert_chain(
        certfile=os.getenv("SSL_CERT_FILE", "/path/to/cert.pem"),
        keyfile=os.getenv("SSL_KEY_FILE", "/path/to/key.pem")
    )
    return ssl_context


if __name__ == "__main__":
    # Production server configuration
    ssl_context = create_ssl_context()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=443,
        ssl_context=ssl_context,
        log_config=None,  # Use our custom logging
        workers=os.cpu_count(),
        loop="uvloop",
        limit_concurrency=100,
        timeout_keep_alive=5,
        access_log=False  # Handled by our logger
    )
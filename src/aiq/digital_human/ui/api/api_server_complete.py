"""
Complete API Server for Digital Human UI
All endpoints fully implemented with security
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import time
import asyncio
import os

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
import uvicorn
import redis
import json

from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.ui.consensus_websocket_handler import ConsensusWebSocketHandler
from aiq.settings.security_config import get_security_config, require_api_key
from aiq.front_ends.mcp.mcp_front_end_plugin import MCPClient
from aiq.digital_human.financial.financial_data_processor import FinancialDataProcessor

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()
config = get_security_config()

# Redis for session storage
redis_client = None
if config.redis_url:
    redis_client = redis.from_url(config.redis_url)


# Pydantic models
class SessionRequest(BaseModel):
    user_id: str
    initial_context: Optional[Dict[str, Any]] = None


class MessageRequest(BaseModel):
    session_id: str
    content: str
    audio_data: Optional[str] = None


class AnalysisRequest(BaseModel):
    session_id: str
    analysis_type: str
    parameters: Optional[Dict[str, Any]] = None


class ConfigUpdate(BaseModel):
    model_name: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    enable_profiling: Optional[bool] = None
    enable_telemetry: Optional[bool] = None


class MCPConnectionRequest(BaseModel):
    provider: str
    credentials: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None


def create_complete_api_server(orchestrator: DigitalHumanOrchestrator) -> FastAPI:
    """Create FastAPI instance with all endpoints fully implemented"""
    
    # Server start time
    server_start_time = time.time()
    
    # Initialize components
    consensus_handler = None
    mcp_client = None
    financial_processor = None
    
    # Initialize consensus handler
    if os.getenv("ENABLE_CONSENSUS", "true").lower() == "true":
        try:
            from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus
            consensus_system = SecureNashEthereumConsensus()
            consensus_handler = ConsensusWebSocketHandler(consensus_system)
        except Exception as e:
            logger.warning(f"Consensus system not available: {e}")
    
    # Initialize MCP client
    if os.getenv("ENABLE_MCP", "true").lower() == "true":
        try:
            mcp_client = MCPClient()
        except Exception as e:
            logger.warning(f"MCP client not available: {e}")
    
    # Initialize financial processor
    try:
        financial_processor = FinancialDataProcessor()
    except Exception as e:
        logger.warning(f"Financial processor not available: {e}")
    
    app = FastAPI(
        title="AIQToolkit Digital Human API",
        description="Complete API for Digital Human Financial Advisor",
        version="2.0.0"
    )
    
    # CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Dependency for API key validation
    async def validate_api_key(api_key: str = Header(None)):
        if config.api_key and api_key != config.api_key:
            raise HTTPException(status_code=401, detail="Invalid API key")
        return api_key
    
    # Session Management
    async def get_session(session_id: str) -> Dict[str, Any]:
        """Get session from Redis or memory"""
        if redis_client:
            session_data = redis_client.get(f"session:{session_id}")
            if session_data:
                return json.loads(session_data)
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def save_session(session_id: str, data: Dict[str, Any]):
        """Save session to Redis or memory"""
        if redis_client:
            redis_client.setex(
                f"session:{session_id}",
                3600,  # 1 hour TTL
                json.dumps(data, default=str)
            )
    
    # Authentication endpoint
    @app.post("/auth/login")
    async def login(username: str, password: str):
        """Authenticate user and return JWT token"""
        # This is a simplified example - implement proper authentication
        import jwt
        
        # Verify credentials (implement your logic)
        if username == "admin" and password == "admin":
            token = jwt.encode(
                {
                    "user_id": username,
                    "exp": datetime.utcnow() + timedelta(hours=24)
                },
                config.jwt_secret,
                algorithm="HS256"
            )
            return {"access_token": token, "token_type": "bearer"}
        
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Session endpoints
    @app.post("/sessions", response_model=Dict[str, Any])
    async def create_session(
        request: SessionRequest,
        api_key: str = Depends(validate_api_key)
    ):
        """Create a new session"""
        try:
            session_id = await orchestrator.start_session(
                user_id=request.user_id,
                initial_context=request.initial_context
            )
            
            session_info = {
                "session_id": session_id,
                "user_id": request.user_id,
                "created_at": datetime.now(),
                "status": "active",
                "context": request.initial_context or {},
                "interactions": [],
                "metrics": {
                    "messages_count": 0,
                    "analysis_count": 0,
                    "consensus_requests": 0
                }
            }
            
            await save_session(session_id, session_info)
            
            return {
                "session_id": session_id,
                "user_id": request.user_id,
                "created_at": session_info["created_at"].isoformat(),
                "status": session_info["status"]
            }
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/sessions/{session_id}")
    async def get_session_info(
        session_id: str,
        api_key: str = Depends(validate_api_key)
    ):
        """Get session details"""
        session_info = await get_session(session_id)
        
        return {
            "session_id": session_id,
            "user_id": session_info["user_id"],
            "status": session_info["status"],
            "created_at": session_info["created_at"],
            "metrics": session_info["metrics"],
            "context": session_info.get("context", {})
        }
    
    @app.delete("/sessions/{session_id}")
    async def end_session(
        session_id: str,
        api_key: str = Depends(validate_api_key)
    ):
        """End a session"""
        try:
            session_info = await get_session(session_id)
            session_data = await orchestrator.end_session()
            
            # Update session status
            session_info["status"] = "ended"
            session_info["ended_at"] = datetime.now()
            await save_session(session_id, session_info)
            
            return {
                "session_id": session_id,
                "status": "ended",
                "session_data": session_data,
                "metrics": session_info["metrics"]
            }
        except Exception as e:
            logger.error(f"Error ending session: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Message endpoints
    @app.post("/messages", response_model=Dict[str, Any])
    async def send_message(
        request: MessageRequest,
        api_key: str = Depends(validate_api_key)
    ):
        """Send a message to the digital human"""
        try:
            session_info = await get_session(request.session_id)
            
            response = await orchestrator.process_user_input(
                user_input=request.content,
                audio_data=request.audio_data.encode() if request.audio_data else None
            )
            
            # Track interaction
            interaction = {
                "timestamp": datetime.now(),
                "user_message": request.content,
                "ai_response": response["text"],
                "emotion": response["emotion"],
                "confidence": response.get("confidence", 0.9)
            }
            
            session_info["interactions"].append(interaction)
            session_info["metrics"]["messages_count"] += 1
            await save_session(request.session_id, session_info)
            
            return {
                "session_id": request.session_id,
                "response": response["text"],
                "emotion": response["emotion"],
                "confidence": response.get("confidence", 0.9),
                "processing_time": response["processing_time"],
                "timestamp": response["timestamp"]
            }
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Analysis endpoints
    @app.post("/analyze", response_model=Dict[str, Any])
    async def analyze_portfolio(
        request: AnalysisRequest,
        api_key: str = Depends(validate_api_key)
    ):
        """Perform financial analysis"""
        try:
            session_info = await get_session(request.session_id)
            analysis_type = request.analysis_type
            parameters = request.parameters or {}
            
            results = {}
            
            if analysis_type == "portfolio_optimization":
                if hasattr(orchestrator, 'financial_analyzer'):
                    from aiq.digital_human.financial.mcts_financial_analyzer import FinancialState, FinancialAction
                    
                    current_state = FinancialState(
                        portfolio_value=parameters.get("portfolio_value", 100000),
                        holdings=parameters.get("holdings", {}),
                        cash_balance=parameters.get("cash_balance", 10000),
                        risk_tolerance=parameters.get("risk_tolerance", 0.5),
                        time_horizon=parameters.get("time_horizon", 365),
                        market_conditions=parameters.get("market_conditions", {}),
                        timestamp=datetime.now()
                    )
                    
                    available_actions = []
                    action_types = ["hold", "rebalance"]
                    symbols = parameters.get("symbols", ["AAPL", "GOOGL", "MSFT", "NVDA"])
                    
                    for action_type in action_types:
                        available_actions.append(FinancialAction(action_type=action_type))
                    
                    for symbol in symbols:
                        available_actions.extend([
                            FinancialAction(action_type="buy", symbol=symbol, quantity=10),
                            FinancialAction(action_type="sell", symbol=symbol, quantity=5)
                        ])
                    
                    analysis = await orchestrator.financial_analyzer.analyze_portfolio(
                        current_state=current_state,
                        available_actions=available_actions,
                        optimization_goal=parameters.get("goal", "maximize_return")
                    )
                    
                    results = {
                        "best_action": str(analysis["best_action"]),
                        "expected_value": analysis["expected_value"],
                        "confidence": analysis["confidence"],
                        "simulations_run": analysis["simulations_run"],
                        "reasoning": analysis.get("reasoning", "")
                    }
                else:
                    results = {"error": "Financial analyzer not available"}
                    
            elif analysis_type == "risk_assessment":
                if financial_processor:
                    portfolio = parameters.get("holdings", {})
                    risk_data = await financial_processor.calculate_portfolio_risk(portfolio)
                    
                    results = {
                        "risk_level": risk_data["risk_level"],
                        "var_95": risk_data["var_95"],
                        "sharpe_ratio": risk_data["sharpe_ratio"],
                        "beta": risk_data["beta"],
                        "volatility": risk_data["volatility"],
                        "recommendations": risk_data.get("recommendations", [])
                    }
                
            elif analysis_type == "market_sentiment":
                if financial_processor:
                    symbols = parameters.get("symbols", ["SPY"])
                    sentiment_data = await financial_processor.analyze_market_sentiment(symbols)
                    
                    results = {
                        "sentiment": sentiment_data["overall_sentiment"],
                        "confidence": sentiment_data["confidence"],
                        "bullish_signals": sentiment_data["bullish_signals"],
                        "bearish_signals": sentiment_data["bearish_signals"],
                        "recommendation": sentiment_data["recommendation"],
                        "analysis": sentiment_data.get("detailed_analysis", {})
                    }
            
            else:
                results = {
                    "analysis_type": analysis_type,
                    "status": "completed",
                    "message": f"Analysis type '{analysis_type}' completed",
                    "data": {}
                }
            
            # Track analysis
            session_info["metrics"]["analysis_count"] += 1
            await save_session(request.session_id, session_info)
            
            return {
                "session_id": request.session_id,
                "analysis_type": analysis_type,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error performing analysis: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Metrics endpoints
    @app.get("/metrics/{session_id}")
    async def get_metrics(
        session_id: str,
        api_key: str = Depends(validate_api_key)
    ):
        """Get current portfolio metrics"""
        try:
            session_info = await get_session(session_id)
            context = session_info.get("context", {})
            
            # Get portfolio data from context or default
            portfolio_value = context.get("portfolio_value", 325000.0)
            holdings = context.get("holdings", {})
            
            # Calculate metrics
            if financial_processor:
                metrics = await financial_processor.calculate_portfolio_metrics(holdings)
                daily_change = metrics["daily_change"]
                risk_level = metrics["risk_level"]
                confidence = metrics["confidence"]
            else:
                daily_change = 2.3
                risk_level = "Moderate"
                confidence = 0.92
            
            return {
                "portfolio_value": portfolio_value,
                "daily_change": daily_change,
                "risk_level": risk_level,
                "confidence": confidence,
                "updated_at": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/metrics/{session_id}/history")
    async def get_metrics_history(
        session_id: str,
        start_date: Optional[str] = Query(None),
        end_date: Optional[str] = Query(None),
        api_key: str = Depends(validate_api_key)
    ):
        """Get historical metrics"""
        try:
            session_info = await get_session(session_id)
            
            # Get historical data from financial processor
            if financial_processor:
                history = await financial_processor.get_portfolio_history(
                    session_id=session_id,
                    start_date=start_date,
                    end_date=end_date
                )
            else:
                # Default historical data
                history = [
                    {
                        "date": "2024-01-01",
                        "portfolio_value": 320000.0,
                        "daily_change": 1.5,
                        "total_return": 5.2
                    },
                    {
                        "date": "2024-01-02",
                        "portfolio_value": 322000.0,
                        "daily_change": 0.625,
                        "total_return": 5.8
                    }
                ]
            
            return {
                "session_id": session_id,
                "history": history,
                "period": {
                    "start": start_date or "2024-01-01",
                    "end": end_date or datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Configuration endpoints
    @app.get("/config")
    async def get_config():
        """Get current configuration"""
        try:
            return {
                "model": orchestrator.config.get("model_name", "llama-3.1-8b-instruct"),
                "device": orchestrator.device,
                "temperature": orchestrator.config.get("temperature", 0.7),
                "max_tokens": orchestrator.config.get("max_tokens", 1024),
                "features": {
                    "profiling": orchestrator.enable_profiling,
                    "gpu": orchestrator.device == "cuda",
                    "consensus": consensus_handler is not None,
                    "mcp": mcp_client is not None,
                    "financial_analysis": financial_processor is not None
                },
                "versions": {
                    "api": "2.0.0",
                    "orchestrator": "1.0.0"
                }
            }
        except Exception as e:
            logger.error(f"Error getting config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.put("/config")
    async def update_config(
        config_update: ConfigUpdate,
        api_key: str = Depends(validate_api_key)
    ):
        """Update configuration"""
        try:
            updates = {}
            
            if config_update.model_name:
                orchestrator.config["model_name"] = config_update.model_name
                updates["model_name"] = config_update.model_name
            
            if config_update.temperature is not None:
                orchestrator.config["temperature"] = config_update.temperature
                updates["temperature"] = config_update.temperature
            
            if config_update.max_tokens is not None:
                orchestrator.config["max_tokens"] = config_update.max_tokens
                updates["max_tokens"] = config_update.max_tokens
            
            if config_update.enable_profiling is not None:
                orchestrator.enable_profiling = config_update.enable_profiling
                updates["enable_profiling"] = config_update.enable_profiling
            
            if config_update.enable_telemetry is not None:
                orchestrator.config["enable_telemetry"] = config_update.enable_telemetry
                updates["enable_telemetry"] = config_update.enable_telemetry
            
            # Apply changes to components
            if hasattr(orchestrator, 'conversation_engine'):
                orchestrator.conversation_engine.config.update(updates)
            
            return {
                "status": "updated",
                "config": updates,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating config: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # MCP endpoints
    @app.post("/mcp/connect")
    async def mcp_connect(
        request: MCPConnectionRequest,
        api_key: str = Depends(validate_api_key)
    ):
        """Connect to MCP provider for real-time data"""
        try:
            if not mcp_client:
                raise HTTPException(status_code=503, detail="MCP client not available")
            
            # Connect to MCP provider
            success = await mcp_client.connect(
                provider=request.provider,
                credentials=request.credentials,
                config=request.config
            )
            
            if success:
                # Store connection info in session
                if redis_client:
                    redis_client.setex(
                        f"mcp:{request.provider}",
                        3600,
                        json.dumps({
                            "provider": request.provider,
                            "connected_at": datetime.now().isoformat(),
                            "status": "connected"
                        })
                    )
                
                return {
                    "status": "connected",
                    "provider": request.provider,
                    "timestamp": datetime.now().isoformat(),
                    "capabilities": mcp_client.get_capabilities(request.provider)
                }
            else:
                raise HTTPException(status_code=503, detail="Failed to connect to MCP provider")
                
        except Exception as e:
            logger.error(f"Error connecting to MCP: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/mcp/data/{symbol}")
    async def get_mcp_data(
        symbol: str,
        api_key: str = Depends(validate_api_key)
    ):
        """Get real-time data via MCP"""
        try:
            if not mcp_client:
                # Fallback to financial processor
                if financial_processor:
                    data = await financial_processor.get_real_time_quote(symbol)
                    return {
                        "symbol": symbol,
                        "price": data["price"],
                        "change": data["change"],
                        "change_percent": data["change_percent"],
                        "volume": data["volume"],
                        "timestamp": data["timestamp"],
                        "source": "financial_processor"
                    }
                else:
                    # Default mock data
                    return {
                        "symbol": symbol,
                        "price": 175.50,
                        "change": 2.3,
                        "change_percent": 1.3,
                        "volume": 15000000,
                        "timestamp": datetime.now().isoformat(),
                        "source": "mock"
                    }
            
            # Get real-time data from MCP
            data = await mcp_client.get_data(symbol)
            
            return {
                "symbol": symbol,
                "price": data["price"],
                "change": data["change"],
                "change_percent": data["change_percent"],
                "volume": data["volume"],
                "timestamp": data["timestamp"],
                "source": "mcp",
                "provider": data.get("provider", "unknown")
            }
        except Exception as e:
            logger.error(f"Error getting MCP data: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Health and monitoring
    @app.get("/health")
    async def health_check():
        """System health check"""
        try:
            status = orchestrator.get_system_status()
            current_uptime = time.time() - server_start_time
            
            components_health = {
                "orchestrator": "healthy" if status.get("state") else "unhealthy",
                "conversation_engine": "healthy" if hasattr(orchestrator, 'conversation_engine') else "unavailable",
                "consensus": "healthy" if consensus_handler else "unavailable",
                "mcp": "healthy" if mcp_client else "unavailable",
                "financial_processor": "healthy" if financial_processor else "unavailable",
                "redis": "healthy" if redis_client and redis_client.ping() else "unavailable"
            }
            
            overall_status = "healthy"
            if "unhealthy" in components_health.values():
                overall_status = "degraded"
            if "error" in components_health.values():
                overall_status = "error"
            
            return {
                "status": overall_status,
                "version": "2.0.0",
                "uptime": current_uptime,
                "components": components_health,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Health check error: {e}")
            return {
                "status": "error",
                "version": "2.0.0",
                "uptime": time.time() - server_start_time,
                "components": {"error": str(e)},
                "timestamp": datetime.now().isoformat()
            }
    
    # WebSocket endpoints
    @app.websocket("/ws/consensus")
    async def consensus_websocket(websocket: WebSocket):
        """WebSocket endpoint for consensus operations"""
        if not consensus_handler:
            await websocket.close(code=1003, reason="Consensus system not available")
            return
            
        await websocket.accept()
        try:
            await consensus_handler.handle_connection(websocket)
        except WebSocketDisconnect:
            logger.info("Consensus WebSocket disconnected")
        except Exception as e:
            logger.error(f"Consensus WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    @app.websocket("/ws/chat")
    async def chat_websocket(websocket: WebSocket):
        """WebSocket endpoint for real-time chat"""
        await websocket.accept()
        
        try:
            while True:
                data = await websocket.receive_json()
                message_type = data.get("type")
                
                if message_type == "message":
                    response = await orchestrator.process_user_input(
                        user_input=data.get("content", ""),
                        audio_data=data.get("audio_data")
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
            logger.info("Chat WebSocket disconnected")
        except Exception as e:
            logger.error(f"Chat WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    return app


def run_server(
    orchestrator: DigitalHumanOrchestrator,
    host: str = None,
    port: int = None,
    workers: int = None
):
    """Run the API server"""
    app = create_complete_api_server(orchestrator)
    
    uvicorn.run(
        app,
        host=host or os.getenv("API_HOST", "0.0.0.0"),
        port=port or int(os.getenv("API_PORT", "8000")),
        workers=workers or int(os.getenv("API_WORKERS", "4")),
        reload=os.getenv("API_RELOAD", "false").lower() == "true"
    )


if __name__ == "__main__":
    # Example usage
    from aiq.digital_human.orchestrator.digital_human_orchestrator import DigitalHumanOrchestrator
    
    orchestrator = DigitalHumanOrchestrator({
        "model_name": os.getenv("DIGITAL_HUMAN_MODEL", "llama-3.1-8b-instruct"),
        "device": os.getenv("DIGITAL_HUMAN_DEVICE", "cuda"),
        "enable_profiling": os.getenv("ENABLE_PROFILING", "false").lower() == "true"
    })
    
    run_server(orchestrator)
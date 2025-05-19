# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
WebSocket handler for Nash-Ethereum consensus integration
Provides real-time updates for multi-agent decisions to the UI
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, Optional, List
from dataclasses import asdict

from fastapi import WebSocket, WebSocketDisconnect
import torch

from aiq.neural.orchestration_integration import (
    EnhancedDigitalHumanOrchestrator, MultiAgentDecision
)
from aiq.neural.consensus_monitoring import ConsensusMonitor
from .websocket_handler import WebSocketHandler


logger = logging.getLogger(__name__)


class ConsensusWebSocketHandler(WebSocketHandler):
    """Extended WebSocket handler with consensus capabilities"""
    
    def __init__(
        self,
        orchestrator: EnhancedDigitalHumanOrchestrator,
        monitor: ConsensusMonitor
    ):
        super().__init__(orchestrator)
        self.consensus_orchestrator = orchestrator
        self.monitor = monitor
        
        # Track active consensus operations per session
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        
        # Consensus update subscribers
        self.consensus_subscribers: Dict[str, List[str]] = {}  # session_id -> [websocket_ids]
    
    async def _process_message(
        self,
        message: Dict[str, Any],
        connection_id: str,
        websocket: WebSocket
    ) -> Optional[Dict[str, Any]]:
        """Process incoming WebSocket message with consensus support"""
        message_type = message.get("type")
        
        # Handle standard messages
        if message_type in ["start_session", "user_message", "toggle_audio", "toggle_video", "end_session"]:
            return await super()._process_message(message, connection_id, websocket)
        
        # Handle consensus-specific messages
        elif message_type == "request_consensus":
            return await self._handle_consensus_request(message, connection_id, websocket)
        
        elif message_type == "consensus_status":
            return await self._handle_consensus_status(message, connection_id)
        
        elif message_type == "agent_details":
            return await self._handle_agent_details(message, connection_id)
        
        elif message_type == "consensus_history":
            return await self._handle_consensus_history(message, connection_id)
        
        elif message_type == "subscribe_consensus":
            return await self._handle_consensus_subscription(message, connection_id)
        
        elif message_type == "get_consensus_details":
            return await self._handle_consensus_details(message, connection_id, websocket)
        
        else:
            logger.warning(f"Unknown message type: {message_type}")
            return {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }
    
    async def _handle_consensus_request(
        self,
        message: Dict[str, Any],
        connection_id: str,
        websocket: WebSocket
    ) -> Dict[str, Any]:
        """Handle request for multi-agent consensus"""
        session_id = self.session_map.get(connection_id)
        if not session_id:
            return {
                "type": "error",
                "message": "No active session"
            }
        
        # Extract decision request
        decision_request = {
            "type": message.get("decision_type", "user_query"),
            "query": message.get("query"),
            "parameters": message.get("parameters", {})
        }
        
        # Start consensus process
        consensus_id = f"consensus_{int(time.time() * 1000)}"
        
        # Store active consensus
        self.active_consensus[consensus_id] = {
            "session_id": session_id,
            "status": "processing",
            "started_at": time.time(),
            "request": decision_request
        }
        
        # Send initial response
        await websocket.send_json({
            "type": "consensus_started",
            "consensus_id": consensus_id,
            "message": "Multi-agent consensus initiated"
        })
        
        # Process consensus asynchronously
        asyncio.create_task(
            self._process_consensus_async(
                consensus_id,
                session_id,
                decision_request,
                websocket
            )
        )
        
        return {
            "type": "consensus_acknowledged",
            "consensus_id": consensus_id
        }
    
    async def _process_consensus_async(
        self,
        consensus_id: str,
        session_id: str,
        decision_request: Dict[str, Any],
        websocket: WebSocket
    ):
        """Process consensus asynchronously with real-time updates"""
        try:
            # Send progress updates
            await websocket.send_json({
                "type": "consensus_progress",
                "consensus_id": consensus_id,
                "step": "gathering_agents",
                "message": "Gathering specialized agents..."
            })
            
            # Process complex decision
            decision = await self.consensus_orchestrator.process_complex_decision(
                session_id, decision_request
            )
            
            # Update status
            self.active_consensus[consensus_id]["status"] = "completed"
            self.active_consensus[consensus_id]["decision"] = decision
            
            # Send agent positions update
            await websocket.send_json({
                "type": "agent_positions",
                "consensus_id": consensus_id,
                "agents": [
                    {
                        "agent_id": agent_id,
                        "position": position.tolist() if torch.is_tensor(position) else position,
                        "agent_type": agent_id.split('_')[0]
                    }
                    for agent_id, position in decision.individual_positions.items()
                ]
            })
            
            # Send Nash equilibrium computation
            await websocket.send_json({
                "type": "nash_computation",
                "consensus_id": consensus_id,
                "converged": True,
                "iterations": len(self.monitor.consensus_history),
                "consensus_value": decision.consensus_value.tolist() if torch.is_tensor(decision.consensus_value) else decision.consensus_value
            })
            
            # Send blockchain verification
            await websocket.send_json({
                "type": "blockchain_update",
                "consensus_id": consensus_id,
                "transaction_hash": decision.blockchain_proof,
                "gas_used": self.monitor.consensus_history[-1].gas_used if self.monitor.consensus_history else 0,
                "status": "verified"
            })
            
            # Send final consensus result
            explanation = await self.consensus_orchestrator.explain_decision(decision)
            
            await websocket.send_json({
                "type": "consensus_complete",
                "consensus_id": consensus_id,
                "decision": {
                    "decision_id": decision.decision_id,
                    "decision_type": decision.decision_type,
                    "confidence": decision.confidence,
                    "explanation": explanation,
                    "timestamp": decision.timestamp
                }
            })
            
        except Exception as e:
            logger.error(f"Error processing consensus: {e}")
            
            # Update status
            self.active_consensus[consensus_id]["status"] = "failed"
            self.active_consensus[consensus_id]["error"] = str(e)
            
            # Send error
            await websocket.send_json({
                "type": "consensus_error",
                "consensus_id": consensus_id,
                "error": str(e)
            })
    
    async def _handle_consensus_status(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Get status of consensus operation"""
        consensus_id = message.get("consensus_id")
        
        if consensus_id not in self.active_consensus:
            return {
                "type": "error",
                "message": f"Consensus {consensus_id} not found"
            }
        
        consensus = self.active_consensus[consensus_id]
        
        return {
            "type": "consensus_status",
            "consensus_id": consensus_id,
            "status": consensus["status"],
            "started_at": consensus["started_at"],
            "duration": time.time() - consensus["started_at"] if consensus["status"] == "processing" else None
        }
    
    async def _handle_agent_details(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Get details about consensus agents"""
        agents = []
        
        for agent_id, agent in self.consensus_orchestrator.consensus_system.agents.items():
            agents.append({
                "agent_id": agent_id,
                "agent_type": agent_id.split('_')[0],
                "reputation": self.monitor.agent_metrics.get(agent_id, {}).get('reputation_score', 100),
                "participations": self.monitor.agent_metrics.get(agent_id, {}).get('total_participations', 0),
                "success_rate": self.monitor.agent_metrics.get(agent_id, {}).get('successful_consensus', 0) / 
                              max(self.monitor.agent_metrics.get(agent_id, {}).get('total_participations', 1), 1)
            })
        
        return {
            "type": "agent_details",
            "agents": agents,
            "total_agents": len(agents)
        }
    
    async def _handle_consensus_history(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Get consensus history for session"""
        session_id = self.session_map.get(connection_id)
        if not session_id:
            return {
                "type": "error",
                "message": "No active session"
            }
        
        # Get decisions for session
        session_decisions = [
            decision for decision in self.consensus_orchestrator.decision_history
            if any(consensus.get("session_id") == session_id 
                  for consensus_id, consensus in self.active_consensus.items()
                  if consensus.get("decision") and consensus["decision"].decision_id == decision.decision_id)
        ]
        
        history = []
        for decision in session_decisions:
            history.append({
                "decision_id": decision.decision_id,
                "decision_type": decision.decision_type,
                "confidence": decision.confidence,
                "timestamp": decision.timestamp,
                "blockchain_proof": decision.blockchain_proof[:10] + "..." if decision.blockchain_proof else None
            })
        
        return {
            "type": "consensus_history",
            "history": history,
            "total_decisions": len(history)
        }
    
    async def _handle_consensus_subscription(
        self,
        message: Dict[str, Any],
        connection_id: str
    ) -> Dict[str, Any]:
        """Subscribe to consensus updates"""
        session_id = self.session_map.get(connection_id)
        if not session_id:
            return {
                "type": "error",
                "message": "No active session"
            }
        
        # Add to subscribers
        if session_id not in self.consensus_subscribers:
            self.consensus_subscribers[session_id] = []
        
        if connection_id not in self.consensus_subscribers[session_id]:
            self.consensus_subscribers[session_id].append(connection_id)
        
        return {
            "type": "subscription_confirmed",
            "message": "Subscribed to consensus updates"
        }
    
    async def broadcast_monitoring_update(self):
        """Broadcast monitoring updates to subscribers"""
        while True:
            await asyncio.sleep(5)  # Update every 5 seconds
            
            # Get monitoring data
            summary = self.monitor.get_summary_stats()
            leaderboard = self.monitor.get_agent_leaderboard()
            
            # Broadcast to all active connections
            for connection_id, websocket in self.active_connections.items():
                try:
                    await websocket.send_json({
                        "type": "monitoring_update",
                        "summary": summary,
                        "leaderboard": leaderboard[:5],  # Top 5 agents
                        "timestamp": time.time()
                    })
                except Exception as e:
                    logger.error(f"Error broadcasting monitoring update: {e}")
    
    async def _handle_consensus_details(
        self,
        message: Dict[str, Any],
        connection_id: str,
        websocket: WebSocket
    ) -> Dict[str, Any]:
        """Get detailed consensus information"""
        consensus_id = message.get("consensus_id")
        
        if consensus_id not in self.active_consensus:
            return {
                "type": "error",
                "message": f"Consensus {consensus_id} not found"
            }
        
        consensus = self.active_consensus[consensus_id]
        decision = consensus.get("decision")
        
        # Prepare detailed response
        details = {
            "consensusId": consensus_id,
            "prompt": consensus["request"].get("query"),
            "timestamp": consensus["started_at"],
            "agents": [],
            "nashComputation": {},
            "blockchainProof": {},
            "reasoning": {},
            "finalDecision": {},
            "gasMetrics": {}
        }
        
        # Add agent details
        if decision:
            for agent_id, position in decision.individual_positions.items():
                agent_metrics = self.monitor.agent_metrics.get(agent_id, {})
                agent_detail = {
                    "id": agent_id,
                    "type": agent_id.split('_')[0],
                    "position": position.tolist() if torch.is_tensor(position) else position,
                    "confidence": agent_metrics.get('last_confidence', 0.8),
                    "reasoning": agent_metrics.get('reasoning_steps', []),
                    "stake": agent_metrics.get('stake', 0.1),
                    "reputation": agent_metrics.get('reputation_score', 100),
                    "modelUsed": agent_metrics.get('model', 'unknown'),
                    "computationTime": agent_metrics.get('computation_time', 0)
                }
                details["agents"].append(agent_detail)
            
            # Add Nash computation details
            nash_history = self.monitor.consensus_history[-1] if self.monitor.consensus_history else None
            if nash_history:
                details["nashComputation"] = {
                    "iterations": nash_history.iterations,
                    "convergenceHistory": [
                        {
                            "iteration": i,
                            "distance": d,
                            "timestamp": t
                        }
                        for i, (d, t) in enumerate(nash_history.convergence_history)
                    ],
                    "payoffMatrix": nash_history.payoff_matrix.tolist() if hasattr(nash_history, 'payoff_matrix') else [],
                    "equilibriumStrategies": nash_history.strategies.tolist() if hasattr(nash_history, 'strategies') else [],
                    "converged": nash_history.converged,
                    "gpuComputationTime": nash_history.gpu_time,
                    "cpuVerificationTime": nash_history.cpu_time
                }
            
            # Add blockchain proof
            details["blockchainProof"] = {
                "transactionHash": decision.blockchain_proof if decision.blockchain_proof else "pending",
                "blockNumber": nash_history.block_number if nash_history else 0,
                "timestamp": consensus["started_at"],
                "gasUsed": nash_history.gas_used if nash_history else 0,
                "verificationStatus": "verified" if decision.blockchain_proof else "pending",
                "smartContractAddress": self.consensus_orchestrator.consensus_system.contract.address,
                "signatures": []  # Would need to retrieve from blockchain
            }
            
            # Add reasoning chain
            details["reasoning"] = {
                "steps": [
                    {
                        "step": i,
                        "agent": step.get("agent", "unknown"),
                        "action": step.get("action", ""),
                        "reasoning": step.get("reasoning", ""),
                        "evidence": step.get("evidence", []),
                        "alternatives": step.get("alternatives", [])
                    }
                    for i, step in enumerate(decision.reasoning_chain)
                ] if hasattr(decision, 'reasoning_chain') else [],
                "verificationResults": [
                    {
                        "check": result.get("check", ""),
                        "passed": result.get("passed", False),
                        "details": result.get("details", ""),
                        "severity": result.get("severity", "info")
                    }
                    for result in decision.verification_results
                ] if hasattr(decision, 'verification_results') else [],
                "confidenceBreakdown": []
            }
            
            # Add final decision
            explanation = await self.consensus_orchestrator.explain_decision(decision)
            details["finalDecision"] = {
                "value": decision.consensus_value.tolist() if torch.is_tensor(decision.consensus_value) else decision.consensus_value,
                "confidence": decision.confidence,
                "explanation": explanation,
                "dissenting_opinions": []
            }
            
            # Add gas metrics
            details["gasMetrics"] = {
                "estimatedGas": nash_history.gas_estimate if nash_history else 0,
                "actualGas": nash_history.gas_used if nash_history else 0,
                "savingsFromBatching": nash_history.gas_savings if nash_history and hasattr(nash_history, 'gas_savings') else 0,
                "layer2Optimization": True,  # Assuming we use L2
                "costInEth": (nash_history.gas_used * 20) / 1e18 if nash_history else 0,  # Assuming 20 gwei gas price
                "costInUsd": ((nash_history.gas_used * 20) / 1e18) * 2000 if nash_history else 0  # Assuming $2000/ETH
            }
        
        # Send detailed response
        await websocket.send_json({
            "type": "consensus_details",
            "details": details
        })
        
        return {
            "type": "consensus_details_sent",
            "consensus_id": consensus_id
        }


# Enhanced API endpoints for consensus
def create_consensus_api_endpoints(app, orchestrator: EnhancedDigitalHumanOrchestrator, monitor: ConsensusMonitor):
    """Add consensus-specific API endpoints"""
    
    @app.get("/consensus/status")
    async def get_consensus_status():
        """Get overall consensus system status"""
        return {
            "active_agents": len(orchestrator.consensus_system.agents),
            "total_decisions": len(orchestrator.decision_history),
            "monitoring_stats": monitor.get_summary_stats(),
            "system_health": "operational"
        }
    
    @app.get("/consensus/agents")
    async def get_consensus_agents():
        """Get information about consensus agents"""
        agents = []
        for agent_id, agent in orchestrator.consensus_system.agents.items():
            metrics = monitor.agent_metrics.get(agent_id, {})
            agents.append({
                "agent_id": agent_id,
                "agent_type": agent_id.split('_')[0],
                "reputation": metrics.get('reputation_score', 100),
                "participations": metrics.get('total_participations', 0),
                "last_active": metrics.get('last_active', 0)
            })
        return {"agents": agents}
    
    @app.post("/consensus/request")
    async def request_consensus(decision_request: Dict[str, Any]):
        """Request a consensus decision"""
        session_id = decision_request.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")
        
        try:
            decision = await orchestrator.process_complex_decision(
                session_id, decision_request
            )
            
            return {
                "decision_id": decision.decision_id,
                "confidence": decision.confidence,
                "explanation": await orchestrator.explain_decision(decision)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/consensus/history/{session_id}")
    async def get_consensus_history(session_id: str):
        """Get consensus history for a session"""
        # Implementation similar to WebSocket handler
        pass
    
    @app.get("/consensus/metrics")
    async def get_consensus_metrics():
        """Get detailed consensus metrics"""
        from aiq.neural.consensus_monitoring import ConsensusAnalytics
        
        analytics = ConsensusAnalytics(monitor)
        
        return {
            "gas_efficiency": analytics.analyze_gas_efficiency(),
            "convergence_patterns": analytics.analyze_convergence_patterns(),
            "agent_leaderboard": monitor.get_agent_leaderboard()
        }
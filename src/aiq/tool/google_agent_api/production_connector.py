# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid
import time

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from .distributed_cache import DistributedCache
from .metrics import metrics
from .health_monitor import HealthMonitor
from .message_queue import MessageQueue

log = logging.getLogger(__name__)


@dataclass 
class AgentEndpoint:
    """Agent connection endpoint"""
    agent_id: str
    project_id: str
    location: str
    capabilities: List[str]
    metadata: Dict[str, Any]
    weight: int = 100  # For weighted routing
    circuit_breaker_state: str = "closed"
    last_failure: float = 0


class ProductionConnectorConfig(FunctionBaseConfig, name="production_connector"):
    """Production-ready multi-agent connector"""
    connections: List[Dict[str, Any]]
    redis_url: str = "redis://localhost:6379"
    amqp_url: str = "amqp://guest:guest@localhost/"
    enable_caching: bool = True
    cache_ttl: int = 300
    max_concurrent_calls: int = 20
    enable_health_checks: bool = True
    enable_metrics: bool = True
    enable_message_queue: bool = False


@register_function(config_type=ProductionConnectorConfig)
async def production_connector(config: ProductionConnectorConfig, builder: Builder):
    """Production multi-agent orchestrator"""
    
    # Initialize components
    cache = DistributedCache(config.redis_url) if config.enable_caching else None
    health_monitor = HealthMonitor() if config.enable_health_checks else None
    message_queue = MessageQueue(config.amqp_url) if config.enable_message_queue else None
    
    # Parse endpoints
    endpoints = {}
    capability_index = {}
    
    for conn in config.connections:
        endpoint = AgentEndpoint(
            agent_id=conn["agent_id"],
            project_id=conn["project_id"],
            location=conn.get("location", "us-central1"),
            capabilities=conn.get("capabilities", []),
            metadata=conn.get("metadata", {}),
            weight=conn.get("weight", 100)
        )
        
        endpoints[endpoint.agent_id] = endpoint
        
        # Build capability index
        for capability in endpoint.capabilities:
            if capability not in capability_index:
                capability_index[capability] = []
            capability_index[capability].append(endpoint)
    
    # Connection pool
    semaphore = asyncio.Semaphore(config.max_concurrent_calls)
    
    # Correlation ID for distributed tracing
    def get_correlation_id() -> str:
        return str(uuid.uuid4())
    
    async def call_agent_with_circuit_breaker(
        endpoint: AgentEndpoint,
        message: str,
        context: Dict[str, Any],
        correlation_id: str
    ) -> Dict[str, Any]:
        """Call agent with circuit breaker pattern"""
        
        # Check circuit breaker
        if endpoint.circuit_breaker_state == "open":
            if time.time() - endpoint.last_failure > 60:  # 1 minute recovery
                endpoint.circuit_breaker_state = "half-open"
            else:
                raise Exception(f"Circuit breaker open for {endpoint.agent_id}")
        
        try:
            # Import production client
            from .production_client import production_google_agent, ProductionGoogleAgentConfig
            
            client_config = ProductionGoogleAgentConfig(
                project_id=endpoint.project_id,
                location=endpoint.location,
                agent_id=endpoint.agent_id
            )
            
            client = await production_google_agent(client_config, builder)
            
            # Add correlation ID to context
            enhanced_context = {
                **context,
                "correlation_id": correlation_id,
                "source_connector": "production_connector"
            }
            
            with metrics.track_request(endpoint.agent_id):
                response = await client.func(
                    message=message,
                    context=enhanced_context
                )
            
            # Success - close circuit breaker
            if endpoint.circuit_breaker_state == "half-open":
                endpoint.circuit_breaker_state = "closed"
            
            return {
                "agent_id": endpoint.agent_id,
                "response": response,
                "status": "success",
                "correlation_id": correlation_id
            }
            
        except Exception as e:
            # Update circuit breaker
            endpoint.last_failure = time.time()
            if endpoint.circuit_breaker_state == "half-open":
                endpoint.circuit_breaker_state = "open"
            elif endpoint.circuit_breaker_state == "closed":
                # Increment failure count (simplified)
                endpoint.circuit_breaker_state = "open"
            
            metrics.update_circuit_breaker(endpoint.agent_id, endpoint.circuit_breaker_state)
            
            return {
                "agent_id": endpoint.agent_id,
                "error": str(e),
                "status": "error",
                "correlation_id": correlation_id
            }
    
    async def route_message(
        message: str,
        target_capabilities: List[str] = None,
        broadcast: bool = False,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Route message to appropriate agents"""
        
        correlation_id = get_correlation_id()
        start_time = time.time()
        
        # Check cache
        cache_key = f"route:{message}:{target_capabilities}:{broadcast}"
        if cache:
            cache_tracker = metrics.track_cache("route")
            cached = await cache.get(cache_key)
            if cached:
                cache_tracker.hit()
                return cached
            cache_tracker.miss()
        
        # Select target endpoints
        if target_capabilities:
            # Capability-based routing
            target_endpoints = []
            for capability in target_capabilities:
                target_endpoints.extend(capability_index.get(capability, []))
            
            # Deduplicate
            target_endpoints = list({ep.agent_id: ep for ep in target_endpoints}.values())
        else:
            target_endpoints = list(endpoints.values())
        
        if not target_endpoints:
            return {
                "error": "No agents found for capabilities",
                "capabilities": target_capabilities,
                "correlation_id": correlation_id
            }
        
        # Execute calls
        async def call_with_semaphore(endpoint):
            async with semaphore:
                return await call_agent_with_circuit_breaker(
                    endpoint, message, context or {}, correlation_id
                )
        
        if broadcast or len(target_endpoints) > 1:
            # Parallel calls
            tasks = [call_with_semaphore(ep) for ep in target_endpoints]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process responses
            processed_responses = []
            for i, response in enumerate(responses):
                if isinstance(response, Exception):
                    processed_responses.append({
                        "agent_id": target_endpoints[i].agent_id,
                        "error": str(response),
                        "status": "error"
                    })
                else:
                    processed_responses.append(response)
            
            result = {
                "broadcast": broadcast,
                "responses": processed_responses,
                "successful": sum(1 for r in processed_responses if r["status"] == "success"),
                "total_agents": len(target_endpoints),
                "correlation_id": correlation_id,
                "duration": time.time() - start_time
            }
            
        else:
            # Single call with weighted selection
            selected = select_weighted_endpoint(target_endpoints)
            response = await call_with_semaphore(selected)
            result = response
            result["duration"] = time.time() - start_time
        
        # Cache result
        if cache and result.get("status") == "success":
            await cache.set(cache_key, result, config.cache_ttl)
        
        # Publish to message queue if enabled
        if message_queue:
            await message_queue.publish(
                f"agent.response.{correlation_id}",
                result
            )
        
        return result
    
    def select_weighted_endpoint(endpoints: List[AgentEndpoint]) -> AgentEndpoint:
        """Select endpoint based on weights and circuit breaker state"""
        # Filter out open circuit breakers
        available = [ep for ep in endpoints if ep.circuit_breaker_state != "open"]
        
        if not available:
            # All circuit breakers open, try anyway
            available = endpoints
        
        # Weighted random selection
        total_weight = sum(ep.weight for ep in available)
        if total_weight == 0:
            return available[0]
        
        import random
        rand = random.uniform(0, total_weight)
        
        for endpoint in available:
            rand -= endpoint.weight
            if rand <= 0:
                return endpoint
        
        return available[-1]
    
    # Add health checks
    if health_monitor:
        for endpoint in endpoints.values():
            async def check_endpoint(ep=endpoint):
                try:
                    from .production_client import production_google_agent, ProductionGoogleAgentConfig
                    
                    config = ProductionGoogleAgentConfig(
                        project_id=ep.project_id,
                        location=ep.location,
                        agent_id=ep.agent_id,
                        timeout=5
                    )
                    
                    client = await production_google_agent(config, builder)
                    await client.func(message="ping")
                    
                except Exception:
                    raise Exception(f"Agent {ep.agent_id} health check failed")
            
            await health_monitor.add_check(f"agent_{endpoint.agent_id}", check_endpoint)
        
        await health_monitor.start()
    
    # Cleanup handler
    async def cleanup():
        if cache:
            await cache.close()
        if health_monitor:
            await health_monitor.stop()
        if message_queue:
            await message_queue.close()
    
    builder.add_cleanup_handler(cleanup)
    
    return FunctionInfo(
        name="production_connector",
        description="Production-ready multi-agent orchestrator",
        func=route_message,
        schema={
            "type": "function",
            "function": {
                "name": "production_connector",
                "description": "Route messages to multiple agents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to route"
                        },
                        "target_capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required capabilities"
                        },
                        "broadcast": {
                            "type": "boolean",
                            "description": "Broadcast to all matching agents"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
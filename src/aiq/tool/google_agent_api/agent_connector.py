# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Dict, List, Optional
import asyncio
from dataclasses import dataclass
import aioredis
import json
import time
from collections import defaultdict

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)

# Performance metrics tracking
metrics = defaultdict(lambda: {"calls": 0, "errors": 0, "total_time": 0})


@dataclass
class AgentConnection:
    """Represents a connection between agents"""
    agent_id: str
    project_id: str
    location: str
    capabilities: List[str]
    metadata: Dict[str, Any]


class AgentToAgentConnectorConfig(FunctionBaseConfig, name="agent_to_agent_connector"):
    """Configuration for agent-to-agent connector"""
    connections: List[Dict[str, Any]] = []
    enable_caching: bool = True
    cache_ttl: int = 300  # 5 minutes
    connection_timeout: int = 30
    max_concurrent_calls: int = 10


@register_function(config_type=AgentToAgentConnectorConfig)
async def agent_to_agent_connector(config: AgentToAgentConnectorConfig, builder: Builder):
    """Creates a connector for orchestrating communication between multiple agents"""
    
    # Parse connections from config
    connections = []
    for conn_config in config.connections:
        connections.append(
            AgentConnection(
                agent_id=conn_config["agent_id"],
                project_id=conn_config.get("project_id", ""),
                location=conn_config.get("location", "us-central1"),
                capabilities=conn_config.get("capabilities", []),
                metadata=conn_config.get("metadata", {})
            )
        )
    
    # Create capability index for faster routing
    capability_index = defaultdict(list)
    for conn in connections:
        for capability in conn.capabilities:
            capability_index[capability].append(conn)
    
    # Enhanced cache with TTL and size limits
    class LRUCache:
        def __init__(self, max_size=1000, ttl=300):
            self.cache = {}
            self.access_times = {}
            self.max_size = max_size
            self.ttl = ttl
            
        def get(self, key):
            if key in self.cache:
                cached_value, cache_time = self.cache[key]
                if time.time() - cache_time < self.ttl:
                    self.access_times[key] = time.time()
                    return cached_value
                else:
                    del self.cache[key]
                    del self.access_times[key]
            return None
            
        def set(self, key, value):
            # If cache is full, remove least recently used item
            if len(self.cache) >= self.max_size:
                lru_key = min(self.access_times, key=self.access_times.get)
                del self.cache[lru_key]
                del self.access_times[lru_key]
            
            self.cache[key] = (value, time.time())
            self.access_times[key] = time.time()
    
    response_cache = LRUCache() if config.enable_caching else None
    
    # Adaptive semaphore that adjusts based on error rates
    class AdaptiveSemaphore:
        def __init__(self, initial_limit=10):
            self.limit = initial_limit
            self.semaphore = asyncio.Semaphore(initial_limit)
            self.success_count = 0
            self.error_count = 0
            self.last_adjustment = time.time()
            
        async def acquire(self):
            return await self.semaphore.acquire()
            
        def release(self):
            self.semaphore.release()
            
        def record_result(self, success: bool):
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
                
            # Adjust limit every 60 seconds
            if time.time() - self.last_adjustment > 60:
                error_rate = self.error_count / (self.success_count + self.error_count + 1)
                
                if error_rate > 0.1:  # More than 10% errors
                    new_limit = max(1, int(self.limit * 0.8))
                elif error_rate < 0.05 and self.success_count > 100:  # Less than 5% errors
                    new_limit = min(50, int(self.limit * 1.2))
                else:
                    new_limit = self.limit
                    
                if new_limit != self.limit:
                    self.limit = new_limit
                    self.semaphore = asyncio.Semaphore(new_limit)
                    log.info(f"Adjusted semaphore limit to {new_limit}")
                    
                self.success_count = 0
                self.error_count = 0
                self.last_adjustment = time.time()
    
    semaphore = AdaptiveSemaphore(config.max_concurrent_calls)
    
    async def _route_to_agents(
        message: str,
        target_capabilities: List[str] = None,
        broadcast: bool = False,
        aggregate_responses: bool = True
    ) -> Dict[str, Any]:
        """Route messages to appropriate agents based on capabilities or broadcast"""
        
        start_time = time.time()
        
        # Use capability index for faster routing
        target_agents = connections
        if target_capabilities:
            agent_set = set()
            for capability in target_capabilities:
                agent_set.update(capability_index.get(capability, []))
            target_agents = list(agent_set)
        
        if not target_agents:
            return {
                "error": "No agents found with requested capabilities",
                "requested_capabilities": target_capabilities
            }
        
        # Check cache
        cache_key = f"{message}:{target_capabilities}:{broadcast}"
        if response_cache:
            cached_response = response_cache.get(cache_key)
            if cached_response:
                log.info("Returning cached response")
                metrics["cache"]["calls"] += 1
                return cached_response
        
        # Prepare tasks for concurrent execution
        tasks = []
        
        async def call_agent(agent: AgentConnection):
            await semaphore.acquire()
            agent_start_time = time.time()
            
            try:
                # Import the client function here to avoid circular imports
                from .agent_client import google_agent_api_client, GoogleAgentAPIConfig
                
                client_config = GoogleAgentAPIConfig(
                    project_id=agent.project_id,
                    location=agent.location,
                    agent_id=agent.agent_id,
                    timeout=config.connection_timeout
                )
                
                client_func = await google_agent_api_client(client_config, builder)
                response = await client_func.func(
                    message=message,
                    agent_id=agent.agent_id,
                    context={"source": "agent_connector", "metadata": agent.metadata}
                )
                
                # Track metrics
                metrics[agent.agent_id]["calls"] += 1
                metrics[agent.agent_id]["total_time"] += time.time() - agent_start_time
                semaphore.record_result(True)
                
                return {
                    "agent_id": agent.agent_id,
                    "response": response,
                    "status": "success",
                    "latency": time.time() - agent_start_time
                }
                
            except Exception as e:
                log.error(f"Error calling agent {agent.agent_id}: {str(e)}")
                
                # Track error metrics
                metrics[agent.agent_id]["errors"] += 1
                semaphore.record_result(False)
                
                return {
                    "agent_id": agent.agent_id,
                    "error": str(e),
                    "status": "error",
                    "latency": time.time() - agent_start_time
                }
            finally:
                semaphore.release()
        
        # Execute calls
        if broadcast or len(target_agents) > 1:
            # Call multiple agents concurrently
            for agent in target_agents:
                tasks.append(call_agent(agent))
            
            responses = await asyncio.gather(*tasks)
            
            # Calculate statistics
            total_latency = sum(r.get("latency", 0) for r in responses)
            avg_latency = total_latency / len(responses) if responses else 0
            
            result = {
                "broadcast": broadcast,
                "responses": responses,
                "total_agents": len(target_agents),
                "successful": sum(1 for r in responses if r["status"] == "success"),
                "average_latency": avg_latency,
                "total_time": time.time() - start_time
            }
            
            # Aggregate responses if requested
            if aggregate_responses and not broadcast:
                successful_responses = [
                    r["response"] for r in responses 
                    if r["status"] == "success"
                ]
                
                if successful_responses:
                    # Smart aggregation - combine responses intelligently
                    if len(successful_responses) == 1:
                        result["aggregated_response"] = successful_responses[0]
                    else:
                        # For financial analysis, provide structured aggregation
                        result["aggregated_response"] = {
                            "summary": "\n\n".join(successful_responses[:2]),  # Top responses
                            "all_responses": successful_responses,
                            "consensus_count": len(successful_responses)
                        }
            
        else:
            # Single agent call
            response = await call_agent(target_agents[0])
            result = response
        
        # Cache the result
        if response_cache:
            response_cache.set(cache_key, result)
        
        return result
    
    async def _get_metrics() -> Dict[str, Any]:
        """Get performance metrics for monitoring"""
        agent_metrics = {}
        for agent_id, stats in metrics.items():
            if stats["calls"] > 0:
                agent_metrics[agent_id] = {
                    "calls": stats["calls"],
                    "errors": stats["errors"],
                    "error_rate": stats["errors"] / stats["calls"],
                    "avg_latency": stats["total_time"] / stats["calls"]
                }
        
        return {
            "agents": agent_metrics,
            "cache_hit_rate": metrics["cache"]["calls"] / (metrics["cache"]["calls"] + metrics["cache"]["misses"] + 1),
            "semaphore_limit": semaphore.limit
        }
    
    async def _list_connected_agents() -> List[Dict[str, Any]]:
        """List all connected agents and their capabilities"""
        return [
            {
                "agent_id": conn.agent_id,
                "project_id": conn.project_id,
                "location": conn.location,
                "capabilities": conn.capabilities,
                "metadata": conn.metadata
            }
            for conn in connections
        ]
    
    return FunctionInfo(
        name="agent_to_agent_connector",
        description="Route messages to appropriate agents based on capabilities",
        func=_route_to_agents,
        schema={
            "type": "function",
            "function": {
                "name": "agent_to_agent_connector",
                "description": "Connect and route messages between multiple agents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to route to agents"
                        },
                        "target_capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required capabilities for target agents"
                        },
                        "broadcast": {
                            "type": "boolean",
                            "description": "Whether to broadcast to all matching agents"
                        },
                        "aggregate_responses": {
                            "type": "boolean",
                            "description": "Whether to aggregate responses from multiple agents"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
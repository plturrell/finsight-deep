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
import json
from datetime import datetime
import asyncio
import aiofiles
from collections import defaultdict
import time

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)

# In-memory cache for fast lookups
_registry_cache = {
    "agents": {},
    "capability_index": defaultdict(list),
    "last_update": 0
}


class AgentRegistryConfig(FunctionBaseConfig, name="agent_registry"):
    """Configuration for agent registry"""
    registry_file: str = "agent_registry.json"
    auto_discovery: bool = True
    refresh_interval: int = 3600  # 1 hour
    enable_health_check: bool = True


@register_function(config_type=AgentRegistryConfig)
async def agent_registry(config: AgentRegistryConfig, builder: Builder):
    """Creates an agent registry for managing and discovering available agents"""
    
    # Initialize or update cache
    async def _load_registry():
        try:
            # Check if cache is still valid
            if _registry_cache["last_update"] + config.refresh_interval > time.time():
                return
                
            async with aiofiles.open(config.registry_file, 'r') as f:
                content = await f.read()
                registry_data = json.loads(content)
                
            agents = registry_data.get("agents", [])
            
            # Update cache
            _registry_cache["agents"] = {agent["agent_id"]: agent for agent in agents}
            _registry_cache["capability_index"].clear()
            
            for agent in agents:
                for capability in agent.get("capabilities", []):
                    _registry_cache["capability_index"][capability].append(agent)
                    
            _registry_cache["last_update"] = time.time()
            
        except FileNotFoundError:
            # Initialize empty registry
            _registry_cache["agents"] = {}
            _registry_cache["capability_index"].clear()
            _registry_cache["last_update"] = time.time()
            
            registry_data = {"agents": [], "last_updated": datetime.utcnow().isoformat()}
            async with aiofiles.open(config.registry_file, 'w') as f:
                await f.write(json.dumps(registry_data, indent=2))
    
    # Ensure registry is loaded
    await _load_registry()
    
    async def _register_agent(
        agent_id: str,
        project_id: str,
        location: str,
        capabilities: List[str],
        metadata: Dict[str, Any] = None,
        endpoints: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Register a new agent or update existing one"""
        
        # Create or update agent entry
        agent_entry = {
            "agent_id": agent_id,
            "project_id": project_id,
            "location": location,
            "capabilities": capabilities,
            "metadata": metadata or {},
            "endpoints": endpoints or {},
            "registered_at": datetime.utcnow().isoformat(),
            "last_updated": datetime.utcnow().isoformat(),
            "status": "active"
        }
        
        # Check if exists in cache
        existing = _registry_cache["agents"].get(agent_id)
        
        # Update cache immediately
        _registry_cache["agents"][agent_id] = agent_entry
        
        # Update capability index
        if existing:
            # Remove from old capability index
            for cap in existing.get("capabilities", []):
                _registry_cache["capability_index"][cap] = [
                    a for a in _registry_cache["capability_index"][cap]
                    if a["agent_id"] != agent_id
                ]
        
        # Add to new capability index
        for capability in capabilities:
            _registry_cache["capability_index"][capability].append(agent_entry)
        
        # Async save to file
        asyncio.create_task(_save_registry(config.registry_file))
        
        return {
            "status": "success",
            "action": "updated" if existing else "registered",
            "agent": agent_entry
        }
    
    async def _discover_agents(
        capabilities: List[str] = None,
        project_id: str = None,
        location: str = None
    ) -> List[Dict[str, Any]]:
        """Discover agents based on capabilities or other criteria"""
        
        await _load_registry()  # Ensure cache is up to date
        
        # Fast capability-based lookup using index
        if capabilities:
            agent_set = set()
            for cap in capabilities:
                agents = _registry_cache["capability_index"].get(cap, [])
                agent_set.update(agent["agent_id"] for agent in agents)
            
            matched_agents = [
                _registry_cache["agents"][agent_id]
                for agent_id in agent_set
                if agent_id in _registry_cache["agents"]
            ]
        else:
            matched_agents = list(_registry_cache["agents"].values())
        
        # Additional filters
        if project_id:
            matched_agents = [a for a in matched_agents if a["project_id"] == project_id]
        
        if location:
            matched_agents = [a for a in matched_agents if a["location"] == location]
        
        # Async health check if enabled
        if config.enable_health_check:
            health_tasks = []
            for agent in matched_agents:
                health_tasks.append(_check_agent_health(agent))
            
            health_results = await asyncio.gather(*health_tasks, return_exceptions=True)
            
            for agent, health in zip(matched_agents, health_results):
                if isinstance(health, Exception):
                    agent["health_status"] = "unknown"
                else:
                    agent["health_status"] = health
        
        return matched_agents
    
    async def _unregister_agent(agent_id: str) -> Dict[str, Any]:
        """Remove an agent from the registry"""
        
        if agent_id in _registry_cache["agents"]:
            removed_agent = _registry_cache["agents"].pop(agent_id)
            
            # Remove from capability index
            for cap in removed_agent.get("capabilities", []):
                _registry_cache["capability_index"][cap] = [
                    a for a in _registry_cache["capability_index"][cap]
                    if a["agent_id"] != agent_id
                ]
            
            # Async save to file
            asyncio.create_task(_save_registry(config.registry_file))
            
            log.info(f"Unregistered agent {agent_id}")
            
            return {
                "status": "success",
                "action": "unregistered",
                "agent": removed_agent
            }
        
        return {
            "status": "error",
            "message": f"Agent {agent_id} not found in registry"
        }
    
    async def _get_agent_details(agent_id: str) -> Optional[Dict[str, Any]]:
        """Get details for a specific agent"""
        
        agent = _registry_cache["agents"].get(agent_id)
        
        if agent:
            # Add real-time health check if enabled
            if config.enable_health_check:
                try:
                    health = await _check_agent_health(agent)
                    agent["health_status"] = health
                except Exception:
                    agent["health_status"] = "unknown"
            
            return agent
        
        return None
    
    async def _save_registry(file_path: str):
        """Save registry to file asynchronously"""
        agents_list = list(_registry_cache["agents"].values())
        registry_data = {
            "agents": agents_list,
            "last_updated": datetime.utcnow().isoformat()
        }
        
        async with aiofiles.open(file_path, 'w') as f:
            await f.write(json.dumps(registry_data, indent=2))
    
    async def _check_agent_health(agent: Dict[str, Any]) -> str:
        """Check if an agent is healthy"""
        # Simple health check - could be expanded to actually ping the agent
        return "healthy" if agent.get("status") == "active" else "unhealthy"
    
    return FunctionInfo(
        name="agent_registry",
        description="Manage and discover available agents",
        func=_register_agent,
        schema={
            "type": "function",
            "function": {
                "name": "agent_registry",
                "description": "Register, discover, and manage agents",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["register", "discover", "unregister", "get_details"],
                            "description": "The action to perform"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "The agent ID"
                        },
                        "project_id": {
                            "type": "string",
                            "description": "The Google Cloud project ID"
                        },
                        "location": {
                            "type": "string",
                            "description": "The agent location"
                        },
                        "capabilities": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Agent capabilities"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Additional agent metadata"
                        }
                    },
                    "required": ["action"]
                }
            }
        }
    )
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
import aiohttp
from google.auth import default
from google.auth.transport.requests import AuthorizedSession
import asyncio
import time
from contextlib import asynccontextmanager

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)

# Global connection pool - shared across instances for better efficiency
_global_connector = None
_global_session = None

async def get_global_session():
    global _global_connector, _global_session
    if _global_session is None:
        _global_connector = aiohttp.TCPConnector(
            limit=100,
            limit_per_host=30,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False
        )
        _global_session = aiohttp.ClientSession(
            connector=_global_connector,
            json_serialize=json.dumps
        )
    return _global_session


class GoogleAgentAPIConfig(FunctionBaseConfig, name="google_agent_api"):
    """Configuration for Google Agent API integration"""
    project_id: str
    location: str = "us-central1"
    agent_id: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3


@register_function(config_type=GoogleAgentAPIConfig)
async def google_agent_api_client(config: GoogleAgentAPIConfig, builder: Builder):
    """Creates a Google Agent API client for agent-to-agent communication"""
    
    # Create a simple cache with TTL
    cache = {}
    cache_ttl = 300  # 5 minutes
    
    # Circuit breaker state
    circuit_state = {"failures": 0, "last_failure": 0, "state": "CLOSED"}
    
    async def _call_google_agent(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Call a Google Agent and return its response"""
        
        target_agent_id = agent_id or config.agent_id
        if not target_agent_id:
            raise ValueError("Agent ID must be provided either in config or function call")
        
        # Check circuit breaker
        if circuit_state["state"] == "OPEN":
            if time.time() - circuit_state["last_failure"] > 60:  # 1 minute recovery
                circuit_state["state"] = "HALF_OPEN"
            else:
                raise Exception("Circuit breaker OPEN - agent temporarily unavailable")
        
        # Check cache
        cache_key = f"{target_agent_id}:{message}:{json.dumps(context or {})}"
        if cache_key in cache:
            cached_response, cached_time = cache[cache_key]
            if time.time() - cached_time < cache_ttl:
                log.info("Returning cached response")
                return cached_response
            else:
                del cache[cache_key]
            
        try:
            # Get default credentials
            credentials, project = default()
            
            # Construct endpoint URL
            base_url = f"https://{config.location}-aiplatform.googleapis.com/v1beta1"
            endpoint = f"{base_url}/projects/{config.project_id}/locations/{config.location}/agents/{target_agent_id}:converse"
            
            # Prepare request payload
            payload = {
                "messages": [
                    {
                        "content": message,
                        "role": "user"
                    }
                ]
            }
            
            if context:
                payload["context"] = context
                
            # Use global session for connection pooling
            session = await get_global_session()
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {credentials.token}"
            }
            
            for attempt in range(config.max_retries):
                try:
                    async with session.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=config.timeout)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result["reply"]["content"]
                            
                            # Reset circuit breaker on success
                            if circuit_state["state"] != "CLOSED":
                                circuit_state["state"] = "CLOSED"
                                circuit_state["failures"] = 0
                            
                            # Cache the response
                            cache[cache_key] = (content, time.time())
                            
                            return content
                        else:
                            error_text = await response.text()
                            log.error(f"Google Agent API error: {response.status} - {error_text}")
                            
                            if attempt < config.max_retries - 1:
                                await asyncio.sleep(min(2 ** attempt, 8))  # Exponential backoff
                                continue
                                
                            raise Exception(f"Google Agent API error: {response.status} - {error_text}")
                                
                except aiohttp.ClientTimeout:
                    log.error(f"Request timeout after {config.timeout} seconds")
                    if attempt < config.max_retries - 1:
                        await asyncio.sleep(min(2 ** attempt, 8))
                        continue
                    raise
                        
        except Exception as e:
            # Update circuit breaker
            circuit_state["failures"] += 1
            circuit_state["last_failure"] = time.time()
            if circuit_state["failures"] >= 5:
                circuit_state["state"] = "OPEN"
                log.warning("Circuit breaker opened due to repeated failures")
            
            log.error(f"Error calling Google Agent: {str(e)}")
            raise
            
    return FunctionInfo(
        name="google_agent_api",
        description="Call Google Agent API for agent-to-agent communication",
        func=_call_google_agent,
        schema={
            "type": "function",
            "function": {
                "name": "google_agent_api",
                "description": "Call a Google Agent using the Agent API",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the agent"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "The target agent ID (optional if set in config)"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context to pass to the agent"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
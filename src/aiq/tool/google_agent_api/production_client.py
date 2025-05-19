# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from typing import Any, Dict, Optional
import aiohttp
from google.auth import default
from google.auth.transport.requests import Request
import time
from contextlib import asynccontextmanager
import ujson as json  # Faster JSON
from tenacity import retry, stop_after_attempt, wait_exponential

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)


class ProductionGoogleAgentConfig(FunctionBaseConfig, name="production_google_agent"):
    """Production-ready Google Agent configuration"""
    project_id: str
    location: str = "us-central1"
    agent_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 20  # Reasonable default
    keepalive_timeout: int = 30
    enable_compression: bool = True


class TokenManager:
    """Manage Google auth tokens with automatic refresh"""
    
    def __init__(self):
        self.credentials = None
        self.token = None
        self.token_expiry = 0
        self._lock = asyncio.Lock()
    
    async def get_token(self) -> str:
        async with self._lock:
            if self.token and time.time() < self.token_expiry - 300:  # 5 min buffer
                return self.token
                
            # Refresh token
            loop = asyncio.get_event_loop()
            self.credentials, _ = await loop.run_in_executor(None, default)
            
            # Get fresh token
            request = Request()
            await loop.run_in_executor(None, self.credentials.refresh, request)
            
            self.token = self.credentials.token
            self.token_expiry = self.credentials.expiry.timestamp()
            
            return self.token


# Singleton token manager
_token_manager = TokenManager()


class HttpClient:
    """Optimized HTTP client with connection pooling"""
    
    def __init__(self, config: ProductionGoogleAgentConfig):
        self.config = config
        self._session = None
        
    async def __aenter__(self):
        if not self._session:
            timeout = aiohttp.ClientTimeout(
                total=self.config.timeout,
                connect=5,
                sock_read=self.config.timeout
            )
            
            connector = aiohttp.TCPConnector(
                limit=self.config.connection_pool_size,
                limit_per_host=10,
                ttl_dns_cache=300,
                enable_cleanup_closed=True,
                keepalive_timeout=self.config.keepalive_timeout
            )
            
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                json_serialize=json.dumps
            )
        return self._session
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()


@register_function(config_type=ProductionGoogleAgentConfig)
async def production_google_agent(config: ProductionGoogleAgentConfig, builder: Builder):
    """Production-ready Google Agent client"""
    
    http_client = HttpClient(config)
    
    # Add cleanup handler
    async def cleanup():
        async with http_client as session:
            await session.close()
    
    builder.add_cleanup_handler(cleanup)
    
    @retry(
        stop=stop_after_attempt(config.max_retries),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True
    )
    async def _call_agent(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None
    ) -> str:
        """Call Google Agent with production-grade reliability"""
        
        target_agent_id = agent_id or config.agent_id
        if not target_agent_id:
            raise ValueError("Agent ID must be provided")
        
        # Get auth token
        token = await _token_manager.get_token()
        
        # Build request
        # Note: Using Dialogflow CX API which is the actual Google agent API
        endpoint = (
            f"https://{config.location}-dialogflow.googleapis.com/v3/"
            f"projects/{config.project_id}/locations/{config.location}/"
            f"agents/{target_agent_id}/sessions/default:detectIntent"
        )
        
        payload = {
            "queryInput": {
                "text": {
                    "text": message
                },
                "languageCode": "en"
            }
        }
        
        if context:
            payload["queryParams"] = {"parameters": context}
        
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept-Encoding": "gzip" if config.enable_compression else "identity"
        }
        
        # Make request
        async with http_client as session:
            async with session.post(
                endpoint,
                json=payload,
                headers=headers,
                compress=config.enable_compression
            ) as response:
                response.raise_for_status()
                
                result = await response.json()
                
                # Extract response from Dialogflow format
                if "queryResult" in result:
                    return result["queryResult"]["responseMessages"][0]["text"]["text"][0]
                else:
                    raise ValueError("Invalid response format from Google Agent")
    
    return FunctionInfo(
        name="production_google_agent",
        description="Production-ready Google Agent client with Dialogflow CX",
        func=_call_agent,
        schema={
            "type": "function",
            "function": {
                "name": "production_google_agent",
                "description": "Call Google Dialogflow CX agent",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send to the agent"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "The Dialogflow agent ID"
                        },
                        "context": {
                            "type": "object",
                            "description": "Context parameters"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
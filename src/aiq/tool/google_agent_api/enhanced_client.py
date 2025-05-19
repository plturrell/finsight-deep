# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import time
from typing import Any, Dict, Optional
import aiohttp
from google.auth import default
import json
from contextlib import asynccontextmanager
import backoff

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

log = logging.getLogger(__name__)


class EnhancedGoogleAgentConfig(FunctionBaseConfig, name="enhanced_google_agent"):
    """Enhanced configuration with performance optimizations"""
    project_id: str
    location: str = "us-central1"
    agent_id: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 100
    max_connections_per_host: int = 30


class ConnectionPool:
    """Manage persistent connections efficiently"""
    
    def __init__(self, size: int = 100, per_host: int = 30):
        self.connector = aiohttp.TCPConnector(
            limit=size,
            limit_per_host=per_host,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            force_close=False
        )
        self.session = None
        
    @asynccontextmanager
    async def get_session(self):
        if not self.session:
            self.session = aiohttp.ClientSession(
                connector=self.connector,
                json_serialize=json.dumps
            )
        yield self.session
        
    async def cleanup(self):
        if self.session:
            await self.session.close()


class CircuitBreaker:
    """Prevent cascade failures"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
        
    def record_success(self):
        if self.state == "HALF_OPEN":
            self.state = "CLOSED"
            self.failure_count = 0
            
    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            
    def is_open(self) -> bool:
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "HALF_OPEN"
                return False
            return True
        return False


class RequestBatcher:
    """Batch multiple requests for efficiency"""
    
    def __init__(self, batch_size: int = 10, timeout: float = 0.1):
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = asyncio.Queue()
        self.results = {}
        self.processing = False
        
    async def add_request(self, request_id: str, request_data: Dict) -> Any:
        future = asyncio.Future()
        await self.queue.put((request_id, request_data, future))
        
        if not self.processing:
            asyncio.create_task(self._process_batch())
            
        return await future
        
    async def _process_batch(self):
        self.processing = True
        batch = []
        
        try:
            # Collect requests up to batch size or timeout
            while len(batch) < self.batch_size:
                try:
                    request = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=self.timeout
                    )
                    batch.append(request)
                except asyncio.TimeoutError:
                    break
                    
            if batch:
                # Process batch
                results = await self._execute_batch([req[1] for req in batch])
                
                # Return results to futures
                for i, (req_id, _, future) in enumerate(batch):
                    if i < len(results):
                        future.set_result(results[i])
                    else:
                        future.set_exception(Exception("Batch processing failed"))
                        
        finally:
            self.processing = False
            
    async def _execute_batch(self, requests: list) -> list:
        # Implementation would send batched requests to API
        return requests  # Placeholder


@register_function(config_type=EnhancedGoogleAgentConfig)
async def enhanced_google_agent_client(config: EnhancedGoogleAgentConfig, builder: Builder):
    """Enhanced Google Agent client with performance optimizations"""
    
    # Initialize components
    connection_pool = ConnectionPool(config.connection_pool_size, config.max_connections_per_host)
    circuit_breaker = CircuitBreaker()
    request_batcher = RequestBatcher()
    
    # Cache for responses
    response_cache = {}
    
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, asyncio.TimeoutError),
        max_tries=config.max_retries,
        max_time=config.timeout
    )
    async def _call_agent_with_retry(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None,
        use_batch: bool = False
    ) -> str:
        """Enhanced agent call with optimizations"""
        
        # Check circuit breaker
        if circuit_breaker.is_open():
            raise Exception("Circuit breaker is open - agent unavailable")
            
        target_agent_id = agent_id or config.agent_id
        if not target_agent_id:
            raise ValueError("Agent ID must be provided")
            
        # Check cache
        cache_key = f"{target_agent_id}:{message}:{json.dumps(context or {})}"
        if cache_key in response_cache:
            log.info("Returning cached response")
            return response_cache[cache_key]
            
        try:
            # Use batching if enabled
            if use_batch:
                request_data = {
                    "agent_id": target_agent_id,
                    "message": message,
                    "context": context
                }
                result = await request_batcher.add_request(cache_key, request_data)
                circuit_breaker.record_success()
                response_cache[cache_key] = result
                return result
                
            # Direct API call
            credentials, project = default()
            
            endpoint = (f"https://{config.location}-aiplatform.googleapis.com/v1beta1"
                       f"/projects/{config.project_id}/locations/{config.location}"
                       f"/agents/{target_agent_id}:converse")
            
            payload = {
                "messages": [{"content": message, "role": "user"}]
            }
            if context:
                payload["context"] = context
                
            async with connection_pool.get_session() as session:
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {credentials.token}"
                }
                
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=config.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        circuit_breaker.record_success()
                        
                        # Cache successful response
                        response_text = result["reply"]["content"]
                        response_cache[cache_key] = response_text
                        
                        return response_text
                    else:
                        error_text = await response.text()
                        circuit_breaker.record_failure()
                        raise Exception(f"API error: {response.status} - {error_text}")
                        
        except Exception as e:
            circuit_breaker.record_failure()
            log.error(f"Error calling agent: {str(e)}")
            raise
            
    # Cleanup function
    async def cleanup():
        await connection_pool.cleanup()
        
    builder.add_cleanup_handler(cleanup)
    
    return FunctionInfo(
        name="enhanced_google_agent",
        description="Enhanced Google Agent client with connection pooling and circuit breaker",
        func=_call_agent_with_retry,
        schema={
            "type": "function",
            "function": {
                "name": "enhanced_google_agent",
                "description": "Call Google Agent with enhanced performance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "The message to send"
                        },
                        "agent_id": {
                            "type": "string",
                            "description": "Target agent ID"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context"
                        },
                        "use_batch": {
                            "type": "boolean",
                            "description": "Use request batching"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
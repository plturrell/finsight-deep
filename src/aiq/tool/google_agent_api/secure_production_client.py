# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, Optional
import asyncio
import time

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from .production_client import TokenManager, HttpClient
from .auth import AuthenticationManager, AuthorizationManager, Permission
from .secrets import SecretManager, create_secret_provider
from .validation import InputValidator, AgentMessage
from .thread_safe import ThreadSafeCircuitBreaker, ThreadSafeCache
from .metrics import metrics
from .health_monitor import HealthMonitor

log = logging.getLogger(__name__)


class SecureProductionConfig(FunctionBaseConfig, name="secure_production_agent"):
    """Secure production configuration with all enhancements"""
    project_id: str
    location: str = "us-central1"
    agent_id: Optional[str] = None
    
    # Security settings
    secret_provider_type: str = "encrypted_file"
    secret_provider_config: Dict[str, Any] = {}
    require_auth: bool = True
    
    # Performance settings
    cache_enabled: bool = True
    cache_ttl: int = 300
    circuit_breaker_enabled: bool = True
    
    # Connection settings
    timeout: int = 30
    max_retries: int = 3
    connection_pool_size: int = 20


@register_function(config_type=SecureProductionConfig)
async def secure_production_agent(config: SecureProductionConfig, builder: Builder):
    """Production Google Agent with all security and reliability features"""
    
    # Initialize security components
    secret_provider = create_secret_provider(
        config.secret_provider_type,
        **config.secret_provider_config
    )
    secret_manager = SecretManager(secret_provider)
    
    auth_manager = AuthenticationManager()
    authz_manager = AuthorizationManager()
    
    # Initialize reliability components
    cache = ThreadSafeCache() if config.cache_enabled else None
    circuit_breaker = ThreadSafeCircuitBreaker() if config.circuit_breaker_enabled else None
    health_monitor = HealthMonitor()
    
    # Initialize token manager
    token_manager = TokenManager()
    
    # Initialize HTTP client
    http_client = HttpClient(config)
    
    # Health check function
    async def agent_health_check():
        """Check agent health"""
        try:
            # Simple ping to verify connectivity
            config_copy = config.copy()
            config_copy.timeout = 5
            
            test_message = AgentMessage(message="ping")
            await _call_agent_internal(
                test_message.message,
                agent_id=config.agent_id,
                skip_cache=True
            )
        except Exception as e:
            raise Exception(f"Agent health check failed: {e}")
    
    # Add health check
    await health_monitor.add_check(f"agent_{config.agent_id}", agent_health_check)
    await health_monitor.start()
    
    async def _call_agent_internal(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None,
        skip_cache: bool = False
    ) -> str:
        """Internal agent call with all protections"""
        
        target_agent_id = agent_id or config.agent_id
        if not target_agent_id:
            raise ValueError("Agent ID required")
        
        # Check circuit breaker
        if circuit_breaker and not circuit_breaker.can_execute():
            raise Exception("Circuit breaker OPEN - agent unavailable")
        
        # Check cache
        cache_key = f"{target_agent_id}:{message}:{json.dumps(context or {})}"
        
        if cache and not skip_cache:
            cached_result = cache.get(cache_key)
            if cached_result:
                metrics.track_cache("agent").hit()
                return cached_result
            else:
                metrics.track_cache("agent").miss()
        
        try:
            # Get auth token
            token = await token_manager.get_token()
            
            # Build request
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
                "Content-Type": "application/json"
            }
            
            # Make request with metrics
            with metrics.track_request(target_agent_id):
                async with http_client as session:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=headers
                    ) as response:
                        response.raise_for_status()
                        result = await response.json()
                        
                        # Extract response
                        if "queryResult" in result:
                            content = result["queryResult"]["responseMessages"][0]["text"]["text"][0]
                            
                            # Cache result
                            if cache and not skip_cache:
                                cache.set(cache_key, content)
                            
                            # Record success
                            if circuit_breaker:
                                circuit_breaker.record_success()
                            
                            return content
                        else:
                            raise ValueError("Invalid response format")
        
        except Exception as e:
            # Record failure
            if circuit_breaker:
                circuit_breaker.record_failure()
            
            metrics.update_circuit_breaker(
                target_agent_id,
                circuit_breaker.get_state() if circuit_breaker else "closed"
            )
            
            log.error(f"Agent call failed: {e}")
            raise
    
    async def secure_call_agent(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None,
        auth_token: Optional[str] = None
    ) -> str:
        """Secure agent call with validation and authorization"""
        
        # Validate input
        validated_msg = AgentMessage(
            message=message,
            context=context
        )
        
        # Check authentication if required
        if config.require_auth:
            if not auth_token:
                raise ValueError("Authentication required")
            
            auth_info = await auth_manager.verify_token(auth_token)
            if not auth_info:
                raise ValueError("Invalid authentication token")
            
            # Check authorization
            resource = f"agent:{agent_id or config.agent_id}"
            if not await authz_manager.check_permission(
                auth_info,
                resource,
                Permission.EXECUTE
            ):
                raise ValueError("Insufficient permissions")
            
            # Add auth context
            enhanced_context = {
                **(context or {}),
                "user_id": auth_info.user_id,
                "roles": list(auth_info.roles)
            }
        else:
            enhanced_context = context
        
        # Call agent with all protections
        return await _call_agent_internal(
            validated_msg.message,
            agent_id=agent_id,
            context=enhanced_context
        )
    
    # Cleanup handler
    async def cleanup():
        await health_monitor.stop()
        await http_client.__aexit__(None, None, None)
    
    builder.add_cleanup_handler(cleanup)
    
    return FunctionInfo(
        name="secure_production_agent",
        description="Secure production Google Agent with full protection",
        func=secure_call_agent,
        schema={
            "type": "function",
            "function": {
                "name": "secure_production_agent",
                "description": "Call Google Agent with security and reliability",
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
                        "auth_token": {
                            "type": "string",
                            "description": "Authentication token"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
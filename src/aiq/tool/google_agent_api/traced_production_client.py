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

from .secure_production_client import SecureProductionConfig, secure_production_agent
from .tracing import (
    DistributedTracer, TracingMiddleware, create_traced_components,
    FinancialAttributes, TraceContext
)
from .thread_safe import ThreadSafeCache, ThreadSafeCircuitBreaker
from .metrics import metrics

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

log = logging.getLogger(__name__)


class TracedProductionConfig(SecureProductionConfig, name="traced_production_agent"):
    """Production config with distributed tracing"""
    
    # Tracing settings
    enable_tracing: bool = True
    otlp_endpoint: str = "localhost:4317"
    service_name: str = "google-agent-api"
    sample_rate: float = 1.0
    console_export: bool = False


@register_function(config_type=TracedProductionConfig)
async def traced_production_agent(config: TracedProductionConfig, builder: Builder):
    """Production agent with full distributed tracing"""
    
    # Initialize tracer
    tracer = None
    if config.enable_tracing:
        tracer = DistributedTracer(
            service_name=config.service_name,
            otlp_endpoint=config.otlp_endpoint,
            console_export=config.console_export,
            sample_rate=config.sample_rate
        )
    
    # Create base agent
    base_agent = await secure_production_agent(config, builder)
    
    # Initialize components
    cache = ThreadSafeCache() if config.cache_enabled else None
    circuit_breaker = ThreadSafeCircuitBreaker() if config.circuit_breaker_enabled else None
    
    # Wrap components with tracing
    if tracer:
        traced_components = create_traced_components(tracer, {
            "cache": cache,
            "circuit_breaker": circuit_breaker
        })
        cache = traced_components.get("cache")
        circuit_breaker = traced_components.get("circuit_breaker")
    
    async def traced_call_agent(
        message: str,
        agent_id: str = None,
        context: Dict[str, Any] = None,
        auth_token: Optional[str] = None,
        trace_context: Optional[TraceContext] = None
    ) -> str:
        """Agent call with distributed tracing"""
        
        # Extract parent trace context
        parent_context = trace_context or (context.get("trace_context") if context else None)
        
        # Create or continue trace
        span_attributes = {
            "agent.id": agent_id or config.agent_id,
            "message.length": len(message),
            "has_auth": bool(auth_token)
        }
        
        # Add financial attributes if applicable
        if context and "financial_analysis" in context:
            span_attributes.update(
                FinancialAttributes.for_analysis(
                    context["financial_analysis"],
                    context.get("market")
                )
            )
        
        async with tracer.trace_async_operation(
            "agent.call.complete",
            kind=SpanKind.CLIENT,
            attributes=span_attributes
        ) as span:
            
            start_time = time.time()
            
            try:
                # Validation phase
                async with tracer.trace_async_operation(
                    "agent.validation",
                    kind=SpanKind.INTERNAL
                ) as validation_span:
                    # Input validation happens in base agent
                    validation_start = time.time()
                    
                    # Add validation timing
                    if validation_span:
                        validation_span.set_attribute(
                            "validation.duration_ms",
                            (time.time() - validation_start) * 1000
                        )
                
                # Authentication phase
                if auth_token:
                    async with tracer.trace_async_operation(
                        "agent.authentication",
                        kind=SpanKind.INTERNAL
                    ) as auth_span:
                        auth_start = time.time()
                        
                        if auth_span:
                            auth_span.set_attribute(
                                "auth.duration_ms",
                                (time.time() - auth_start) * 1000
                            )
                
                # Cache lookup
                cache_key = f"{agent_id or config.agent_id}:{message}:{context}"
                cache_result = None
                
                if cache:
                    async with tracer.trace_async_operation(
                        "cache.lookup",
                        kind=SpanKind.INTERNAL
                    ) as cache_span:
                        cache_result = cache.get(cache_key)
                        
                        if cache_span:
                            cache_span.set_attribute("cache.hit", bool(cache_result))
                
                if cache_result:
                    if span:
                        span.set_attribute("cache.hit", True)
                        span.set_attribute("response.cached", True)
                    return cache_result
                
                # Circuit breaker check
                if circuit_breaker:
                    async with tracer.trace_async_operation(
                        "circuit_breaker.check",
                        kind=SpanKind.INTERNAL
                    ) as cb_span:
                        if not circuit_breaker.can_execute():
                            if cb_span:
                                cb_span.set_attribute("circuit_breaker.open", True)
                            raise Exception("Circuit breaker OPEN")
                
                # Actual agent call
                async with tracer.trace_async_operation(
                    "agent.http_call",
                    kind=SpanKind.CLIENT,
                    attributes={
                        "http.method": "POST",
                        "http.url": f"https://{config.location}-dialogflow.googleapis.com/v3/agents/{agent_id}",
                        "net.peer.name": f"{config.location}-dialogflow.googleapis.com"
                    }
                ) as http_span:
                    
                    # Add trace context to request
                    enhanced_context = context or {}
                    if http_span:
                        span_context = http_span.get_span_context()
                        enhanced_context.update({
                            "trace_id": format(span_context.trace_id, "032x"),
                            "span_id": format(span_context.span_id, "016x"),
                            "parent_id": format(span.get_span_context().span_id, "016x") if span else None
                        })
                    
                    # Call base agent
                    result = await base_agent.func(
                        message=message,
                        agent_id=agent_id,
                        context=enhanced_context,
                        auth_token=auth_token
                    )
                    
                    # Cache result
                    if cache:
                        async with tracer.trace_async_operation(
                            "cache.set",
                            kind=SpanKind.INTERNAL
                        ) as cache_set_span:
                            cache.set(cache_key, result)
                    
                    # Record success
                    if circuit_breaker:
                        circuit_breaker.record_success()
                    
                    # Add response attributes
                    if span:
                        duration = time.time() - start_time
                        span.set_attributes({
                            "agent.response_time_ms": duration * 1000,
                            "agent.response_length": len(result),
                            "agent.success": True
                        })
                        span.set_status(Status(StatusCode.OK))
                    
                    return result
            
            except Exception as e:
                # Record failure
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                # Record exception in span
                if span:
                    span.record_exception(e)
                    span.set_attributes({
                        "agent.error": str(e),
                        "agent.success": False,
                        "agent.response_time_ms": (time.time() - start_time) * 1000
                    })
                    span.set_status(Status(StatusCode.ERROR, str(e)))
                
                raise
    
    # Create tracing middleware if needed
    if tracer:
        middleware = TracingMiddleware(tracer)
        builder.add_middleware(middleware)
    
    return FunctionInfo(
        name="traced_production_agent",
        description="Production agent with distributed tracing",
        func=traced_call_agent,
        schema={
            "type": "function",
            "function": {
                "name": "traced_production_agent",
                "description": "Call agent with full tracing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "message": {
                            "type": "string",
                            "description": "Message to send"
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
                            "description": "Auth token"
                        },
                        "trace_context": {
                            "type": "object",
                            "description": "Parent trace context"
                        }
                    },
                    "required": ["message"]
                }
            }
        }
    )
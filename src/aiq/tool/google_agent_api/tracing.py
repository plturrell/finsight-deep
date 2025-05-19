# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from typing import Any, Dict, Optional, List
from contextlib import asynccontextmanager, contextmanager
import asyncio
from dataclasses import dataclass
import json
import uuid

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, SpanKind
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor, ConsoleSpanExporter
)
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.semconv.trace import SpanAttributes

log = logging.getLogger(__name__)


@dataclass
class TraceContext:
    """Context for distributed tracing"""
    trace_id: str
    span_id: str
    parent_id: Optional[str] = None
    baggage: Dict[str, str] = None
    
    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers for propagation"""
        headers = {
            "x-trace-id": self.trace_id,
            "x-span-id": self.span_id
        }
        if self.parent_id:
            headers["x-parent-id"] = self.parent_id
        if self.baggage:
            headers["x-baggage"] = json.dumps(self.baggage)
        return headers
    
    @classmethod
    def from_headers(cls, headers: Dict[str, str]) -> Optional['TraceContext']:
        """Create from HTTP headers"""
        if "x-trace-id" not in headers:
            return None
        
        baggage = None
        if "x-baggage" in headers:
            try:
                baggage = json.loads(headers["x-baggage"])
            except:
                pass
        
        return cls(
            trace_id=headers["x-trace-id"],
            span_id=headers.get("x-span-id", ""),
            parent_id=headers.get("x-parent-id"),
            baggage=baggage
        )


class DistributedTracer:
    """Distributed tracing implementation"""
    
    def __init__(self, 
                 service_name: str,
                 otlp_endpoint: str = None,
                 console_export: bool = False,
                 sample_rate: float = 1.0):
        self.service_name = service_name
        self.sample_rate = sample_rate
        
        # Create resource
        resource = Resource.create({
            "service.name": service_name,
            "service.version": "1.0.0",
            "deployment.environment": "production"
        })
        
        # Create tracer provider
        provider = TracerProvider(resource=resource)
        
        # Add exporters
        if otlp_endpoint:
            otlp_exporter = OTLPSpanExporter(
                endpoint=otlp_endpoint,
                insecure=True  # For development
            )
            provider.add_span_processor(
                BatchSpanProcessor(otlp_exporter)
            )
        
        if console_export:
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(
                BatchSpanProcessor(console_exporter)
            )
        
        # Set as global provider
        trace.set_tracer_provider(provider)
        
        # Get tracer
        self.tracer = trace.get_tracer(service_name)
        
        # Propagator for context propagation
        self.propagator = TraceContextTextMapPropagator()
        
        # Instrument libraries
        self._instrument_libraries()
    
    def _instrument_libraries(self):
        """Auto-instrument common libraries"""
        try:
            AioHttpClientInstrumentor().instrument()
            RedisInstrumentor().instrument()
        except Exception as e:
            log.warning(f"Failed to instrument libraries: {e}")
    
    @contextmanager
    def trace_operation(self, 
                       operation_name: str,
                       kind: SpanKind = SpanKind.INTERNAL,
                       attributes: Dict[str, Any] = None):
        """Trace a synchronous operation"""
        
        # Check sampling
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                yield None
                return
        
        with self.tracer.start_as_current_span(
            operation_name,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    @asynccontextmanager
    async def trace_async_operation(self,
                                   operation_name: str,
                                   kind: SpanKind = SpanKind.INTERNAL,
                                   attributes: Dict[str, Any] = None):
        """Trace an asynchronous operation"""
        
        # Check sampling
        if self.sample_rate < 1.0:
            import random
            if random.random() > self.sample_rate:
                yield None
                return
        
        with self.tracer.start_as_current_span(
            operation_name,
            kind=kind,
            attributes=attributes or {}
        ) as span:
            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                raise
    
    def inject_context(self, headers: Dict[str, str]) -> Dict[str, str]:
        """Inject trace context into headers"""
        self.propagator.inject(headers)
        return headers
    
    def extract_context(self, headers: Dict[str, str]):
        """Extract trace context from headers"""
        return self.propagator.extract(headers)
    
    def create_child_span(self, 
                         name: str,
                         parent_context: TraceContext = None) -> trace.Span:
        """Create a child span"""
        
        # Get or create context
        if parent_context:
            # Set parent context
            ctx = trace.set_span_in_context(
                trace.NonRecordingSpan(
                    trace.SpanContext(
                        trace_id=int(parent_context.trace_id, 16),
                        span_id=int(parent_context.span_id, 16),
                        is_remote=True,
                        trace_flags=trace.TraceFlags(0x01)
                    )
                )
            )
        else:
            ctx = None
        
        # Create span
        return self.tracer.start_span(name, context=ctx)


class TracingMiddleware:
    """HTTP middleware for distributed tracing"""
    
    def __init__(self, tracer: DistributedTracer):
        self.tracer = tracer
    
    async def __call__(self, request: Dict[str, Any], next_handler):
        """Middleware to add tracing to requests"""
        
        # Extract trace context from headers
        headers = request.get("headers", {})
        trace_context = TraceContext.from_headers(headers)
        
        # Start span
        async with self.tracer.trace_async_operation(
            f"{request.get('method', 'GET')} {request.get('path', '/')}",
            kind=SpanKind.SERVER,
            attributes={
                SpanAttributes.HTTP_METHOD: request.get("method", "GET"),
                SpanAttributes.HTTP_URL: request.get("path", "/"),
                SpanAttributes.HTTP_USER_AGENT: headers.get("User-Agent", ""),
                "client.address": request.get("client_address", "")
            }
        ) as span:
            if span:
                # Add trace context to request
                request["trace_span"] = span
                request["trace_context"] = trace_context
                
                # Process request
                response = await next_handler(request)
                
                # Add response attributes
                span.set_attributes({
                    SpanAttributes.HTTP_STATUS_CODE: response.get("status", 200)
                })
                
                if response.get("status", 200) >= 400:
                    span.set_status(Status(StatusCode.ERROR))
                else:
                    span.set_status(Status(StatusCode.OK))
                
                return response
            else:
                # No tracing (sampling)
                return await next_handler(request)


class TracedAgent:
    """Wrapper to add tracing to agent calls"""
    
    def __init__(self, agent_func, tracer: DistributedTracer):
        self.agent_func = agent_func
        self.tracer = tracer
    
    async def __call__(self, 
                      message: str,
                      agent_id: str = None,
                      context: Dict[str, Any] = None,
                      **kwargs):
        """Traced agent call"""
        
        # Extract trace context if provided
        trace_context = context.get("trace_context") if context else None
        
        async with self.tracer.trace_async_operation(
            "agent.call",
            kind=SpanKind.CLIENT,
            attributes={
                "agent.id": agent_id or "default",
                "message.length": len(message),
                "has_context": context is not None
            }
        ) as span:
            if span:
                # Add trace ID to context
                if context is None:
                    context = {}
                
                span_context = span.get_span_context()
                context["trace_id"] = format(span_context.trace_id, "032x")
                context["span_id"] = format(span_context.span_id, "016x")
                
                # Call agent
                start_time = time.time()
                try:
                    result = await self.agent_func(
                        message=message,
                        agent_id=agent_id,
                        context=context,
                        **kwargs
                    )
                    
                    # Record metrics
                    duration = time.time() - start_time
                    span.set_attributes({
                        "agent.response_time_ms": duration * 1000,
                        "agent.response_length": len(result) if result else 0
                    })
                    
                    return result
                    
                except Exception as e:
                    span.record_exception(e)
                    raise
            else:
                # No tracing
                return await self.agent_func(
                    message=message,
                    agent_id=agent_id,
                    context=context,
                    **kwargs
                )


class TracedCache:
    """Wrapper to add tracing to cache operations"""
    
    def __init__(self, cache, tracer: DistributedTracer):
        self.cache = cache
        self.tracer = tracer
    
    def get(self, key: str) -> Optional[Any]:
        """Traced cache get"""
        with self.tracer.trace_operation(
            "cache.get",
            attributes={
                "cache.key": key,
                "cache.operation": "get"
            }
        ) as span:
            value = self.cache.get(key)
            
            if span:
                span.set_attributes({
                    "cache.hit": value is not None
                })
            
            return value
    
    def set(self, key: str, value: Any):
        """Traced cache set"""
        with self.tracer.trace_operation(
            "cache.set",
            attributes={
                "cache.key": key,
                "cache.operation": "set",
                "cache.value_size": len(str(value))
            }
        ) as span:
            self.cache.set(key, value)


class TracedCircuitBreaker:
    """Wrapper to add tracing to circuit breaker"""
    
    def __init__(self, breaker, tracer: DistributedTracer):
        self.breaker = breaker
        self.tracer = tracer
    
    def can_execute(self) -> bool:
        """Traced execution check"""
        with self.tracer.trace_operation(
            "circuit_breaker.check",
            attributes={
                "circuit_breaker.state": self.breaker.get_state()
            }
        ) as span:
            can_execute = self.breaker.can_execute()
            
            if span:
                span.set_attributes({
                    "circuit_breaker.can_execute": can_execute
                })
            
            return can_execute
    
    def record_success(self):
        """Traced success recording"""
        with self.tracer.trace_operation(
            "circuit_breaker.record_success",
            attributes={
                "circuit_breaker.state": self.breaker.get_state()
            }
        ):
            self.breaker.record_success()
    
    def record_failure(self):
        """Traced failure recording"""
        with self.tracer.trace_operation(
            "circuit_breaker.record_failure",
            attributes={
                "circuit_breaker.state": self.breaker.get_state()
            }
        ):
            self.breaker.record_failure()


# Helper function to create traced components
def create_traced_components(tracer: DistributedTracer, components: Dict[str, Any]) -> Dict[str, Any]:
    """Wrap components with tracing"""
    
    traced = {}
    
    for name, component in components.items():
        if name == "agent":
            traced[name] = TracedAgent(component, tracer)
        elif name == "cache":
            traced[name] = TracedCache(component, tracer)
        elif name == "circuit_breaker":
            traced[name] = TracedCircuitBreaker(component, tracer)
        else:
            traced[name] = component
    
    return traced


# Custom span attributes for financial domain
class FinancialAttributes:
    """Custom attributes for financial operations"""
    
    TRANSACTION_ID = "finance.transaction_id"
    INSTRUMENT_TYPE = "finance.instrument_type"
    MARKET = "finance.market"
    RISK_LEVEL = "finance.risk_level"
    AMOUNT = "finance.amount"
    CURRENCY = "finance.currency"
    
    @staticmethod
    def for_analysis(analysis_type: str, market: str = None) -> Dict[str, Any]:
        """Attributes for financial analysis operations"""
        attrs = {
            "finance.analysis_type": analysis_type,
            "finance.timestamp": time.time()
        }
        if market:
            attrs[FinancialAttributes.MARKET] = market
        return attrs
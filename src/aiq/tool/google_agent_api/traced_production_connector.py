# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any, Dict, List, Optional
import asyncio
import time
import uuid

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.data_models.function import FunctionBaseConfig

from .production_connector import ProductionConnectorConfig, production_connector
from .tracing import (
    DistributedTracer, TracingMiddleware, TraceContext,
    FinancialAttributes
)

from opentelemetry import trace
from opentelemetry.trace import SpanKind, Status, StatusCode

log = logging.getLogger(__name__)


class TracedConnectorConfig(ProductionConnectorConfig, name="traced_production_connector"):
    """Multi-agent connector with distributed tracing"""
    
    # Tracing settings
    enable_tracing: bool = True
    otlp_endpoint: str = "localhost:4317"
    service_name: str = "agent-connector"
    sample_rate: float = 1.0
    console_export: bool = False


@register_function(config_type=TracedConnectorConfig)
async def traced_production_connector(config: TracedConnectorConfig, builder: Builder):
    """Production connector with distributed tracing"""
    
    # Initialize tracer
    tracer = None
    if config.enable_tracing:
        tracer = DistributedTracer(
            service_name=config.service_name,
            otlp_endpoint=config.otlp_endpoint,
            console_export=config.console_export,
            sample_rate=config.sample_rate
        )
    
    # Create base connector
    base_connector = await production_connector(config, builder)
    
    async def traced_route_message(
        message: str,
        target_capabilities: List[str] = None,
        broadcast: bool = False,
        context: Dict[str, Any] = None,
        trace_context: Optional[TraceContext] = None
    ) -> Dict[str, Any]:
        """Route message with distributed tracing"""
        
        # Create correlation ID
        correlation_id = str(uuid.uuid4())
        
        # Extract parent trace
        parent_context = trace_context or (context.get("trace_context") if context else None)
        
        # Start root span for routing operation
        span_attributes = {
            "connector.correlation_id": correlation_id,
            "connector.target_capabilities": ",".join(target_capabilities) if target_capabilities else "all",
            "connector.broadcast": broadcast,
            "connector.message_length": len(message)
        }
        
        # Add financial attributes if present
        if context and "financial_operation" in context:
            span_attributes.update(
                FinancialAttributes.for_analysis(
                    context["financial_operation"],
                    context.get("market")
                )
            )
        
        async with tracer.trace_async_operation(
            "connector.route_message",
            kind=SpanKind.SERVER,
            attributes=span_attributes
        ) as root_span:
            
            start_time = time.time()
            
            try:
                # Agent selection phase
                selected_agents = []
                
                async with tracer.trace_async_operation(
                    "connector.select_agents",
                    kind=SpanKind.INTERNAL
                ) as selection_span:
                    
                    # Get agents from base connector's capability index
                    if target_capabilities:
                        # Use capability-based selection
                        agent_set = set()
                        for capability in target_capabilities:
                            agents = config.connections  # Simplified for this example
                            matching = [a for a in agents if capability in a.get("capabilities", [])]
                            agent_set.update(a["agent_id"] for a in matching)
                        
                        selected_agents = [a for a in config.connections if a["agent_id"] in agent_set]
                    else:
                        selected_agents = config.connections
                    
                    if selection_span:
                        selection_span.set_attributes({
                            "agents.selected_count": len(selected_agents),
                            "agents.selected_ids": ",".join(a["agent_id"] for a in selected_agents)
                        })
                
                if not selected_agents:
                    return {
                        "error": "No agents found",
                        "correlation_id": correlation_id,
                        "trace_id": format(root_span.get_span_context().trace_id, "032x") if root_span else None
                    }
                
                # Parallel agent calls
                async def call_single_agent(agent_config: Dict[str, Any]):
                    """Call single agent with tracing"""
                    
                    agent_id = agent_config["agent_id"]
                    
                    # Create child span for this agent call
                    async with tracer.trace_async_operation(
                        f"connector.call_agent.{agent_id}",
                        kind=SpanKind.CLIENT,
                        attributes={
                            "agent.id": agent_id,
                            "agent.capabilities": ",".join(agent_config.get("capabilities", [])),
                            "agent.location": agent_config.get("location", "unknown")
                        }
                    ) as agent_span:
                        
                        agent_start_time = time.time()
                        
                        try:
                            # Create traced context for agent call
                            agent_context = context or {}
                            
                            if agent_span:
                                span_context = agent_span.get_span_context()
                                agent_context.update({
                                    "trace_id": format(span_context.trace_id, "032x"),
                                    "parent_span_id": format(span_context.span_id, "016x"),
                                    "correlation_id": correlation_id
                                })
                            
                            # Import and call the traced agent
                            from .traced_production_client import traced_production_agent, TracedProductionConfig
                            
                            agent_config_obj = TracedProductionConfig(
                                project_id=agent_config["project_id"],
                                location=agent_config.get("location", "us-central1"),
                                agent_id=agent_id,
                                enable_tracing=config.enable_tracing,
                                otlp_endpoint=config.otlp_endpoint
                            )
                            
                            agent_func = await traced_production_agent(agent_config_obj, builder)
                            
                            # Call agent
                            response = await agent_func.func(
                                message=message,
                                context=agent_context
                            )
                            
                            # Record success
                            if agent_span:
                                duration = time.time() - agent_start_time
                                agent_span.set_attributes({
                                    "agent.response_time_ms": duration * 1000,
                                    "agent.response_length": len(response),
                                    "agent.success": True
                                })
                                agent_span.set_status(Status(StatusCode.OK))
                            
                            return {
                                "agent_id": agent_id,
                                "response": response,
                                "status": "success",
                                "duration_ms": (time.time() - agent_start_time) * 1000
                            }
                        
                        except Exception as e:
                            # Record failure
                            if agent_span:
                                agent_span.record_exception(e)
                                agent_span.set_attributes({
                                    "agent.error": str(e),
                                    "agent.success": False
                                })
                                agent_span.set_status(Status(StatusCode.ERROR, str(e)))
                            
                            return {
                                "agent_id": agent_id,
                                "error": str(e),
                                "status": "error",
                                "duration_ms": (time.time() - agent_start_time) * 1000
                            }
                
                # Execute calls with tracing
                responses = []
                
                if broadcast or len(selected_agents) > 1:
                    # Parallel execution
                    async with tracer.trace_async_operation(
                        "connector.parallel_execution",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            "execution.type": "parallel",
                            "execution.agent_count": len(selected_agents)
                        }
                    ) as parallel_span:
                        
                        tasks = [call_single_agent(agent) for agent in selected_agents]
                        responses = await asyncio.gather(*tasks)
                        
                        if parallel_span:
                            parallel_span.set_attributes({
                                "execution.duration_ms": (time.time() - start_time) * 1000,
                                "execution.success_count": sum(1 for r in responses if r["status"] == "success")
                            })
                
                else:
                    # Single execution
                    async with tracer.trace_async_operation(
                        "connector.single_execution",
                        kind=SpanKind.INTERNAL,
                        attributes={
                            "execution.type": "single"
                        }
                    ) as single_span:
                        
                        response = await call_single_agent(selected_agents[0])
                        responses = [response]
                
                # Aggregate results if needed
                result = None
                
                if broadcast:
                    result = {
                        "broadcast": True,
                        "responses": responses,
                        "correlation_id": correlation_id,
                        "total_agents": len(selected_agents),
                        "successful": sum(1 for r in responses if r["status"] == "success"),
                        "duration_ms": (time.time() - start_time) * 1000
                    }
                
                elif len(responses) > 1:
                    # Aggregate responses
                    async with tracer.trace_async_operation(
                        "connector.aggregate_responses",
                        kind=SpanKind.INTERNAL
                    ) as agg_span:
                        
                        successful_responses = [
                            r["response"] for r in responses 
                            if r["status"] == "success"
                        ]
                        
                        if successful_responses:
                            result = {
                                "aggregated": True,
                                "responses": successful_responses,
                                "correlation_id": correlation_id,
                                "success_count": len(successful_responses),
                                "total_count": len(responses),
                                "duration_ms": (time.time() - start_time) * 1000
                            }
                        else:
                            result = {
                                "error": "All agents failed",
                                "correlation_id": correlation_id,
                                "duration_ms": (time.time() - start_time) * 1000
                            }
                
                else:
                    # Single response
                    result = responses[0]
                    result["correlation_id"] = correlation_id
                
                # Add trace ID to result
                if root_span:
                    result["trace_id"] = format(root_span.get_span_context().trace_id, "032x")
                    
                    # Set final span attributes
                    root_span.set_attributes({
                        "connector.result_type": result.get("broadcast", False) and "broadcast" or "targeted",
                        "connector.duration_ms": (time.time() - start_time) * 1000,
                        "connector.success": "error" not in result
                    })
                    
                    if "error" not in result:
                        root_span.set_status(Status(StatusCode.OK))
                    else:
                        root_span.set_status(Status(StatusCode.ERROR, result.get("error", "")))
                
                return result
            
            except Exception as e:
                # Record exception
                if root_span:
                    root_span.record_exception(e)
                    root_span.set_status(Status(StatusCode.ERROR, str(e)))
                
                raise
    
    # Add tracing middleware
    if tracer:
        middleware = TracingMiddleware(tracer)
        builder.add_middleware(middleware)
    
    return FunctionInfo(
        name="traced_production_connector",
        description="Multi-agent connector with distributed tracing",
        func=traced_route_message,
        schema={
            "type": "function",
            "function": {
                "name": "traced_production_connector",
                "description": "Route messages with full tracing",
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
                            "description": "Broadcast to all agents"
                        },
                        "context": {
                            "type": "object",
                            "description": "Additional context"
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
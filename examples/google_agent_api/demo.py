#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Google Agent API Demo

This script demonstrates how to use the Google Agent API integration
for agent-to-agent communication in AIQToolkit.

Usage:
    python demo.py

Environment Variables:
    GOOGLE_PROJECT_ID: Your Google Cloud project ID
    GOOGLE_AGENT_ID: The ID of your Google Agent
"""

import os
import asyncio
import logging
from typing import Dict, Any

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.tool.google_agent_api import (
    GoogleAgentAPIConfig,
    AgentToAgentConnectorConfig,
    AgentRegistryConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def simple_agent_call():
    """Demonstrate a simple call to a Google Agent"""
    
    print("\n=== Simple Agent Call Demo ===")
    
    # Get configuration from environment
    project_id = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
    agent_id = os.getenv("GOOGLE_AGENT_ID", "your-agent-id")
    
    # Configure the Google Agent client
    config = GoogleAgentAPIConfig(
        project_id=project_id,
        location="us-central1",
        agent_id=agent_id,
        timeout=30
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("google_agent", config)
    
    # Add an LLM for reasoning
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # Execute a query
    try:
        result = await workflow.run({
            "query": "What's the current weather in San Francisco?",
            "tools": ["google_agent"]
        })
        
        print(f"Agent Response: {result}")
        
    except Exception as e:
        logger.error(f"Error calling agent: {e}")


async def multi_agent_demo():
    """Demonstrate multi-agent orchestration"""
    
    print("\n=== Multi-Agent Orchestration Demo ===")
    
    project_id = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
    
    # Configure multiple agents
    connector_config = AgentToAgentConnectorConfig(
        connections=[
            {
                "agent_id": "weather-agent",
                "project_id": project_id,
                "location": "us-central1",
                "capabilities": ["weather", "forecast", "climate"],
                "metadata": {
                    "name": "Weather Information Agent",
                    "version": "1.0"
                }
            },
            {
                "agent_id": "news-agent",
                "project_id": project_id,
                "location": "us-central1",
                "capabilities": ["news", "headlines", "analysis"],
                "metadata": {
                    "name": "News Analysis Agent",
                    "version": "2.0"
                }
            },
            {
                "agent_id": "research-agent",
                "project_id": project_id,
                "location": "us-west1",
                "capabilities": ["research", "facts", "analysis"],
                "metadata": {
                    "name": "Research Assistant",
                    "version": "1.5"
                }
            }
        ],
        enable_caching=True,
        cache_ttl=300,
        max_concurrent_calls=5
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("agent_connector", connector_config)
    
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # Example 1: Query specific capability
    print("\n1. Querying weather agents...")
    try:
        weather_result = await workflow.run({
            "query": "Get weather forecast",
            "tools": ["agent_connector"],
            "tool_args": {
                "message": "What's the weather forecast for the next week in New York?",
                "target_capabilities": ["weather", "forecast"]
            }
        })
        print(f"Weather Result: {weather_result}")
    except Exception as e:
        logger.error(f"Weather query error: {e}")
    
    # Example 2: Broadcast to all agents
    print("\n2. Broadcasting to all agents...")
    try:
        broadcast_result = await workflow.run({
            "query": "Broadcast query",
            "tools": ["agent_connector"],
            "tool_args": {
                "message": "What are the most important events happening today?",
                "broadcast": True
            }
        })
        print(f"Broadcast Result: {broadcast_result}")
    except Exception as e:
        logger.error(f"Broadcast error: {e}")
    
    # Example 3: Aggregate multiple responses
    print("\n3. Aggregating research from multiple agents...")
    try:
        research_result = await workflow.run({
            "query": "Research and analyze",
            "tools": ["agent_connector"],
            "tool_args": {
                "message": "Analyze the impact of renewable energy on climate change",
                "target_capabilities": ["research", "analysis"],
                "aggregate_responses": True
            }
        })
        print(f"Research Result: {research_result}")
    except Exception as e:
        logger.error(f"Research query error: {e}")


async def agent_registry_demo():
    """Demonstrate agent registry functionality"""
    
    print("\n=== Agent Registry Demo ===")
    
    # Configure the registry
    registry_config = AgentRegistryConfig(
        registry_file="/tmp/demo_agent_registry.json",
        auto_discovery=True,
        enable_health_check=True
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("agent_registry", registry_config)
    
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # 1. Register agents
    print("\n1. Registering agents...")
    agents_to_register = [
        {
            "agent_id": "demo-weather-agent",
            "project_id": "demo-project",
            "location": "us-central1",
            "capabilities": ["weather", "forecast"],
            "metadata": {"type": "weather", "version": "1.0"}
        },
        {
            "agent_id": "demo-news-agent",
            "project_id": "demo-project",
            "location": "us-central1",
            "capabilities": ["news", "analysis"],
            "metadata": {"type": "news", "version": "2.0"}
        }
    ]
    
    for agent in agents_to_register:
        try:
            result = await workflow.run({
                "query": f"Register {agent['agent_id']}",
                "tools": ["agent_registry"],
                "tool_args": {
                    "action": "register",
                    **agent
                }
            })
            print(f"Registered: {result}")
        except Exception as e:
            logger.error(f"Registration error: {e}")
    
    # 2. Discover agents
    print("\n2. Discovering agents with weather capabilities...")
    try:
        discover_result = await workflow.run({
            "query": "Discover weather agents",
            "tools": ["agent_registry"],
            "tool_args": {
                "action": "discover",
                "capabilities": ["weather"]
            }
        })
        print(f"Discovered agents: {discover_result}")
    except Exception as e:
        logger.error(f"Discovery error: {e}")
    
    # 3. Get agent details
    print("\n3. Getting details for specific agent...")
    try:
        details_result = await workflow.run({
            "query": "Get agent details",
            "tools": ["agent_registry"],
            "tool_args": {
                "action": "get_details",
                "agent_id": "demo-weather-agent"
            }
        })
        print(f"Agent details: {details_result}")
    except Exception as e:
        logger.error(f"Details error: {e}")


async def main():
    """Run all demos"""
    
    print("Google Agent API Integration Demo")
    print("=================================")
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_PROJECT_ID"):
        print("\nWarning: GOOGLE_PROJECT_ID not set. Using demo values.")
        print("Set GOOGLE_PROJECT_ID and GOOGLE_AGENT_ID for real API calls.")
    
    try:
        # Run demos
        await simple_agent_call()
        await multi_agent_demo()
        await agent_registry_demo()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    print("\n=== Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
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

"""Example usage of Google Agent API integration"""

import asyncio
from aiq.builder.workflow_builder import WorkflowBuilder
from .agent_client import GoogleAgentAPIConfig
from .agent_connector import AgentToAgentConnectorConfig
from .agent_registry import AgentRegistryConfig


async def example_single_agent_call():
    """Example of calling a single Google Agent"""
    
    # Configure the client
    config = GoogleAgentAPIConfig(
        project_id="your-project-id",
        location="us-central1",
        agent_id="your-agent-id"
    )
    
    # Build a workflow with the Google Agent API
    builder = WorkflowBuilder()
    builder.add_function("google_agent", config)
    
    workflow = builder.build()
    
    # Call the agent
    result = await workflow.run({
        "query": "What's the weather in San Francisco?",
        "tools": ["google_agent"]
    })
    
    print(f"Agent response: {result}")


async def example_multi_agent_orchestration():
    """Example of orchestrating multiple agents"""
    
    # Configure the connector with multiple agents
    connector_config = AgentToAgentConnectorConfig(
        connections=[
            {
                "agent_id": "weather-agent",
                "project_id": "your-project-id",
                "location": "us-central1",
                "capabilities": ["weather", "forecast"],
                "metadata": {"version": "1.0"}
            },
            {
                "agent_id": "news-agent",
                "project_id": "your-project-id",
                "location": "us-central1",
                "capabilities": ["news", "headlines"],
                "metadata": {"version": "2.0"}
            },
            {
                "agent_id": "research-agent",
                "project_id": "your-project-id",
                "location": "us-west1",
                "capabilities": ["research", "analysis"],
                "metadata": {"version": "1.5"}
            }
        ],
        enable_caching=True,
        max_concurrent_calls=5
    )
    
    # Build workflow
    builder = WorkflowBuilder()
    builder.add_function("agent_connector", connector_config)
    
    workflow = builder.build()
    
    # Example 1: Route to agents with specific capabilities
    weather_result = await workflow.run({
        "query": "Get weather information",
        "tools": ["agent_connector"],
        "tool_args": {
            "target_capabilities": ["weather"],
            "message": "What's the weather forecast for next week?"
        }
    })
    
    # Example 2: Broadcast to all agents
    broadcast_result = await workflow.run({
        "query": "Broadcast question",
        "tools": ["agent_connector"],
        "tool_args": {
            "broadcast": True,
            "message": "What are the top trending topics today?"
        }
    })
    
    # Example 3: Aggregate responses from multiple agents
    research_result = await workflow.run({
        "query": "Research query",
        "tools": ["agent_connector"],
        "tool_args": {
            "target_capabilities": ["research", "news"],
            "aggregate_responses": True,
            "message": "Analyze the impact of AI on healthcare"
        }
    })
    
    print(f"Weather result: {weather_result}")
    print(f"Broadcast result: {broadcast_result}")
    print(f"Research result: {research_result}")


async def example_agent_registry():
    """Example of using the agent registry"""
    
    # Configure the registry
    registry_config = AgentRegistryConfig(
        registry_file="my_agents.json",
        auto_discovery=True,
        enable_health_check=True
    )
    
    # Build workflow
    builder = WorkflowBuilder()
    builder.add_function("agent_registry", registry_config)
    
    workflow = builder.build()
    
    # Register a new agent
    register_result = await workflow.run({
        "query": "Register agent",
        "tools": ["agent_registry"],
        "tool_args": {
            "action": "register",
            "agent_id": "my-custom-agent",
            "project_id": "my-project",
            "location": "us-central1",
            "capabilities": ["custom", "analysis"],
            "metadata": {"version": "1.0", "author": "user@example.com"}
        }
    })
    
    # Discover agents with specific capabilities
    discover_result = await workflow.run({
        "query": "Discover agents",
        "tools": ["agent_registry"],
        "tool_args": {
            "action": "discover",
            "capabilities": ["analysis"]
        }
    })
    
    # Get details for a specific agent
    details_result = await workflow.run({
        "query": "Get agent details",
        "tools": ["agent_registry"],
        "tool_args": {
            "action": "get_details",
            "agent_id": "my-custom-agent"
        }
    })
    
    print(f"Register result: {register_result}")
    print(f"Discover result: {discover_result}")
    print(f"Details result: {details_result}")


if __name__ == "__main__":
    # Run examples
    asyncio.run(example_single_agent_call())
    asyncio.run(example_multi_agent_orchestration())
    asyncio.run(example_agent_registry())
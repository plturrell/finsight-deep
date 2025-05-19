#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Finsight Deep Demo

This script demonstrates how to use the Finsight Deep agent
for financial analysis and insights through Google Agent API.

Usage:
    python finsight_deep_demo.py

Environment Variables:
    GOOGLE_PROJECT_ID: Your Google Cloud project ID
"""

import os
import asyncio
import logging
from typing import Dict, Any, List

from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.tool.google_agent_api import (
    GoogleAgentAPIConfig,
    AgentToAgentConnectorConfig,
    AgentRegistryConfig
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def finsight_deep_direct_call():
    """Demonstrate direct calls to Finsight Deep agent"""
    
    print("\n=== Finsight Deep Direct Call Demo ===")
    
    # Get configuration from environment
    project_id = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
    
    # Configure Finsight Deep client
    config = GoogleAgentAPIConfig(
        project_id=project_id,
        location="us-central1",
        agent_id="finsight_deep",
        timeout=60
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("finsight_deep", config)
    
    # Add LLM for reasoning
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # Example queries
    queries = [
        "Analyze the current state of the tech sector and identify top growth opportunities",
        "What are the key risk factors for investing in emerging markets?",
        "Provide a comprehensive analysis of NVIDIA's financial position",
        "Compare the performance of major cryptocurrency ETFs vs traditional tech ETFs"
    ]
    
    for query in queries:
        try:
            print(f"\nQuery: {query}")
            result = await workflow.run({
                "query": query,
                "tools": ["finsight_deep"]
            })
            
            print(f"Finsight Deep Response: {result}")
            
        except Exception as e:
            logger.error(f"Error calling Finsight Deep: {e}")


async def financial_multi_agent_analysis():
    """Demonstrate multi-agent financial analysis"""
    
    print("\n=== Multi-Agent Financial Analysis Demo ===")
    
    project_id = os.getenv("GOOGLE_PROJECT_ID", "your-project-id")
    
    # Configure financial agent network
    connector_config = AgentToAgentConnectorConfig(
        connections=[
            {
                "agent_id": "finsight_deep",
                "project_id": project_id,
                "location": "us-central1",
                "capabilities": ["finance", "analysis", "insights", "market_data", "risk_assessment"],
                "metadata": {
                    "name": "Finsight Deep",
                    "version": "3.0",
                    "specialization": "Comprehensive financial analysis"
                }
            },
            {
                "agent_id": "market-data-agent",
                "project_id": project_id,
                "location": "us-central1",
                "capabilities": ["market_data", "real_time_quotes", "trading", "technical_analysis"],
                "metadata": {
                    "name": "Market Data Specialist",
                    "version": "2.0",
                    "specialization": "Real-time market data and technical indicators"
                }
            },
            {
                "agent_id": "risk-analysis-agent",
                "project_id": project_id,
                "location": "us-central1",
                "capabilities": ["risk_assessment", "portfolio_analysis", "compliance", "stress_testing"],
                "metadata": {
                    "name": "Risk Analysis Expert",
                    "version": "1.5",
                    "specialization": "Portfolio risk and compliance analysis"
                }
            },
            {
                "agent_id": "news-sentiment-agent",
                "project_id": project_id,
                "location": "us-west1",
                "capabilities": ["news", "sentiment_analysis", "market_impact", "earnings_analysis"],
                "metadata": {
                    "name": "News Sentiment Analyzer",
                    "version": "2.0",
                    "specialization": "Financial news and market sentiment"
                }
            }
        ],
        enable_caching=True,
        cache_ttl=300,
        max_concurrent_calls=5
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("financial_agents", connector_config)
    
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # Example 1: Comprehensive market analysis
    print("\n1. Comprehensive Market Analysis...")
    try:
        market_analysis = await workflow.run({
            "query": "Perform comprehensive market analysis",
            "tools": ["financial_agents"],
            "tool_args": {
                "message": "Provide a comprehensive analysis of current market conditions, including tech sector performance, key risks, and investment opportunities",
                "target_capabilities": ["finance", "analysis", "market_data"],
                "aggregate_responses": True
            }
        })
        print(f"Market Analysis Result: {market_analysis}")
    except Exception as e:
        logger.error(f"Market analysis error: {e}")
    
    # Example 2: Risk assessment for portfolio
    print("\n2. Portfolio Risk Assessment...")
    try:
        risk_assessment = await workflow.run({
            "query": "Assess portfolio risk",
            "tools": ["financial_agents"],
            "tool_args": {
                "message": "Analyze risk factors for a portfolio heavily weighted in tech stocks and cryptocurrencies",
                "target_capabilities": ["risk_assessment", "portfolio_analysis"],
                "aggregate_responses": True
            }
        })
        print(f"Risk Assessment Result: {risk_assessment}")
    except Exception as e:
        logger.error(f"Risk assessment error: {e}")
    
    # Example 3: Market sentiment analysis
    print("\n3. Market Sentiment Analysis...")
    try:
        sentiment_analysis = await workflow.run({
            "query": "Analyze market sentiment",
            "tools": ["financial_agents"],
            "tool_args": {
                "message": "What is the current market sentiment around AI stocks and how is it affecting valuations?",
                "target_capabilities": ["sentiment_analysis", "market_impact"],
                "aggregate_responses": True
            }
        })
        print(f"Sentiment Analysis Result: {sentiment_analysis}")
    except Exception as e:
        logger.error(f"Sentiment analysis error: {e}")
    
    # Example 4: Broadcast financial query to all agents
    print("\n4. Broadcast Financial Query...")
    try:
        broadcast_result = await workflow.run({
            "query": "Broadcast financial analysis",
            "tools": ["financial_agents"],
            "tool_args": {
                "message": "What are the most significant financial events or trends to watch this week?",
                "broadcast": True
            }
        })
        print(f"Broadcast Result: {broadcast_result}")
    except Exception as e:
        logger.error(f"Broadcast error: {e}")


async def financial_agent_registry():
    """Demonstrate financial agent registry and discovery"""
    
    print("\n=== Financial Agent Registry Demo ===")
    
    # Configure the registry
    registry_config = AgentRegistryConfig(
        registry_file="/tmp/financial_agent_registry.json",
        auto_discovery=True,
        enable_health_check=True
    )
    
    # Build the workflow
    builder = WorkflowBuilder()
    builder.add_function("financial_registry", registry_config)
    
    builder.add_llm({
        "llm_type": "nim_llm",
        "model": "llama-3.1-70b-instruct",
        "api_base": "https://integrate.api.nvidia.com/v1",
        "api_key_env": "NVIDIA_API_KEY"
    })
    
    workflow = builder.build()
    
    # Register financial agents
    print("\n1. Registering Financial Agents...")
    financial_agents = [
        {
            "agent_id": "finsight_deep",
            "project_id": os.getenv("GOOGLE_PROJECT_ID", "your-project-id"),
            "location": "us-central1",
            "capabilities": ["finance", "analysis", "insights", "market_data", "risk_assessment"],
            "metadata": {
                "type": "financial_analysis",
                "version": "3.0",
                "features": ["deep_learning", "real_time_analysis", "predictive_modeling"]
            }
        },
        {
            "agent_id": "crypto-analysis-agent",
            "project_id": os.getenv("GOOGLE_PROJECT_ID", "your-project-id"),
            "location": "us-central1",
            "capabilities": ["cryptocurrency", "blockchain", "defi", "nft_analysis"],
            "metadata": {
                "type": "crypto_specialist",
                "version": "1.0",
                "features": ["on_chain_analysis", "defi_protocols", "nft_valuation"]
            }
        },
        {
            "agent_id": "esg-analysis-agent",
            "project_id": os.getenv("GOOGLE_PROJECT_ID", "your-project-id"),
            "location": "us-west1",
            "capabilities": ["esg", "sustainability", "impact_investing", "carbon_analysis"],
            "metadata": {
                "type": "esg_specialist",
                "version": "1.5",
                "features": ["esg_scoring", "carbon_footprint", "social_impact"]
            }
        }
    ]
    
    for agent in financial_agents:
        try:
            result = await workflow.run({
                "query": f"Register {agent['agent_id']}",
                "tools": ["financial_registry"],
                "tool_args": {
                    "action": "register",
                    **agent
                }
            })
            print(f"Registered: {agent['agent_id']} - {result}")
        except Exception as e:
            logger.error(f"Registration error for {agent['agent_id']}: {e}")
    
    # Discover agents by capability
    print("\n2. Discovering Agents by Capability...")
    capabilities_to_search = ["finance", "cryptocurrency", "esg", "risk_assessment"]
    
    for capability in capabilities_to_search:
        try:
            discover_result = await workflow.run({
                "query": f"Discover {capability} agents",
                "tools": ["financial_registry"],
                "tool_args": {
                    "action": "discover",
                    "capabilities": [capability]
                }
            })
            print(f"\nAgents with {capability} capability: {discover_result}")
        except Exception as e:
            logger.error(f"Discovery error for {capability}: {e}")
    
    # Get detailed information about Finsight Deep
    print("\n3. Getting Finsight Deep Details...")
    try:
        details_result = await workflow.run({
            "query": "Get Finsight Deep details",
            "tools": ["financial_registry"],
            "tool_args": {
                "action": "get_details",
                "agent_id": "finsight_deep"
            }
        })
        print(f"Finsight Deep Details: {details_result}")
    except Exception as e:
        logger.error(f"Details error: {e}")


async def main():
    """Run all Finsight Deep demos"""
    
    print("Finsight Deep Financial Analysis Demo")
    print("====================================")
    
    # Check for required environment variables
    if not os.getenv("GOOGLE_PROJECT_ID"):
        print("\nWarning: GOOGLE_PROJECT_ID not set.")
        print("Please set GOOGLE_PROJECT_ID for real API calls.")
        print("Example: export GOOGLE_PROJECT_ID='your-project-id'")
    
    try:
        # Run demos
        await finsight_deep_direct_call()
        await financial_multi_agent_analysis()
        await financial_agent_registry()
        
    except Exception as e:
        logger.error(f"Demo error: {e}")
    
    print("\n=== Finsight Deep Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
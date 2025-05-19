# Google Agent API Integration - Finsight Deep

This example demonstrates how to use the Google Agent API integration in AIQToolkit for agent-to-agent communication, with a focus on the **Finsight Deep** financial analysis agent.

## Overview

The Google Agent API integration provides:

1. **Google Agent Client**: Direct communication with Google AI agents, including Finsight Deep
2. **Agent-to-Agent Connector**: Orchestrate multiple financial agents with capability-based routing
3. **Agent Registry**: Discover and manage available financial agents

### Finsight Deep

Finsight Deep is a specialized financial analysis agent that provides:
- Comprehensive market analysis
- Risk assessment and portfolio evaluation
- Real-time financial insights
- Predictive modeling for investment opportunities
- Integration with other financial data sources

## Setup

### Prerequisites

1. Google Cloud Project with necessary permissions
2. Google authentication configured:
   ```bash
   gcloud auth application-default login
   ```
3. Environment variables:
   ```bash
   export GOOGLE_PROJECT_ID="your-project-id"
   export GOOGLE_AGENT_ID="your-agent-id"
   ```

### Installation

Install the required dependencies:

```bash
pip install google-auth google-auth-httplib2 aiohttp
```

## Configuration

The example configuration file (`config.yml`) includes three main components:

### 1. Google Agent API Client

```yaml
functions:
  - function_type: "google_agent_api"
    project_id: "${GOOGLE_PROJECT_ID}"
    location: "us-central1"
    agent_id: "${GOOGLE_AGENT_ID}"
    timeout: 60
    max_retries: 3
```

### 2. Agent-to-Agent Connector

```yaml
  - function_type: "agent_to_agent_connector"
    connections:
      - agent_id: "weather-agent"
        capabilities: ["weather", "forecast"]
      - agent_id: "news-agent"
        capabilities: ["news", "headlines"]
    enable_caching: true
    max_concurrent_calls: 10
```

### 3. Agent Registry

```yaml
  - function_type: "agent_registry"
    registry_file: "/tmp/agent_registry.json"
    auto_discovery: true
    enable_health_check: true
```

## Usage

### Running the Finsight Deep Example

```bash
aiq run --config_file examples/google_agent_api/config.yml
```

### Running the Financial Demo

```bash
python examples/google_agent_api/finsight_deep_demo.py
```

### Example Queries for Finsight Deep

1. **Market Analysis**:
   ```
   "Analyze the current state of the tech sector and identify growth opportunities"
   ```

2. **Risk Assessment**:
   ```
   "Evaluate the risk factors for a portfolio heavily weighted in AI stocks"
   ```

3. **Investment Insights**:
   ```
   "Compare NVIDIA's financial position with other semiconductor companies"
   ```

4. **Multi-Agent Financial Analysis**:
   ```
   "Provide a comprehensive analysis combining market data, risk assessment, and sentiment analysis for the renewable energy sector"
   ```

5. **Cryptocurrency Analysis**:
   ```
   "Analyze the correlation between Bitcoin ETFs and traditional tech stock performance"
   ```

## Python Usage

```python
import asyncio
from aiq.builder.workflow_builder import WorkflowBuilder
from aiq.tool.google_agent_api import GoogleAgentAPIConfig

async def main():
    # Configure Finsight Deep
    config = GoogleAgentAPIConfig(
        project_id="your-project-id",
        location="us-central1",
        agent_id="finsight_deep"
    )
    
    # Build workflow
    builder = WorkflowBuilder()
    builder.add_function("finsight_deep", config)
    workflow = builder.build()
    
    # Call Finsight Deep for financial analysis
    result = await workflow.run({
        "query": "Analyze NVIDIA's financial position and growth prospects",
        "tools": ["finsight_deep"]
    })
    
    print(result)

asyncio.run(main())
```

## Features

### Finsight Deep Capabilities

Finsight Deep offers specialized financial analysis capabilities:

```python
# Direct financial analysis
result = await workflow.run({
    "tools": ["finsight_deep"],
    "tool_args": {
        "message": "Provide a comprehensive analysis of the semiconductor industry",
        "context": {
            "focus_areas": ["market_trends", "competitive_landscape", "growth_projections"],
            "time_horizon": "5_years"
        }
    }
})
```

### Multi-Agent Financial Analysis

Combine Finsight Deep with other financial agents for comprehensive insights:

```python
# Route to multiple financial agents
result = await workflow.run({
    "tools": ["agent_connector"],
    "tool_args": {
        "target_capabilities": ["finance", "risk_assessment", "market_data"],
        "message": "Analyze the investment potential of AI companies",
        "aggregate_responses": True
    }
})
```

### Broadcasting

Send messages to multiple agents simultaneously:

```python
result = await workflow.run({
    "tools": ["agent_connector"],
    "tool_args": {
        "broadcast": True,
        "message": "What are today's top stories?"
    }
})
```

### Response Aggregation

Combine responses from multiple agents:

```python
result = await workflow.run({
    "tools": ["agent_connector"],
    "tool_args": {
        "target_capabilities": ["research", "analysis"],
        "aggregate_responses": True,
        "message": "Analyze market trends"
    }
})
```

## Security Considerations

1. Use Google Cloud IAM to control access to agents
2. Set appropriate timeouts and rate limits
3. Enable caching judiciously to avoid stale data
4. Use the registry to track and audit agent usage

## Troubleshooting

1. **Authentication errors**: Ensure Google credentials are properly configured
2. **Timeout errors**: Adjust timeout settings based on agent response times
3. **Rate limiting**: Use the semaphore settings to control concurrent calls

## Advanced Usage

See `examples.py` for more advanced usage patterns including:
- Custom routing logic
- Health monitoring
- Dynamic agent registration
- Error handling and retries
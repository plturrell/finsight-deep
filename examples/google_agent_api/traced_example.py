"""
Example: Using the Traced Google Agent API with Distributed Tracing
"""
import asyncio
import logging
from typing import Dict, Any

from aiq.tool.google_agent_api import (
    TracedGoogleAgentClient,
    TracedAgentToAgentConnector,
    AgentRegistry
)
from aiq.tool.google_agent_api.auth import AuthenticationManager, Permission
from aiq.tool.google_agent_api.secrets import SecretManager
from aiq.tool.google_agent_api.tracing import DistributedTracer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def setup_infrastructure():
    """Set up authentication, secrets, and tracing infrastructure."""
    # Initialize authentication
    auth_manager = AuthenticationManager(
        secret_key="your-secret-key",
        issuer="aiq-agent-system",
        audience="agent-api"
    )
    
    # Initialize secrets
    secret_manager = SecretManager()
    await secret_manager.initialize()
    
    # Initialize distributed tracing
    tracer = DistributedTracer(
        service_name="google-agent-api-example",
        tracing_endpoint="http://jaeger:14268/api/traces",  # Jaeger endpoint
        metrics_endpoint="http://prometheus:9090/api/v1/write"  # Prometheus endpoint
    )
    
    return auth_manager, secret_manager, tracer


async def setup_agents(auth_manager, secret_manager, tracer):
    """Set up agent registry and clients."""
    registry = AgentRegistry()
    
    # Create clients for different Google agents
    sentiment_client = TracedGoogleAgentClient(
        project_id="your-project-id",
        location="us-central1",
        agent_id="sentiment-analyzer-agent",
        auth_manager=auth_manager,
        secret_manager=secret_manager,
        tracer=tracer
    )
    
    summarization_client = TracedGoogleAgentClient(
        project_id="your-project-id",
        location="us-central1",
        agent_id="text-summarizer-agent",
        auth_manager=auth_manager,
        secret_manager=secret_manager,
        tracer=tracer
    )
    
    translation_client = TracedGoogleAgentClient(
        project_id="your-project-id",
        location="us-central1",
        agent_id="translator-agent",
        auth_manager=auth_manager,
        secret_manager=secret_manager,
        tracer=tracer
    )
    
    # Register agents with their capabilities
    registry.register_agent("sentiment", sentiment_client)
    registry.add_capability("sentiment", "sentiment_analysis", ["analyze", "emotion", "tone"])
    
    registry.register_agent("summarizer", summarization_client)
    registry.add_capability("summarizer", "text_summarization", ["summarize", "brief", "abstract"])
    
    registry.register_agent("translator", translation_client)
    registry.add_capability("translator", "translation", ["translate", "convert", "language"])
    
    return registry


async def demonstrate_single_agent_tracing(client: TracedGoogleAgentClient):
    """Demonstrate single agent communication with tracing."""
    logger.info("Starting single agent demonstration...")
    
    # Create a traced request
    response = await client.send_message(
        session_id="demo-session-single",
        message="Analyze the sentiment of: I love using distributed systems!",
        language_code="en"
    )
    
    logger.info(f"Single agent response: {response}")
    return response


async def demonstrate_multi_agent_tracing(connector: TracedAgentToAgentConnector):
    """Demonstrate multi-agent orchestration with tracing."""
    logger.info("Starting multi-agent demonstration...")
    
    # Complex request that requires multiple agents
    message = """
    Translate this text to Spanish and then analyze the sentiment:
    'The new distributed tracing system is incredibly powerful and easy to use!'
    """
    
    response = await connector.route_message(
        message=message,
        session_id="demo-session-multi",
        metadata={
            "user_preference": "detailed_analysis",
            "target_language": "es"
        }
    )
    
    logger.info(f"Multi-agent response: {response}")
    return response


async def demonstrate_parallel_processing(connector: TracedAgentToAgentConnector):
    """Demonstrate parallel agent processing with tracing."""
    logger.info("Starting parallel processing demonstration...")
    
    # Request that can be processed by multiple agents in parallel
    message = """
    Analyze this text in multiple ways:
    'Artificial intelligence is transforming how we interact with technology.'
    
    1. Analyze the sentiment
    2. Summarize the main points
    3. Translate to French
    """
    
    response = await connector.route_message(
        message=message,
        session_id="demo-session-parallel",
        metadata={
            "processing_mode": "parallel",
            "include_confidence": True
        }
    )
    
    logger.info(f"Parallel processing response: {response}")
    return response


async def demonstrate_error_handling(connector: TracedAgentToAgentConnector):
    """Demonstrate error handling with tracing."""
    logger.info("Starting error handling demonstration...")
    
    try:
        # Intentionally malformed request
        response = await connector.route_message(
            message="<script>alert('xss')</script>",  # Will be caught by validation
            session_id="demo-session-error"
        )
    except Exception as e:
        logger.error(f"Caught expected error: {e}")
        # Error will be traced with proper span status
    
    # Request that might trigger circuit breaker
    for i in range(5):
        try:
            response = await connector.route_message(
                message=f"Test request {i}",
                session_id=f"demo-session-circuit-{i}",
                metadata={"force_error": True}  # Simulate errors
            )
        except Exception as e:
            logger.warning(f"Request {i} failed: {e}")


async def view_trace_data(tracer: DistributedTracer):
    """View collected trace data and metrics."""
    logger.info("Viewing trace data...")
    
    # Export trace data
    tracer.export_metrics()
    
    # Get trace statistics
    stats = await tracer.get_trace_statistics()
    logger.info(f"Trace statistics: {stats}")
    
    # Get slow operations
    slow_ops = await tracer.get_slow_operations(threshold_ms=500)
    logger.info(f"Slow operations: {slow_ops}")


async def main():
    """Run the complete traced example."""
    # Set up infrastructure
    auth_manager, secret_manager, tracer = await setup_infrastructure()
    
    # Create authentication token
    token = await auth_manager.create_token(
        user_id="demo-user",
        permissions={Permission.READ, Permission.WRITE, Permission.EXECUTE},
        roles={"developer", "admin"}
    )
    
    # Set up agents
    registry = await setup_agents(auth_manager, secret_manager, tracer)
    
    # Create traced connector
    connector = TracedAgentToAgentConnector(
        registry=registry,
        auth_manager=auth_manager,
        tracer=tracer
    )
    
    # Run demonstrations
    await demonstrate_single_agent_tracing(
        registry.get_agent("sentiment")
    )
    
    await demonstrate_multi_agent_tracing(connector)
    
    await demonstrate_parallel_processing(connector)
    
    await demonstrate_error_handling(connector)
    
    # View trace data
    await view_trace_data(tracer)
    
    # Cleanup
    await tracer.shutdown()
    logger.info("Example completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
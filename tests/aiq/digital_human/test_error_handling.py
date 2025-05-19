"""Error handling tests for Digital Human system"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from unittest.mock import AsyncMock, Mock, MagicMock, patch
import uuid

import numpy as np
import pandas as pd
import pytest
import torch

from aiq.digital_human.orchestrator import (
    DigitalHumanOrchestrator,
    OrchestratorConfig,
    ConversationContext,
    OrchestratorError,
    ComponentError,
    ValidationError,
    TimeoutError,
    ResourceError
)
from aiq.digital_human.conversation_engine import (
    ConversationEngine,
    ConversationError,
    NLPError,
    IntentClassificationError
)
from aiq.digital_human.financial.mcts_financial_analyzer import (
    MCTSFinancialAnalyzer,
    AnalysisError,
    DataError
)
from aiq.digital_human.financial.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationError,
    ConstraintError
)
from aiq.digital_human.financial.financial_data_processor import (
    FinancialDataProcessor,
    ProcessingError,
    MarketDataError
)
from aiq.digital_human.financial.risk_assessment_engine import (
    RiskAssessmentEngine,
    RiskCalculationError,
    StressTestError
)
from aiq.digital_human.avatar import (
    AvatarSystem,
    AnimationError,
    RenderingError
)


# Test fixtures
@pytest.fixture
def config():
    """Create orchestrator configuration for error testing"""
    return OrchestratorConfig(
        enable_gpu=False,
        max_concurrent_sessions=10,
        session_timeout_minutes=30,
        health_check_interval_seconds=60,
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_logging=True,
        log_level="ERROR",
        enable_metrics=True,
        enable_tracing=True,
        error_retry_attempts=3,
        error_retry_delay=0.1,
        graceful_degradation=True
    )


@pytest.fixture
def conversation_context():
    """Create sample conversation context"""
    return ConversationContext(
        session_id="error-test-123",
        user_id="error-user-456",
        conversation_history=[],
        user_profile={
            "name": "Error Test User",
            "risk_tolerance": "moderate"
        }
    )


@pytest.fixture
async def orchestrator(config):
    """Create Digital Human Orchestrator with mocked components"""
    orchestrator = DigitalHumanOrchestrator(config)
    
    # Create mocked components
    orchestrator.conversation_engine = AsyncMock(spec=ConversationEngine)
    orchestrator.financial_analyzer = AsyncMock(spec=MCTSFinancialAnalyzer)
    orchestrator.portfolio_optimizer = AsyncMock(spec=PortfolioOptimizer)
    orchestrator.data_processor = AsyncMock(spec=FinancialDataProcessor)
    orchestrator.risk_engine = AsyncMock(spec=RiskAssessmentEngine)
    orchestrator.avatar_system = AsyncMock(spec=AvatarSystem)
    
    # Initialize
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.shutdown()


# Test classes
class TestValidationErrors:
    """Test input validation and error handling"""
    
    @pytest.mark.asyncio
    async def test_invalid_input_validation(self, orchestrator, conversation_context):
        """Test handling of invalid user inputs"""
        invalid_inputs = [
            "",  # Empty input
            " " * 1000,  # Whitespace only
            "a" * 10000,  # Too long
            "\x00\x01\x02",  # Binary data
            "🔥" * 500,  # Excessive emojis
            "<script>alert('xss')</script>",  # Potential XSS
            "'; DROP TABLE users; --",  # SQL injection attempt
        ]
        
        for invalid_input in invalid_inputs:
            response = await orchestrator.process_user_input(
                invalid_input,
                conversation_context
            )
            
            assert response.error is not None
            assert isinstance(response.error, ValidationError)
            assert response.fallback_response is not None
            assert "invalid input" in response.error.message.lower()
    
    @pytest.mark.asyncio
    async def test_invalid_context_validation(self, orchestrator):
        """Test handling of invalid conversation contexts"""
        user_input = "Valid user input"
        
        invalid_contexts = [
            None,  # Null context
            ConversationContext(
                session_id="",  # Empty session ID
                user_id="user-123",
                conversation_history=[]
            ),
            ConversationContext(
                session_id="session-123",
                user_id="",  # Empty user ID
                conversation_history=[]
            ),
            ConversationContext(
                session_id="session-123",
                user_id="user-123",
                conversation_history=[{"invalid": "structure"}]  # Invalid history
            ),
            ConversationContext(
                session_id="session-123",
                user_id="user-123",
                conversation_history=[],
                user_profile={"portfolio_value": "not_a_number"}  # Invalid profile data
            ),
        ]
        
        for invalid_context in invalid_contexts:
            response = await orchestrator.process_user_input(
                user_input,
                invalid_context
            )
            
            assert response.error is not None
            assert isinstance(response.error, ValidationError)
            assert response.fallback_response is not None
    
    @pytest.mark.asyncio
    async def test_malformed_request_handling(self, orchestrator, conversation_context):
        """Test handling of malformed requests"""
        # Mock conversation engine to return malformed response
        orchestrator.conversation_engine.process_input.return_value = {
            "intent": None,  # Missing required field
            "confidence": "not_a_number"
        }
        
        response = await orchestrator.process_user_input(
            "Test input",
            conversation_context
        )
        
        assert response.error is not None
        assert response.fallback_response is not None
        assert "processing error" in response.error.message.lower()


class TestComponentErrors:
    """Test component-specific error handling"""
    
    @pytest.mark.asyncio
    async def test_conversation_engine_error(self, orchestrator, conversation_context):
        """Test handling of conversation engine errors"""
        error_scenarios = [
            NLPError("NLP model unavailable"),
            IntentClassificationError("Unable to classify intent"),
            ConversationError("Context overflow"),
            Exception("Unexpected conversation error")
        ]
        
        for error in error_scenarios:
            orchestrator.conversation_engine.process_input.side_effect = error
            
            response = await orchestrator.process_user_input(
                "What's my portfolio value?",
                conversation_context
            )
            
            assert response.error is not None
            assert response.fallback_response is not None
            assert response.suggested_actions == ["retry", "contact_support"]
            
            # Reset for next iteration
            orchestrator.conversation_engine.process_input.side_effect = None
    
    @pytest.mark.asyncio
    async def test_financial_analyzer_error(self, orchestrator, conversation_context):
        """Test handling of financial analyzer errors"""
        # Mock successful conversation response
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="portfolio_analysis",
            entities={},
            response_text="Analyzing portfolio..."
        )
        
        error_scenarios = [
            AnalysisError("Insufficient market data"),
            DataError("Corrupted portfolio data"),
            TimeoutError("Analysis timeout"),
            Exception("GPU computation failed")
        ]
        
        for error in error_scenarios:
            orchestrator.financial_analyzer.analyze_portfolio.side_effect = error
            
            response = await orchestrator.process_user_input(
                "Analyze my portfolio",
                conversation_context
            )
            
            assert response.error is not None or response.partial_failure
            assert response.text is not None  # Should still have conversation response
            assert response.warnings is not None
            assert len(response.warnings) > 0
    
    @pytest.mark.asyncio
    async def test_portfolio_optimizer_error(self, orchestrator, conversation_context):
        """Test handling of portfolio optimizer errors"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="portfolio_optimization",
            entities={"goal": "maximize_return"},
            response_text="Optimizing portfolio..."
        )
        
        error_scenarios = [
            OptimizationError("No feasible solution"),
            ConstraintError("Constraints cannot be satisfied"),
            ResourceError("Insufficient GPU memory"),
            ValueError("Invalid optimization parameters")
        ]
        
        for error in error_scenarios:
            orchestrator.portfolio_optimizer.optimize_portfolio.side_effect = error
            
            response = await orchestrator.process_user_input(
                "Optimize my portfolio",
                conversation_context
            )
            
            assert response.error is not None or response.partial_failure
            assert response.alternative_suggestions is not None
            assert len(response.alternative_suggestions) > 0
    
    @pytest.mark.asyncio
    async def test_data_processor_error(self, orchestrator, conversation_context):
        """Test handling of data processor errors"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="market_analysis",
            entities={"symbols": ["AAPL", "GOOGL"]},
            response_text="Fetching market data..."
        )
        
        error_scenarios = [
            ProcessingError("Data processing failed"),
            MarketDataError("Market data unavailable"),
            ConnectionError("API connection failed"),
            JSONDecodeError("Invalid response format", "", 0)
        ]
        
        for error in error_scenarios:
            orchestrator.data_processor.fetch_market_data.side_effect = error
            
            response = await orchestrator.process_user_input(
                "Show market data for AAPL",
                conversation_context
            )
            
            assert response.partial_failure
            assert response.cached_data is not None or response.fallback_data is not None
            assert "data unavailable" in response.warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_risk_engine_error(self, orchestrator, conversation_context):
        """Test handling of risk engine errors"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="risk_analysis",
            entities={"analysis_type": "stress_test"},
            response_text="Running risk analysis..."
        )
        
        error_scenarios = [
            RiskCalculationError("VaR calculation failed"),
            StressTestError("Stress test scenario invalid"),
            MemoryError("Insufficient memory for Monte Carlo"),
            RuntimeError("Risk model convergence failed")
        ]
        
        for error in error_scenarios:
            orchestrator.risk_engine.assess_portfolio_risk.side_effect = error
            
            response = await orchestrator.process_user_input(
                "Run risk analysis",
                conversation_context
            )
            
            assert response.error is not None or response.partial_failure
            assert response.simplified_analysis is not None
            assert "simplified risk metrics" in response.text.lower()
    
    @pytest.mark.asyncio
    async def test_avatar_system_error(self, orchestrator, conversation_context):
        """Test handling of avatar system errors"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="greeting",
            entities={},
            response_text="Hello! How can I help you?"
        )
        
        error_scenarios = [
            AnimationError("Animation file corrupted"),
            RenderingError("GPU rendering failed"),
            ResourceError("3D model loading failed"),
            Exception("Avatar system crash")
        ]
        
        for error in error_scenarios:
            orchestrator.avatar_system.generate_response.side_effect = error
            
            response = await orchestrator.process_user_input(
                "Hello",
                conversation_context
            )
            
            # Avatar errors should not block response
            assert response.text is not None
            assert response.avatar_state is None or response.avatar_state.is_fallback
            assert response.warnings is not None
            assert "avatar unavailable" in response.warnings[0].lower()


class TestTimeoutErrors:
    """Test timeout handling"""
    
    @pytest.mark.asyncio
    async def test_component_timeout(self, orchestrator, conversation_context):
        """Test handling of component timeouts"""
        async def slow_operation():
            await asyncio.sleep(10)  # Simulate slow operation
            return Mock()
        
        # Test conversation engine timeout
        orchestrator.conversation_engine.process_input.side_effect = slow_operation
        orchestrator.config.component_timeout_seconds = 0.1
        
        response = await orchestrator.process_user_input(
            "Test input",
            conversation_context
        )
        
        assert response.error is not None
        assert isinstance(response.error, TimeoutError)
        assert response.fallback_response is not None
    
    @pytest.mark.asyncio
    async def test_cascading_timeout(self, orchestrator, conversation_context):
        """Test handling of cascading timeouts across components"""
        # Mock successful but slow conversation response
        async def slow_conversation():
            await asyncio.sleep(0.1)
            return Mock(intent="portfolio_analysis", entities={})
        
        async def slow_analysis():
            await asyncio.sleep(5)
            return Mock()
        
        orchestrator.conversation_engine.process_input.side_effect = slow_conversation
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = slow_analysis
        orchestrator.config.request_timeout_seconds = 0.5
        
        response = await orchestrator.process_user_input(
            "Analyze portfolio",
            conversation_context
        )
        
        assert response.error is not None or response.partial_failure
        assert response.text is not None  # Should have partial response
        assert response.incomplete_operations == ["financial_analysis"]
    
    @pytest.mark.asyncio
    async def test_timeout_recovery(self, orchestrator, conversation_context):
        """Test recovery after timeout errors"""
        call_count = 0
        
        async def intermittent_timeout():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                await asyncio.sleep(10)  # Timeout on first call
            return Mock(intent="simple_query", response_text="Success")
        
        orchestrator.conversation_engine.process_input.side_effect = intermittent_timeout
        orchestrator.config.component_timeout_seconds = 0.1
        orchestrator.config.error_retry_attempts = 2
        
        response = await orchestrator.process_user_input(
            "Test query",
            conversation_context
        )
        
        # Should succeed on retry
        assert call_count == 2
        assert response.error is None
        assert response.text == "Success"


class TestResourceErrors:
    """Test resource exhaustion handling"""
    
    @pytest.mark.asyncio
    async def test_memory_exhaustion(self, orchestrator, conversation_context):
        """Test handling of memory exhaustion"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="large_computation",
            entities={}
        )
        
        # Simulate memory error
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = MemoryError(
            "Unable to allocate memory"
        )
        
        response = await orchestrator.process_user_input(
            "Analyze large portfolio",
            conversation_context
        )
        
        assert response.error is not None or response.partial_failure
        assert response.resource_optimization_applied
        assert "reduced computation" in response.warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_gpu_resource_error(self, orchestrator, conversation_context):
        """Test handling of GPU resource errors"""
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="portfolio_optimization",
            entities={}
        )
        
        # Simulate GPU resource error
        orchestrator.portfolio_optimizer.optimize_portfolio.side_effect = ResourceError(
            "CUDA out of memory"
        )
        
        response = await orchestrator.process_user_input(
            "Optimize portfolio with GPU",
            conversation_context
        )
        
        assert response.partial_failure
        assert response.fallback_mode == "cpu"
        assert "using CPU fallback" in response.warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self, orchestrator):
        """Test handling of concurrent request limits"""
        orchestrator.config.max_concurrent_sessions = 2
        
        # Create multiple concurrent requests
        async def create_request(i):
            context = ConversationContext(
                session_id=f"concurrent-{i}",
                user_id=f"user-{i}",
                conversation_history=[]
            )
            return await orchestrator.process_user_input("Test", context)
        
        # Launch more requests than limit
        tasks = [create_request(i) for i in range(5)]
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Some requests should be rejected
        errors = [r for r in responses if isinstance(r, Exception) or 
                  (hasattr(r, 'error') and r.error is not None)]
        
        assert len(errors) >= 3  # At least 3 should fail
        assert any("concurrent request limit" in str(e).lower() for e in errors)


class TestErrorRecovery:
    """Test error recovery mechanisms"""
    
    @pytest.mark.asyncio
    async def test_retry_mechanism(self, orchestrator, conversation_context):
        """Test automatic retry on transient errors"""
        call_count = 0
        
        def transient_error():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Transient network error")
            return Mock(intent="success", response_text="Success after retry")
        
        orchestrator.conversation_engine.process_input.side_effect = transient_error
        orchestrator.config.error_retry_attempts = 3
        orchestrator.config.error_retry_delay = 0.01
        
        response = await orchestrator.process_user_input(
            "Test retry",
            conversation_context
        )
        
        assert call_count == 3
        assert response.error is None
        assert response.text == "Success after retry"
        assert response.retry_count == 2
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self, orchestrator, conversation_context):
        """Test graceful degradation when components fail"""
        # Mock conversation success
        orchestrator.conversation_engine.process_input.return_value = Mock(
            intent="complex_analysis",
            entities={"include_optimization": True, "include_risk": True},
            response_text="Performing analysis..."
        )
        
        # Mock component failures
        orchestrator.portfolio_optimizer.optimize_portfolio.side_effect = Exception(
            "Optimizer failed"
        )
        orchestrator.risk_engine.assess_portfolio_risk.side_effect = Exception(
            "Risk engine failed"
        )
        
        # But financial analyzer works
        orchestrator.financial_analyzer.analyze_portfolio.return_value = Mock(
            performance_metrics={"return": 0.05}
        )
        
        response = await orchestrator.process_user_input(
            "Complete analysis",
            conversation_context
        )
        
        assert response.partial_failure
        assert response.completed_operations == ["conversation", "financial_analysis"]
        assert response.failed_operations == ["optimization", "risk_assessment"]
        assert response.financial_data is not None
        assert response.degraded_functionality is True
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, orchestrator, conversation_context):
        """Test circuit breaker pattern for failing components"""
        # Configure circuit breaker
        orchestrator.config.circuit_breaker_threshold = 3
        orchestrator.config.circuit_breaker_timeout = 0.5
        
        # Make component fail repeatedly
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = Exception(
            "Persistent failure"
        )
        
        # First few requests should attempt the operation
        for i in range(3):
            response = await orchestrator.process_user_input(
                f"Analyze portfolio {i}",
                conversation_context
            )
            assert response.error is not None or response.partial_failure
        
        # Circuit should now be open
        response = await orchestrator.process_user_input(
            "Analyze portfolio 4",
            conversation_context
        )
        
        assert response.circuit_breaker_open
        assert "service temporarily unavailable" in response.error.message.lower()
        
        # Wait for circuit breaker timeout
        await asyncio.sleep(0.6)
        
        # Circuit should attempt to close
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = None
        orchestrator.financial_analyzer.analyze_portfolio.return_value = Mock(
            performance_metrics={"return": 0.05}
        )
        
        response = await orchestrator.process_user_input(
            "Analyze portfolio 5",
            conversation_context
        )
        
        assert response.error is None
        assert response.circuit_breaker_open is False


class TestErrorLogging:
    """Test error logging and monitoring"""
    
    @pytest.mark.asyncio
    async def test_error_logging(self, orchestrator, conversation_context, caplog):
        """Test that errors are properly logged"""
        orchestrator.conversation_engine.process_input.side_effect = Exception(
            "Test error for logging"
        )
        
        with caplog.at_level(logging.ERROR):
            response = await orchestrator.process_user_input(
                "Test logging",
                conversation_context
            )
        
        assert response.error is not None
        assert len(caplog.records) > 0
        assert any("Test error for logging" in record.message for record in caplog.records)
        assert any(record.levelname == "ERROR" for record in caplog.records)
    
    @pytest.mark.asyncio
    async def test_error_metrics(self, orchestrator, conversation_context):
        """Test error metrics collection"""
        # Enable metrics
        orchestrator.config.enable_metrics = True
        orchestrator.metrics_collector.reset()
        
        # Generate various errors
        error_types = [
            ValidationError("Invalid input"),
            TimeoutError("Operation timeout"),
            ComponentError("Component failure"),
            ResourceError("Resource exhausted")
        ]
        
        for error in error_types:
            orchestrator.conversation_engine.process_input.side_effect = error
            await orchestrator.process_user_input("Test", conversation_context)
        
        # Check metrics
        metrics = orchestrator.metrics_collector.get_error_metrics()
        
        assert metrics["total_errors"] == 4
        assert metrics["error_by_type"]["ValidationError"] == 1
        assert metrics["error_by_type"]["TimeoutError"] == 1
        assert metrics["error_by_type"]["ComponentError"] == 1
        assert metrics["error_by_type"]["ResourceError"] == 1
        assert metrics["error_rate"] > 0
    
    @pytest.mark.asyncio
    async def test_error_reporting(self, orchestrator, conversation_context):
        """Test error reporting and alerting"""
        # Configure error reporting
        orchestrator.config.error_reporting_threshold = 2
        orchestrator.config.error_alert_email = "admin@example.com"
        
        # Mock alert sender
        orchestrator.alert_sender = Mock()
        
        # Generate errors
        orchestrator.conversation_engine.process_input.side_effect = Exception(
            "Critical error"
        )
        
        # First error - no alert
        await orchestrator.process_user_input("Test 1", conversation_context)
        orchestrator.alert_sender.send_alert.assert_not_called()
        
        # Second error - should trigger alert
        await orchestrator.process_user_input("Test 2", conversation_context)
        orchestrator.alert_sender.send_alert.assert_called_once()
        
        alert_data = orchestrator.alert_sender.send_alert.call_args[0][0]
        assert alert_data["error_count"] == 2
        assert alert_data["error_type"] == "Exception"
        assert "Critical error" in alert_data["message"]


class TestEdgeCases:
    """Test edge cases and unusual error conditions"""
    
    @pytest.mark.asyncio
    async def test_nested_errors(self, orchestrator, conversation_context):
        """Test handling of nested errors"""
        # Create nested error scenario
        def create_nested_error():
            try:
                raise ValueError("Inner error")
            except ValueError as e:
                raise ComponentError("Outer error") from e
        
        orchestrator.conversation_engine.process_input.side_effect = create_nested_error
        
        response = await orchestrator.process_user_input(
            "Test nested error",
            conversation_context
        )
        
        assert response.error is not None
        assert "Outer error" in response.error.message
        assert response.error.cause is not None
        assert "Inner error" in str(response.error.cause)
    
    @pytest.mark.asyncio
    async def test_error_during_error_handling(self, orchestrator, conversation_context):
        """Test error that occurs during error handling"""
        # Make primary operation fail
        orchestrator.conversation_engine.process_input.side_effect = Exception(
            "Primary error"
        )
        
        # Make error handler also fail
        orchestrator.error_handler.handle_error = Mock(
            side_effect=Exception("Error handler failed")
        )
        
        response = await orchestrator.process_user_input(
            "Test error in handler",
            conversation_context
        )
        
        # Should still return a response
        assert response is not None
        assert response.error is not None
        assert response.fallback_response is not None
        assert "system error" in response.fallback_response.lower()
    
    @pytest.mark.asyncio
    async def test_infinite_retry_prevention(self, orchestrator, conversation_context):
        """Test prevention of infinite retry loops"""
        call_count = 0
        
        def persistent_error():
            nonlocal call_count
            call_count += 1
            raise ConnectionError("Persistent network error")
        
        orchestrator.conversation_engine.process_input.side_effect = persistent_error
        orchestrator.config.error_retry_attempts = 5
        orchestrator.config.max_retry_duration_seconds = 0.5
        
        start_time = time.time()
        response = await orchestrator.process_user_input(
            "Test infinite retry",
            conversation_context
        )
        duration = time.time() - start_time
        
        # Should stop retrying after max attempts or duration
        assert call_count <= 6  # Initial + 5 retries
        assert duration < 1.0  # Should not retry forever
        assert response.error is not None
        assert response.retry_count == 5
    
    @pytest.mark.asyncio
    async def test_memory_cleanup_after_error(self, orchestrator, conversation_context):
        """Test memory cleanup after errors"""
        import gc
        import weakref
        
        # Create objects that should be cleaned up
        large_data = [np.random.rand(1000, 1000) for _ in range(5)]
        weak_refs = [weakref.ref(d) for d in large_data]
        
        # Mock error that references large data
        def error_with_data():
            # Reference large_data in error
            raise MemoryError(f"Failed with {len(large_data)} matrices")
        
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = error_with_data
        
        response = await orchestrator.process_user_input(
            "Analyze with large data",
            conversation_context
        )
        
        assert response.error is not None
        
        # Clear references and force garbage collection
        del large_data
        del error_with_data
        gc.collect()
        
        # Check that memory was cleaned up
        alive_refs = [ref for ref in weak_refs if ref() is not None]
        assert len(alive_refs) == 0  # All should be garbage collected
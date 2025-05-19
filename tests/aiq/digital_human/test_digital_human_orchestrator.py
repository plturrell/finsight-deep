"""Integration tests for Digital Human Orchestrator"""

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
    SystemHealth,
    IntentClassification,
    Response,
    SessionState
)
from aiq.digital_human.conversation_engine import (
    ConversationEngine,
    ConversationConfig,
    DialogueState,
    UserIntent,
    ConversationResponse
)
from aiq.digital_human.financial.mcts_financial_analyzer import (
    MCTSFinancialAnalyzer,
    FinancialState,
    FinancialAction,
    AnalysisResult
)
from aiq.digital_human.financial.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationResult,
    Asset,
    OptimizationConstraints
)
from aiq.digital_human.financial.financial_data_processor import (
    FinancialDataProcessor,
    MarketData,
    MarketIndicators,
    FinancialMetrics
)
from aiq.digital_human.financial.risk_assessment_engine import (
    RiskAssessmentEngine,
    RiskProfile,
    StressTestResult,
    RiskMitigation
)
from aiq.digital_human.avatar import (
    AvatarSystem,
    AnimationState,
    EmotionalState,
    GestureType
)


# Test fixtures
@pytest.fixture
def config():
    """Create orchestrator configuration"""
    return OrchestratorConfig(
        enable_gpu=False,
        max_concurrent_sessions=10,
        session_timeout_minutes=30,
        health_check_interval_seconds=60,
        enable_caching=True,
        cache_ttl_seconds=3600,
        enable_logging=True,
        log_level="INFO",
        enable_metrics=True,
        enable_tracing=True
    )


@pytest.fixture
def conversation_context():
    """Create sample conversation context"""
    return ConversationContext(
        session_id="test-session-123",
        user_id="user-456",
        conversation_history=[
            {"role": "user", "content": "What's my portfolio performance?"},
            {"role": "assistant", "content": "Let me analyze your portfolio..."}
        ],
        user_profile={
            "name": "Test User",
            "risk_tolerance": "moderate",
            "investment_goals": ["retirement", "growth"],
            "portfolio_value": 100000.0
        },
        current_state={
            "intent": "portfolio_analysis",
            "entities": {"timeframe": "1M", "metrics": ["return", "volatility"]},
            "sentiment": "neutral"
        }
    )


@pytest.fixture
def financial_state():
    """Create sample financial state"""
    return FinancialState(
        portfolio={
            "AAPL": {"quantity": 100, "avg_price": 150.0},
            "GOOGL": {"quantity": 50, "avg_price": 2800.0},
            "MSFT": {"quantity": 75, "avg_price": 300.0}
        },
        cash_balance=10000.0,
        total_value=325000.0,
        historical_returns=[0.05, 0.03, -0.02, 0.04, 0.06, -0.01, 0.03],
        risk_metrics={
            "portfolio_volatility": 0.15,
            "sharpe_ratio": 1.2,
            "max_drawdown": -0.08,
            "value_at_risk": -5000.0
        }
    )


@pytest.fixture
def market_data():
    """Create sample market data"""
    return [
        MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=150.0,
            high=152.0,
            low=149.0,
            close=151.0,
            volume=1000000,
            market_cap=2.5e12,
            pe_ratio=25.0,
            dividend_yield=0.01
        ),
        MarketData(
            symbol="GOOGL",
            timestamp=datetime.now(),
            open=2800.0,
            high=2820.0,
            low=2790.0,
            close=2810.0,
            volume=500000,
            market_cap=1.8e12,
            pe_ratio=30.0,
            dividend_yield=0.0
        )
    ]


@pytest.fixture
async def orchestrator(config):
    """Create Digital Human Orchestrator with mocked components"""
    # Create mocked components
    conversation_engine = AsyncMock(spec=ConversationEngine)
    financial_analyzer = AsyncMock(spec=MCTSFinancialAnalyzer)
    portfolio_optimizer = AsyncMock(spec=PortfolioOptimizer)
    data_processor = AsyncMock(spec=FinancialDataProcessor)
    risk_engine = AsyncMock(spec=RiskAssessmentEngine)
    avatar_system = AsyncMock(spec=AvatarSystem)
    
    # Create orchestrator
    orchestrator = DigitalHumanOrchestrator(config)
    
    # Inject mocked components
    orchestrator.conversation_engine = conversation_engine
    orchestrator.financial_analyzer = financial_analyzer
    orchestrator.portfolio_optimizer = portfolio_optimizer
    orchestrator.data_processor = data_processor
    orchestrator.risk_engine = risk_engine
    orchestrator.avatar_system = avatar_system
    
    # Mock component initialization
    orchestrator.conversation_engine.initialize = AsyncMock()
    orchestrator.financial_analyzer.initialize = AsyncMock()
    orchestrator.portfolio_optimizer.initialize = AsyncMock()
    orchestrator.data_processor.initialize = AsyncMock()
    orchestrator.risk_engine.initialize = AsyncMock()
    orchestrator.avatar_system.initialize = AsyncMock()
    
    await orchestrator.initialize()
    
    yield orchestrator
    
    await orchestrator.shutdown()


# Test classes
class TestDigitalHumanOrchestrator:
    """Test Digital Human Orchestrator integration"""
    
    @pytest.mark.asyncio
    async def test_initialize_orchestrator(self, config):
        """Test orchestrator initialization"""
        orchestrator = DigitalHumanOrchestrator(config)
        
        # Mock component initialization
        with patch.object(orchestrator, '_initialize_components', new_callable=AsyncMock):
            await orchestrator.initialize()
            
            assert orchestrator.is_initialized
            assert orchestrator.health_monitor is not None
            assert orchestrator.session_manager is not None
            assert orchestrator.metrics_collector is not None
    
    @pytest.mark.asyncio
    async def test_process_user_input(self, orchestrator, conversation_context):
        """Test processing user input through the orchestration pipeline"""
        user_input = "What's my portfolio performance over the last month?"
        
        # Mock conversation engine response
        conversation_response = ConversationResponse(
            intent="portfolio_analysis",
            entities={"timeframe": "1M", "metrics": ["return", "volatility"]},
            sentiment="neutral",
            response_text="I'll analyze your portfolio performance for the last month.",
            confidence=0.95,
            suggested_actions=["show_returns", "show_risk_metrics"]
        )
        orchestrator.conversation_engine.process_input.return_value = conversation_response
        
        # Mock financial analysis
        analysis_result = AnalysisResult(
            recommended_action=FinancialAction(
                action_type="hold",
                symbol="PORTFOLIO",
                quantity=0,
                confidence=0.85
            ),
            expected_value=105000.0,
            risk_assessment={"volatility": 0.15, "max_loss": -5000.0},
            performance_metrics={
                "monthly_return": 0.042,
                "sharpe_ratio": 1.2,
                "volatility": 0.15
            }
        )
        orchestrator.financial_analyzer.analyze_portfolio.return_value = analysis_result
        
        # Mock avatar response
        avatar_response = AnimationState(
            gesture=GestureType.PRESENTING,
            facial_expression="confident",
            body_posture="open",
            eye_contact=True
        )
        orchestrator.avatar_system.generate_response.return_value = avatar_response
        
        # Process input
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Verify response
        assert response is not None
        assert response.text == conversation_response.response_text
        assert response.intent == conversation_response.intent
        assert response.financial_data is not None
        assert response.financial_data["monthly_return"] == 0.042
        assert response.avatar_state is not None
        assert response.avatar_state.gesture == GestureType.PRESENTING
        
        # Verify component interactions
        orchestrator.conversation_engine.process_input.assert_called_once()
        orchestrator.financial_analyzer.analyze_portfolio.assert_called_once()
        orchestrator.avatar_system.generate_response.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization_flow(self, orchestrator, conversation_context, financial_state):
        """Test portfolio optimization workflow integration"""
        user_input = "Optimize my portfolio for maximum Sharpe ratio"
        
        # Mock conversation understanding
        conversation_response = ConversationResponse(
            intent="portfolio_optimization",
            entities={"optimization_goal": "sharpe_ratio", "risk_constraint": "moderate"},
            sentiment="neutral",
            response_text="I'll optimize your portfolio to maximize the Sharpe ratio.",
            confidence=0.92
        )
        orchestrator.conversation_engine.process_input.return_value = conversation_response
        
        # Mock market data processing
        market_indicators = MarketIndicators(
            symbol="MARKET",
            timestamp=datetime.now(),
            sma_20=3000.0,
            sma_50=2950.0,
            rsi=55.0,
            macd={"macd": 10.0, "signal": 5.0, "histogram": 5.0},
            bollinger_bands={"upper": 3100.0, "middle": 3000.0, "lower": 2900.0}
        )
        orchestrator.data_processor.calculate_indicators.return_value = market_indicators
        
        # Mock portfolio optimization
        optimization_result = OptimizationResult(
            weights={"AAPL": 0.4, "GOOGL": 0.3, "MSFT": 0.3},
            expected_return=0.12,
            expected_volatility=0.15,
            sharpe_ratio=0.8,
            metrics={
                "max_drawdown": -0.08,
                "value_at_risk": -5000.0,
                "conditional_value_at_risk": -7000.0
            }
        )
        orchestrator.portfolio_optimizer.optimize_portfolio.return_value = optimization_result
        
        # Mock risk assessment
        risk_profile = RiskProfile(
            portfolio_volatility=0.15,
            value_at_risk=-5000.0,
            conditional_value_at_risk=-7000.0,
            max_drawdown=-0.08,
            sharpe_ratio=0.8,
            downside_deviation=0.10,
            tail_risk_probability=0.05
        )
        orchestrator.risk_engine.assess_portfolio_risk.return_value = risk_profile
        
        # Process optimization request
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Verify optimization flow
        assert response.intent == "portfolio_optimization"
        assert response.optimization_result is not None
        assert response.optimization_result["sharpe_ratio"] == 0.8
        assert response.risk_assessment is not None
        assert response.risk_assessment["value_at_risk"] == -5000.0
        
        # Verify all components were called
        orchestrator.conversation_engine.process_input.assert_called_once()
        orchestrator.data_processor.calculate_indicators.assert_called()
        orchestrator.portfolio_optimizer.optimize_portfolio.assert_called_once()
        orchestrator.risk_engine.assess_portfolio_risk.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_risk_analysis_workflow(self, orchestrator, conversation_context, financial_state, market_data):
        """Test risk analysis workflow integration"""
        user_input = "What are the risks in my portfolio? Run stress tests."
        
        # Mock conversation understanding
        conversation_response = ConversationResponse(
            intent="risk_analysis",
            entities={"analysis_type": "stress_test", "include_mitigation": True},
            sentiment="concerned",
            response_text="I'll perform a comprehensive risk analysis including stress tests.",
            confidence=0.88
        )
        orchestrator.conversation_engine.process_input.return_value = conversation_response
        
        # Mock risk assessment
        risk_profile = RiskProfile(
            portfolio_volatility=0.18,
            value_at_risk=-8000.0,
            conditional_value_at_risk=-12000.0,
            max_drawdown=-0.15,
            sharpe_ratio=0.6,
            downside_deviation=0.12,
            tail_risk_probability=0.08
        )
        orchestrator.risk_engine.assess_portfolio_risk.return_value = risk_profile
        
        # Mock stress test results
        stress_results = [
            StressTestResult(
                scenario="Market Crash",
                probability=0.10,
                portfolio_impact=-0.25,
                value_change=-81250.0,
                affected_positions={"AAPL": -0.30, "GOOGL": -0.25, "MSFT": -0.20},
                recovery_time_estimate=180
            ),
            StressTestResult(
                scenario="Interest Rate Spike",
                probability=0.15,
                portfolio_impact=-0.10,
                value_change=-32500.0,
                affected_positions={"AAPL": -0.12, "GOOGL": -0.08, "MSFT": -0.10},
                recovery_time_estimate=90
            )
        ]
        orchestrator.risk_engine.run_stress_tests.return_value = stress_results
        
        # Mock mitigation strategies
        mitigation_strategies = [
            RiskMitigation(
                strategy="Diversification",
                description="Add defensive sectors and international exposure",
                expected_risk_reduction=0.20,
                implementation_cost=500.0,
                recommended_actions=[
                    "Add 10% allocation to utilities sector",
                    "Add 15% international bonds"
                ]
            ),
            RiskMitigation(
                strategy="Hedging",
                description="Use options to protect downside",
                expected_risk_reduction=0.30,
                implementation_cost=1500.0,
                recommended_actions=[
                    "Buy put options on major holdings",
                    "Consider VIX calls for tail risk"
                ]
            )
        ]
        orchestrator.risk_engine.generate_mitigation_strategies.return_value = mitigation_strategies
        
        # Process risk analysis request
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Verify risk analysis flow
        assert response.intent == "risk_analysis"
        assert response.risk_profile is not None
        assert response.risk_profile["portfolio_volatility"] == 0.18
        assert len(response.stress_test_results) == 2
        assert response.stress_test_results[0]["scenario"] == "Market Crash"
        assert len(response.mitigation_strategies) == 2
        assert response.mitigation_strategies[0]["strategy"] == "Diversification"
        
        # Verify emotional response based on risk
        assert response.avatar_state.facial_expression == "concerned"
        assert response.suggested_actions == ["review_risk_mitigation", "adjust_portfolio"]
    
    @pytest.mark.asyncio
    async def test_multi_intent_conversation(self, orchestrator, conversation_context):
        """Test handling multiple intents in a single conversation"""
        user_input = "Show me my performance and then optimize for lower risk"
        
        # Mock complex intent understanding
        conversation_response = ConversationResponse(
            intent="multi_intent",
            entities={
                "intents": ["portfolio_analysis", "portfolio_optimization"],
                "analysis_metrics": ["return", "risk"],
                "optimization_goal": "risk_reduction"
            },
            sentiment="neutral",
            response_text="I'll show your performance and then optimize for lower risk.",
            confidence=0.85,
            suggested_actions=["show_performance", "optimize_portfolio"]
        )
        orchestrator.conversation_engine.process_input.return_value = conversation_response
        
        # Mock performance analysis
        performance_metrics = FinancialMetrics(
            total_return=0.08,
            annualized_return=0.096,
            volatility=0.18,
            sharpe_ratio=0.53,
            max_drawdown=-0.12,
            win_rate=0.58,
            profit_factor=1.4,
            recovery_factor=0.67
        )
        orchestrator.data_processor.calculate_performance_metrics.return_value = performance_metrics
        
        # Mock risk-optimized portfolio
        optimization_result = OptimizationResult(
            weights={"AAPL": 0.25, "GOOGL": 0.25, "MSFT": 0.25, "BND": 0.25},
            expected_return=0.07,
            expected_volatility=0.12,
            sharpe_ratio=0.58,
            metrics={
                "risk_reduction": 0.33,
                "return_impact": -0.01
            }
        )
        orchestrator.portfolio_optimizer.optimize_portfolio.return_value = optimization_result
        
        # Process multi-intent request
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Verify multi-intent handling
        assert response.intent == "multi_intent"
        assert "performance" in response.financial_data
        assert "optimization" in response.financial_data
        assert response.financial_data["performance"]["annualized_return"] == 0.096
        assert response.financial_data["optimization"]["expected_volatility"] == 0.12
        assert len(response.suggested_actions) > 0
    
    @pytest.mark.asyncio
    async def test_session_management(self, orchestrator, conversation_context):
        """Test session management and state persistence"""
        session_id = "test-session-789"
        
        # Create new session
        session = await orchestrator.create_session(
            user_id="user-123",
            initial_context={"portfolio_value": 100000.0}
        )
        
        assert session.session_id is not None
        assert session.user_id == "user-123"
        assert session.state == SessionState.ACTIVE
        
        # Update session state
        await orchestrator.update_session_state(
            session.session_id,
            {
                "last_action": "portfolio_analysis",
                "risk_tolerance": "moderate"
            }
        )
        
        # Retrieve session
        retrieved_session = await orchestrator.get_session(session.session_id)
        assert retrieved_session.context["last_action"] == "portfolio_analysis"
        assert retrieved_session.context["risk_tolerance"] == "moderate"
        
        # Close session
        await orchestrator.close_session(session.session_id)
        closed_session = await orchestrator.get_session(session.session_id)
        assert closed_session.state == SessionState.CLOSED
    
    @pytest.mark.asyncio
    async def test_health_monitoring(self, orchestrator):
        """Test health monitoring and component status"""
        # Mock component health checks
        orchestrator.conversation_engine.health_check.return_value = {"status": "healthy", "latency": 0.01}
        orchestrator.financial_analyzer.health_check.return_value = {"status": "healthy", "memory": "2GB"}
        orchestrator.portfolio_optimizer.health_check.return_value = {"status": "healthy", "gpu": True}
        orchestrator.data_processor.health_check.return_value = {"status": "healthy", "cache_hit_rate": 0.85}
        orchestrator.risk_engine.health_check.return_value = {"status": "healthy", "queue_size": 0}
        orchestrator.avatar_system.health_check.return_value = {"status": "healthy", "fps": 60}
        
        # Get system health
        health = await orchestrator.get_system_health()
        
        assert health.overall_status == "healthy"
        assert health.component_status["conversation_engine"] == "healthy"
        assert health.component_status["financial_analyzer"] == "healthy"
        assert health.component_status["portfolio_optimizer"] == "healthy"
        assert health.metrics["cache_hit_rate"] == 0.85
        assert health.metrics["avatar_fps"] == 60
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, orchestrator, conversation_context):
        """Test error handling and graceful degradation"""
        user_input = "Analyze my portfolio"
        
        # Mock conversation engine error
        orchestrator.conversation_engine.process_input.side_effect = Exception("NLP service unavailable")
        
        # Process should handle error gracefully
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        assert response.error is not None
        assert "service unavailable" in response.error.lower()
        assert response.fallback_response is not None
        assert response.avatar_state.facial_expression == "apologetic"
        
        # Reset and test partial failure
        orchestrator.conversation_engine.process_input.side_effect = None
        orchestrator.conversation_engine.process_input.return_value = ConversationResponse(
            intent="portfolio_analysis",
            entities={},
            sentiment="neutral",
            response_text="Analyzing portfolio...",
            confidence=0.9
        )
        
        # Mock financial analyzer error
        orchestrator.financial_analyzer.analyze_portfolio.side_effect = Exception("Market data unavailable")
        
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Should provide partial response
        assert response.text is not None
        assert response.partial_failure is True
        assert "data unavailable" in response.warnings[0].lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_request_handling(self, orchestrator, conversation_context):
        """Test handling multiple concurrent requests"""
        # Create multiple concurrent requests
        requests = [
            ("What's my portfolio value?", "session-1"),
            ("Optimize my portfolio", "session-2"),
            ("Show risk analysis", "session-3"),
            ("What are market trends?", "session-4")
        ]
        
        # Mock different responses for each request type
        response_map = {
            "portfolio_value": ConversationResponse(
                intent="portfolio_value",
                entities={},
                sentiment="neutral",
                response_text="Your portfolio value is $325,000",
                confidence=0.95
            ),
            "portfolio_optimization": ConversationResponse(
                intent="portfolio_optimization",
                entities={"goal": "balanced"},
                sentiment="neutral",
                response_text="Optimizing for balanced growth...",
                confidence=0.90
            ),
            "risk_analysis": ConversationResponse(
                intent="risk_analysis",
                entities={"type": "comprehensive"},
                sentiment="neutral",
                response_text="Analyzing portfolio risks...",
                confidence=0.88
            ),
            "market_trends": ConversationResponse(
                intent="market_trends",
                entities={"timeframe": "current"},
                sentiment="neutral",
                response_text="Current market trends show...",
                confidence=0.85
            )
        }
        
        def mock_conversation_response(user_input, context):
            if "value" in user_input.lower():
                return response_map["portfolio_value"]
            elif "optimize" in user_input.lower():
                return response_map["portfolio_optimization"]
            elif "risk" in user_input.lower():
                return response_map["risk_analysis"]
            else:
                return response_map["market_trends"]
        
        orchestrator.conversation_engine.process_input.side_effect = mock_conversation_response
        
        # Process requests concurrently
        async def process_request(user_input, session_id):
            context = ConversationContext(
                session_id=session_id,
                user_id="user-123",
                conversation_history=[]
            )
            return await orchestrator.process_user_input(user_input, context)
        
        # Execute concurrent requests
        responses = await asyncio.gather(*[
            process_request(user_input, session_id)
            for user_input, session_id in requests
        ])
        
        # Verify all requests were processed
        assert len(responses) == 4
        assert all(response is not None for response in responses)
        assert responses[0].text == "Your portfolio value is $325,000"
        assert responses[1].intent == "portfolio_optimization"
        assert responses[2].intent == "risk_analysis"
        assert responses[3].intent == "market_trends"
    
    @pytest.mark.asyncio
    async def test_caching_and_performance(self, orchestrator, conversation_context, financial_state):
        """Test caching mechanisms and performance optimization"""
        user_input = "Show my portfolio performance"
        
        # Mock responses
        conversation_response = ConversationResponse(
            intent="portfolio_analysis",
            entities={"timeframe": "1M"},
            sentiment="neutral",
            response_text="Analyzing performance...",
            confidence=0.92
        )
        orchestrator.conversation_engine.process_input.return_value = conversation_response
        
        analysis_result = AnalysisResult(
            recommended_action=FinancialAction("hold", "PORTFOLIO", 0, 0.9),
            expected_value=105000.0,
            risk_assessment={"volatility": 0.15},
            performance_metrics={"return": 0.05}
        )
        orchestrator.financial_analyzer.analyze_portfolio.return_value = analysis_result
        
        # First request - should cache results
        start_time = datetime.now()
        response1 = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        first_duration = (datetime.now() - start_time).total_seconds()
        
        # Second identical request - should use cache
        start_time = datetime.now()
        response2 = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        second_duration = (datetime.now() - start_time).total_seconds()
        
        # Verify caching worked
        assert response1.financial_data == response2.financial_data
        assert second_duration < first_duration  # Cached response should be faster
        
        # Financial analyzer should only be called once due to caching
        orchestrator.financial_analyzer.analyze_portfolio.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation_flow(self, orchestrator):
        """Test complete end-to-end conversation flow"""
        # Create new session
        session = await orchestrator.create_session(
            user_id="user-456",
            initial_context={
                "portfolio_value": 250000.0,
                "risk_tolerance": "moderate"
            }
        )
        
        # Conversation flow
        conversation_flow = [
            ("Hello, I'd like to review my portfolio", "greeting"),
            ("What's my current portfolio value?", "portfolio_value"),
            ("How has it performed this month?", "performance_analysis"),
            ("I'm concerned about risk. Can you analyze it?", "risk_analysis"),
            ("What changes would you recommend?", "recommendation"),
            ("Please optimize for lower risk", "portfolio_optimization"),
            ("Thank you, that's helpful", "closing")
        ]
        
        # Mock conversation responses for each turn
        mock_responses = {
            "greeting": ConversationResponse(
                intent="greeting",
                entities={},
                sentiment="neutral",
                response_text="Hello! I'd be happy to help you review your portfolio.",
                confidence=0.95
            ),
            "portfolio_value": ConversationResponse(
                intent="portfolio_value",
                entities={},
                sentiment="neutral",
                response_text="Your current portfolio value is $250,000.",
                confidence=0.98
            ),
            "performance_analysis": ConversationResponse(
                intent="performance_analysis",
                entities={"timeframe": "1M"},
                sentiment="neutral",
                response_text="This month your portfolio has gained 3.2%.",
                confidence=0.92
            ),
            "risk_analysis": ConversationResponse(
                intent="risk_analysis",
                entities={"concern_level": "high"},
                sentiment="concerned",
                response_text="Let me analyze your portfolio risk...",
                confidence=0.89
            ),
            "recommendation": ConversationResponse(
                intent="recommendation",
                entities={"focus": "risk_reduction"},
                sentiment="helpful",
                response_text="Based on the analysis, I recommend...",
                confidence=0.87
            ),
            "portfolio_optimization": ConversationResponse(
                intent="portfolio_optimization",
                entities={"goal": "risk_reduction"},
                sentiment="confident",
                response_text="I'll optimize your portfolio for lower risk.",
                confidence=0.91
            ),
            "closing": ConversationResponse(
                intent="closing",
                entities={},
                sentiment="positive",
                response_text="You're welcome! Feel free to ask if you need anything else.",
                confidence=0.96
            )
        }
        
        conversation_history = []
        
        for user_input, expected_intent in conversation_flow:
            # Mock conversation response
            orchestrator.conversation_engine.process_input.return_value = mock_responses[expected_intent]
            
            # Create context with conversation history
            context = ConversationContext(
                session_id=session.session_id,
                user_id=session.user_id,
                conversation_history=conversation_history,
                user_profile=session.initial_context
            )
            
            # Process user input
            response = await orchestrator.process_user_input(user_input, context)
            
            # Verify response
            assert response is not None
            assert response.intent == expected_intent
            assert response.session_id == session.session_id
            
            # Update conversation history
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": response.text})
            
            # Verify avatar emotional state matches conversation sentiment
            if expected_intent == "risk_analysis":
                assert response.avatar_state.facial_expression == "concerned"
            elif expected_intent == "closing":
                assert response.avatar_state.facial_expression == "friendly"
        
        # Verify session state was maintained
        final_session = await orchestrator.get_session(session.session_id)
        assert len(final_session.conversation_history) == len(conversation_flow) * 2
        assert final_session.metrics["total_interactions"] == len(conversation_flow)
    
    @pytest.mark.asyncio
    async def test_resource_cleanup(self, orchestrator):
        """Test proper resource cleanup on shutdown"""
        # Create multiple sessions
        sessions = []
        for i in range(5):
            session = await orchestrator.create_session(
                user_id=f"user-{i}",
                initial_context={"test": True}
            )
            sessions.append(session)
        
        # Verify sessions are active
        active_sessions = await orchestrator.get_active_sessions()
        assert len(active_sessions) == 5
        
        # Shutdown orchestrator
        await orchestrator.shutdown()
        
        # Verify all components were properly shut down
        orchestrator.conversation_engine.shutdown.assert_called_once()
        orchestrator.financial_analyzer.shutdown.assert_called_once()
        orchestrator.portfolio_optimizer.shutdown.assert_called_once()
        orchestrator.data_processor.shutdown.assert_called_once()
        orchestrator.risk_engine.shutdown.assert_called_once()
        orchestrator.avatar_system.shutdown.assert_called_once()
        
        # Verify sessions were closed
        assert orchestrator.session_manager.active_sessions == 0
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("failure_component", [
        "conversation_engine",
        "financial_analyzer",
        "risk_engine"
    ])
    async def test_component_failure_resilience(self, orchestrator, conversation_context, failure_component):
        """Test system resilience when individual components fail"""
        user_input = "Analyze my portfolio and show risks"
        
        # Mock successful conversation understanding
        orchestrator.conversation_engine.process_input.return_value = ConversationResponse(
            intent="combined_analysis",
            entities={"include_risk": True},
            sentiment="neutral",
            response_text="I'll analyze your portfolio and risks.",
            confidence=0.90
        )
        
        # Simulate component failure
        component = getattr(orchestrator, failure_component)
        component.side_effect = Exception(f"{failure_component} failed")
        
        # Process should handle component failure gracefully
        response = await orchestrator.process_user_input(
            user_input,
            conversation_context
        )
        
        # Verify graceful degradation
        assert response is not None
        if failure_component == "conversation_engine":
            assert response.fallback_response is not None
        else:
            assert response.partial_failure is True
            assert len(response.warnings) > 0
            assert failure_component in response.warnings[0]
        
        # System should still be operational
        health = await orchestrator.get_system_health()
        assert health.overall_status in ["degraded", "healthy"]
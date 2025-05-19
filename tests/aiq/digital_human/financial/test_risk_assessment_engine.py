"""
Comprehensive tests for Risk Assessment Engine

Tests risk analysis, stress testing, Monte Carlo simulations,
and risk mitigation strategies.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from aiq.digital_human.financial.risk_assessment_engine import (
    RiskAssessmentEngine,
    RiskProfile,
    StressTestResult,
    RiskMitigation
)
from aiq.digital_human.financial.mcts_financial_analyzer import FinancialState
from aiq.digital_human.financial.financial_data_processor import MarketData


class TestRiskProfile:
    """Test RiskProfile dataclass"""
    
    def test_risk_profile_creation(self):
        """Test creating risk profile"""
        profile = RiskProfile(
            portfolio_var=-0.05,
            portfolio_cvar=-0.08,
            max_drawdown=-0.15,
            volatility=0.18,
            beta=1.2,
            correlation_risk=0.65,
            concentration_risk=0.35,
            liquidity_risk=0.2,
            tail_risk=0.12,
            black_swan_probability=0.02,
            risk_rating="medium"
        )
        
        assert profile.portfolio_var == -0.05
        assert profile.portfolio_cvar == -0.08
        assert profile.max_drawdown == -0.15
        assert profile.volatility == 0.18
        assert profile.beta == 1.2
        assert profile.correlation_risk == 0.65
        assert profile.concentration_risk == 0.35
        assert profile.liquidity_risk == 0.2
        assert profile.tail_risk == 0.12
        assert profile.black_swan_probability == 0.02
        assert profile.risk_rating == "medium"


class TestStressTestResult:
    """Test StressTestResult dataclass"""
    
    def test_stress_test_result_creation(self):
        """Test creating stress test result"""
        result = StressTestResult(
            scenario_name="Market Crash",
            portfolio_impact=-0.30,
            worst_case_loss=-0.45,
            recovery_time=180,
            affected_assets=["AAPL", "GOOGL", "MSFT"],
            recommendations=[
                "Increase cash allocation",
                "Add defensive assets",
                "Implement stop-loss orders"
            ]
        )
        
        assert result.scenario_name == "Market Crash"
        assert result.portfolio_impact == -0.30
        assert result.worst_case_loss == -0.45
        assert result.recovery_time == 180
        assert len(result.affected_assets) == 3
        assert len(result.recommendations) == 3


class TestRiskMitigation:
    """Test RiskMitigation dataclass"""
    
    def test_risk_mitigation_creation(self):
        """Test creating risk mitigation strategy"""
        mitigation = RiskMitigation(
            strategy_type="var_reduction",
            target_risk_level=0.05,
            suggested_actions=[
                {"action": "reduce_position", "symbol": "TSLA", "percentage": 0.3},
                {"action": "add_diversification", "asset_class": "bonds", "percentage": 0.15}
            ],
            expected_risk_reduction=0.25,
            implementation_cost=5000.0
        )
        
        assert mitigation.strategy_type == "var_reduction"
        assert mitigation.target_risk_level == 0.05
        assert len(mitigation.suggested_actions) == 2
        assert mitigation.expected_risk_reduction == 0.25
        assert mitigation.implementation_cost == 5000.0


class TestRiskAssessmentEngine:
    """Test Risk Assessment Engine implementation"""
    
    @pytest.fixture
    def engine(self):
        """Create engine instance"""
        return RiskAssessmentEngine(
            device="cpu",
            confidence_level=0.95,
            monte_carlo_simulations=1000,
            enable_ml_detection=False  # Disable for faster tests
        )
    
    @pytest.fixture
    def portfolio_state(self):
        """Create test portfolio state"""
        return FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 200, "GOOGL": 50, "MSFT": 100, "AMZN": 30},
            cash_balance=10000.0,
            risk_tolerance=0.6,
            time_horizon=252,
            market_conditions={"volatility": 0.18, "vix": 20.5},
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def market_data(self):
        """Create test market data"""
        data = {}
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN"]
        base_date = datetime.now() - timedelta(days=365)
        
        for symbol in symbols:
            data_list = []
            price = 100.0
            
            for i in range(365):
                # Add some volatility and trend
                daily_return = np.random.normal(0.0008, 0.02)
                price *= (1 + daily_return)
                
                data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i),
                    open=price * 0.99,
                    high=price * 1.01,
                    low=price * 0.98,
                    close=price,
                    volume=1000000,
                    vwap=price
                ))
            
            data[symbol] = data_list
        
        return data
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.device == "cpu"
        assert engine.confidence_level == 0.95
        assert engine.monte_carlo_simulations == 1000
        assert engine.enable_ml_detection is False
        assert engine.data_processor is not None
        assert len(engine.stress_scenarios) > 0
    
    def test_engine_gpu_initialization(self):
        """Test engine with GPU initialization"""
        if torch.cuda.is_available():
            engine = RiskAssessmentEngine(
                device="cuda",
                enable_ml_detection=True
            )
            
            assert engine.device == "cuda"
            assert engine.tensor_optimizer is not None
            assert engine.anomaly_detector is not None
    
    def test_initialize_stress_scenarios(self, engine):
        """Test stress scenario initialization"""
        scenarios = engine._initialize_stress_scenarios()
        
        assert len(scenarios) > 0
        
        # Check scenario structure
        for scenario in scenarios:
            assert "name" in scenario
            assert "description" in scenario
            assert "parameters" in scenario
            
            # Check common parameters
            params = scenario["parameters"]
            assert isinstance(params, dict)
    
    @pytest.mark.asyncio
    async def test_assess_portfolio_risk(self, engine, portfolio_state, market_data):
        """Test portfolio risk assessment"""
        risk_profile = await engine.assess_portfolio_risk(
            portfolio_state,
            market_data,
            lookback_period=252
        )
        
        assert isinstance(risk_profile, RiskProfile)
        assert risk_profile.portfolio_var < 0  # VaR should be negative
        assert risk_profile.portfolio_cvar < risk_profile.portfolio_var  # CVaR worse than VaR
        assert risk_profile.max_drawdown < 0
        assert risk_profile.volatility > 0
        assert risk_profile.beta > 0
        assert 0 <= risk_profile.correlation_risk <= 1
        assert 0 <= risk_profile.concentration_risk <= 1
        assert risk_profile.liquidity_risk >= 0
        assert risk_profile.tail_risk >= 0
        assert 0 <= risk_profile.black_swan_probability <= 1
        assert risk_profile.risk_rating in ["low", "medium", "high", "extreme"]
    
    def test_prepare_returns_data(self, engine, market_data):
        """Test returns data preparation"""
        returns_data = engine._prepare_returns_data(market_data, 252)
        
        assert len(returns_data) == len(market_data)
        
        for symbol, returns in returns_data.items():
            assert isinstance(returns, np.ndarray)
            assert len(returns) == 251  # 252 prices -> 251 returns
            assert not np.any(np.isnan(returns))
    
    @pytest.mark.asyncio
    async def test_gpu_risk_assessment(self, portfolio_state, market_data):
        """Test GPU-accelerated risk assessment"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        engine = RiskAssessmentEngine(device="cuda")
        returns_data = engine._prepare_returns_data(market_data, 252)
        
        risk_metrics = await engine._gpu_risk_assessment(
            portfolio_state,
            returns_data
        )
        
        assert isinstance(risk_metrics, dict)
        assert "var" in risk_metrics
        assert "cvar" in risk_metrics
        assert "max_drawdown" in risk_metrics
        assert "volatility" in risk_metrics
        assert "beta" in risk_metrics
        assert "correlation_risk" in risk_metrics
        assert "concentration_risk" in risk_metrics
        assert "liquidity_risk" in risk_metrics
        assert "tail_risk" in risk_metrics
        assert "black_swan_probability" in risk_metrics
    
    def test_calculate_var_gpu(self, engine):
        """Test VaR calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        returns = torch.randn(1000, device="cuda") * 0.02 - 0.001
        var = engine._calculate_var_gpu(returns, 0.95)
        
        assert isinstance(var, torch.Tensor)
        assert var.device.type == "cuda"
        assert var.item() < 0  # VaR should be negative for losses
    
    def test_calculate_cvar_gpu(self, engine):
        """Test CVaR calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        returns = torch.randn(1000, device="cuda") * 0.02 - 0.001
        var = torch.quantile(returns, 0.05)
        cvar = engine._calculate_cvar_gpu(returns, var)
        
        assert isinstance(cvar, torch.Tensor)
        assert cvar.device.type == "cuda"
        assert cvar.item() <= var.item()  # CVaR should be worse than VaR
    
    def test_calculate_max_drawdown_gpu(self, engine):
        """Test maximum drawdown calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Create returns with known drawdown
        returns = torch.cat([
            torch.ones(50) * 0.01,   # Up 1% daily
            torch.ones(30) * -0.02,  # Down 2% daily
            torch.ones(20) * 0.005   # Up 0.5% daily
        ], device="cuda")
        
        max_dd = engine._calculate_max_drawdown_gpu(returns)
        
        assert isinstance(max_dd, torch.Tensor)
        assert max_dd.item() < 0  # Drawdown should be negative
    
    @pytest.mark.asyncio
    async def test_calculate_tail_risk_gpu(self, engine):
        """Test tail risk calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Generate returns with fat tails
        returns = torch.cat([
            torch.randn(950, device="cuda") * 0.01,  # Normal returns
            torch.randn(50, device="cuda") * 0.05    # Extreme returns
        ])
        
        tail_risk = await engine._calculate_tail_risk_gpu(returns)
        
        assert isinstance(tail_risk, float)
        assert tail_risk > 0
    
    @pytest.mark.asyncio
    async def test_estimate_black_swan_probability_gpu(self, engine):
        """Test black swan probability estimation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Normal returns
        normal_returns = torch.randn(1000, device="cuda") * 0.02
        normal_prob = await engine._estimate_black_swan_probability_gpu(normal_returns)
        
        # Returns with extreme events
        extreme_returns = torch.cat([
            torch.randn(950, device="cuda") * 0.02,
            torch.randn(50, device="cuda") * 0.10  # 5 sigma events
        ])
        extreme_prob = await engine._estimate_black_swan_probability_gpu(extreme_returns)
        
        assert 0 <= normal_prob <= 0.1
        assert 0 <= extreme_prob <= 0.1
        assert extreme_prob > normal_prob  # Should detect higher probability
    
    def test_calculate_kurtosis_gpu(self, engine):
        """Test kurtosis calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        # Normal distribution (kurtosis ≈ 3)
        normal_returns = torch.randn(10000, device="cuda")
        normal_kurtosis = engine._calculate_kurtosis_gpu(normal_returns)
        
        # Leptokurtic distribution (fat tails)
        fat_tail_returns = torch.cat([
            torch.randn(9000, device="cuda") * 0.5,
            torch.randn(1000, device="cuda") * 2.0
        ])
        fat_kurtosis = engine._calculate_kurtosis_gpu(fat_tail_returns)
        
        assert 2.5 < normal_kurtosis < 3.5  # Close to 3
        assert fat_kurtosis > 3.5  # Excess kurtosis
    
    @pytest.mark.asyncio
    async def test_cpu_risk_assessment(self, engine, portfolio_state, market_data):
        """Test CPU fallback for risk assessment"""
        engine.device = "cpu"
        returns_data = engine._prepare_returns_data(market_data, 252)
        
        risk_metrics = await engine._cpu_risk_assessment(
            portfolio_state,
            returns_data
        )
        
        assert isinstance(risk_metrics, dict)
        assert all(key in risk_metrics for key in [
            "var", "cvar", "max_drawdown", "volatility", "beta",
            "correlation_risk", "concentration_risk", "liquidity_risk",
            "tail_risk", "black_swan_probability"
        ])
    
    @pytest.mark.asyncio
    async def test_detect_anomalies(self, engine, market_data):
        """Test anomaly detection"""
        engine.enable_ml_detection = True
        engine.anomaly_detector = Mock()
        engine.anomaly_detector.fit = Mock()
        engine.anomaly_detector.decision_function = Mock(
            return_value=np.array([-1, 1, 1, -1, 1])  # Mix of anomalies
        )
        
        returns_data = engine._prepare_returns_data(market_data, 252)
        anomaly_score = await engine._detect_anomalies(returns_data)
        
        assert 0 <= anomaly_score <= 1
        assert engine.anomaly_detector.fit.called
        assert engine.anomaly_detector.decision_function.called
    
    def test_calculate_risk_rating(self, engine):
        """Test risk rating calculation"""
        # Low risk metrics
        low_risk_metrics = {
            "var": -0.02,
            "volatility": 0.10,
            "concentration_risk": 0.15,
            "tail_risk": 0.05,
            "black_swan_probability": 0.001
        }
        
        rating = engine._calculate_risk_rating(low_risk_metrics)
        assert rating == "low"
        
        # High risk metrics
        high_risk_metrics = {
            "var": -0.15,
            "volatility": 0.35,
            "concentration_risk": 0.60,
            "tail_risk": 0.25,
            "black_swan_probability": 0.05
        }
        
        rating = engine._calculate_risk_rating(high_risk_metrics)
        assert rating in ["high", "extreme"]
    
    @pytest.mark.asyncio
    async def test_run_stress_tests(self, engine, portfolio_state, market_data):
        """Test stress testing"""
        stress_results = await engine.run_stress_tests(
            portfolio_state,
            market_data
        )
        
        assert len(stress_results) == len(engine.stress_scenarios)
        
        for result in stress_results:
            assert isinstance(result, StressTestResult)
            assert result.scenario_name in [s["name"] for s in engine.stress_scenarios]
            assert result.portfolio_impact <= 0  # Should be negative
            assert result.worst_case_loss <= result.portfolio_impact
            assert result.recovery_time > 0
            assert isinstance(result.affected_assets, list)
            assert isinstance(result.recommendations, list)
    
    @pytest.mark.asyncio
    async def test_run_single_stress_test(self, engine, portfolio_state, market_data):
        """Test single stress test scenario"""
        scenario = {
            "name": "Market Crash",
            "parameters": {
                "market_drop": -0.40,
                "volatility_spike": 3.0,
                "duration_days": 90
            }
        }
        
        result = await engine._run_single_stress_test(
            portfolio_state,
            market_data,
            scenario
        )
        
        assert isinstance(result, StressTestResult)
        assert result.scenario_name == "Market Crash"
        assert result.portfolio_impact < 0
        assert result.worst_case_loss < 0
        assert len(result.recommendations) > 0
    
    @pytest.mark.asyncio
    async def test_gpu_stress_simulation(self, portfolio_state, market_data):
        """Test GPU stress simulation"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        engine = RiskAssessmentEngine(device="cuda")
        
        parameters = {
            "market_drop": -0.30,
            "volatility_spike": 2.5
        }
        
        impact_results = await engine._gpu_stress_simulation(
            portfolio_state,
            market_data,
            parameters
        )
        
        assert "portfolio_impact" in impact_results
        assert "worst_case_loss" in impact_results
        assert "recovery_time" in impact_results
        assert "affected_assets" in impact_results
        
        assert impact_results["portfolio_impact"] < 0
        assert impact_results["worst_case_loss"] < impact_results["portfolio_impact"]
    
    @pytest.mark.asyncio
    async def test_cpu_stress_simulation(self, engine, portfolio_state, market_data):
        """Test CPU stress simulation"""
        parameters = {
            "market_drop": -0.25,
            "duration_days": 60
        }
        
        impact_results = await engine._cpu_stress_simulation(
            portfolio_state,
            market_data,
            parameters
        )
        
        assert "portfolio_impact" in impact_results
        assert "worst_case_loss" in impact_results
        assert "recovery_time" in impact_results
        assert "affected_assets" in impact_results
    
    def test_generate_stress_recommendations(self, engine):
        """Test stress test recommendation generation"""
        # Market crash scenario
        recommendations = engine._generate_stress_recommendations(
            "Market Crash",
            {
                "portfolio_impact": -0.35,
                "worst_case_loss": -0.50,
                "recovery_time": 365
            }
        )
        
        assert len(recommendations) > 0
        assert any("hedge" in r.lower() or "protection" in r.lower() for r in recommendations)
        
        # Liquidity crisis scenario
        recommendations = engine._generate_stress_recommendations(
            "Liquidity Crisis",
            {
                "portfolio_impact": -0.20,
                "worst_case_loss": -0.30,
                "recovery_time": 90
            }
        )
        
        assert any("cash" in r.lower() or "liquid" in r.lower() for r in recommendations)
    
    @pytest.mark.asyncio
    async def test_generate_mitigation_strategies(self, engine, portfolio_state):
        """Test mitigation strategy generation"""
        risk_profile = RiskProfile(
            portfolio_var=-0.12,
            portfolio_cvar=-0.18,
            max_drawdown=-0.25,
            volatility=0.22,
            beta=1.4,
            correlation_risk=0.75,
            concentration_risk=0.45,
            liquidity_risk=0.30,
            tail_risk=0.20,
            black_swan_probability=0.03,
            risk_rating="high"
        )
        
        strategies = await engine.generate_mitigation_strategies(
            risk_profile,
            portfolio_state
        )
        
        assert len(strategies) > 0
        
        for strategy in strategies:
            assert isinstance(strategy, RiskMitigation)
            assert strategy.strategy_type in [
                "var_reduction", "concentration_reduction",
                "correlation_reduction", "tail_risk_hedging"
            ]
            assert len(strategy.suggested_actions) > 0
            assert strategy.expected_risk_reduction > 0
            assert strategy.implementation_cost >= 0
    
    @pytest.mark.asyncio
    async def test_create_var_mitigation(self, engine, portfolio_state):
        """Test VaR mitigation strategy creation"""
        risk_profile = RiskProfile(
            portfolio_var=-0.15,
            portfolio_cvar=-0.20,
            max_drawdown=-0.30,
            volatility=0.25,
            beta=1.2,
            correlation_risk=0.60,
            concentration_risk=0.35,
            liquidity_risk=0.25,
            tail_risk=0.18,
            black_swan_probability=0.02,
            risk_rating="high"
        )
        
        mitigation = await engine._create_var_mitigation(
            risk_profile,
            portfolio_state
        )
        
        assert isinstance(mitigation, RiskMitigation)
        assert mitigation.strategy_type == "var_reduction"
        assert mitigation.target_risk_level == 0.05
        assert len(mitigation.suggested_actions) > 0
        assert mitigation.expected_risk_reduction > 0
    
    @pytest.mark.asyncio
    async def test_create_concentration_mitigation(self, engine, portfolio_state):
        """Test concentration risk mitigation"""
        risk_profile = RiskProfile(
            portfolio_var=-0.08,
            portfolio_cvar=-0.12,
            max_drawdown=-0.20,
            volatility=0.18,
            beta=1.0,
            correlation_risk=0.50,
            concentration_risk=0.55,  # High concentration
            liquidity_risk=0.20,
            tail_risk=0.10,
            black_swan_probability=0.01,
            risk_rating="medium"
        )
        
        # Set concentrated portfolio
        portfolio_state.holdings = {"AAPL": 500, "GOOGL": 10}  # Highly concentrated
        
        mitigation = await engine._create_concentration_mitigation(
            risk_profile,
            portfolio_state
        )
        
        assert mitigation.strategy_type == "concentration_reduction"
        assert len(mitigation.suggested_actions) > 0
        assert any(action["action"] == "rebalance" for action in mitigation.suggested_actions)
    
    def test_calculate_risk_adjusted_returns(self, engine):
        """Test risk-adjusted return calculations"""
        returns = np.random.normal(0.0008, 0.02, 252)
        
        metrics = engine.calculate_risk_adjusted_returns(returns, risk_free_rate=0.03)
        
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "calmar_ratio" in metrics
        assert "information_ratio" in metrics
        
        # Check reasonable values
        assert -5 < metrics["sharpe_ratio"] < 5
        assert -5 < metrics["sortino_ratio"] < 5
        assert metrics["calmar_ratio"] != 0
    
    def test_calculate_max_drawdown_cpu(self, engine):
        """Test CPU maximum drawdown calculation"""
        # Create returns with known drawdown pattern
        returns = np.concatenate([
            np.ones(50) * 0.02,    # Up 2% daily
            np.ones(30) * -0.03,   # Down 3% daily  
            np.ones(20) * 0.01     # Up 1% daily
        ])
        
        max_dd = engine._calculate_max_drawdown_cpu(returns)
        
        assert max_dd < 0  # Should be negative
        assert -1 < max_dd < 0  # Reasonable range


class TestIntegration:
    """Integration tests for Risk Assessment Engine"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_risk_assessment_workflow(self):
        """Test complete risk assessment workflow"""
        engine = RiskAssessmentEngine(
            device="cpu",
            confidence_level=0.95,
            monte_carlo_simulations=500  # Reduced for faster tests
        )
        
        # Create realistic portfolio
        portfolio_state = FinancialState(
            portfolio_value=1000000.0,
            holdings={
                "AAPL": 1000,
                "GOOGL": 200,
                "MSFT": 500,
                "AMZN": 150,
                "JPM": 800,
                "GLD": 300,  # Gold ETF for diversification
                "TLT": 400   # Bond ETF
            },
            cash_balance=50000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={
                "volatility": 0.20,
                "vix": 22.5,
                "interest_rate": 0.045
            },
            timestamp=datetime.now()
        )
        
        # Generate market data with realistic patterns
        market_data = {}
        base_date = datetime.now() - timedelta(days=500)
        
        # Different asset characteristics
        asset_params = {
            "AAPL": {"trend": 0.0008, "volatility": 0.022},
            "GOOGL": {"trend": 0.0010, "volatility": 0.025},
            "MSFT": {"trend": 0.0009, "volatility": 0.020},
            "AMZN": {"trend": 0.0012, "volatility": 0.028},
            "JPM": {"trend": 0.0006, "volatility": 0.024},
            "GLD": {"trend": 0.0003, "volatility": 0.015},
            "TLT": {"trend": 0.0002, "volatility": 0.010}
        }
        
        for symbol, params in asset_params.items():
            data_list = []
            price = 100.0
            
            for i in range(500):
                daily_return = np.random.normal(params["trend"], params["volatility"])
                
                # Add some correlation during market stress
                if 200 <= i <= 250:  # Stress period
                    daily_return -= 0.01  # Negative bias
                    daily_return *= 1.5   # Higher volatility
                
                price *= (1 + daily_return)
                
                data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i),
                    open=price * 0.98,
                    high=price * 1.02,
                    low=price * 0.97,
                    close=price,
                    volume=np.random.randint(500000, 2000000),
                    vwap=price * 0.995
                ))
            
            market_data[symbol] = data_list
        
        # 1. Assess portfolio risk
        risk_profile = await engine.assess_portfolio_risk(
            portfolio_state,
            market_data,
            lookback_period=365
        )
        
        print(f"\nPortfolio Risk Profile:")
        print(f"  VaR (95%): {risk_profile.portfolio_var:.3f}")
        print(f"  CVaR (95%): {risk_profile.portfolio_cvar:.3f}")
        print(f"  Volatility: {risk_profile.volatility:.3f}")
        print(f"  Max Drawdown: {risk_profile.max_drawdown:.3f}")
        print(f"  Risk Rating: {risk_profile.risk_rating}")
        
        # 2. Run stress tests
        stress_results = await engine.run_stress_tests(
            portfolio_state,
            market_data
        )
        
        print(f"\nStress Test Results:")
        for result in stress_results:
            print(f"  {result.scenario_name}:")
            print(f"    Impact: {result.portfolio_impact:.3f}")
            print(f"    Worst Case: {result.worst_case_loss:.3f}")
            print(f"    Recovery: {result.recovery_time} days")
        
        # 3. Generate mitigation strategies
        mitigations = await engine.generate_mitigation_strategies(
            risk_profile,
            portfolio_state
        )
        
        print(f"\nMitigation Strategies:")
        for mitigation in mitigations:
            print(f"  {mitigation.strategy_type}:")
            print(f"    Expected Risk Reduction: {mitigation.expected_risk_reduction:.2%}")
            print(f"    Implementation Cost: ${mitigation.implementation_cost:,.0f}")
            print(f"    Actions: {len(mitigation.suggested_actions)}")
        
        # Verify results
        assert isinstance(risk_profile, RiskProfile)
        assert len(stress_results) > 0
        assert len(mitigations) > 0
        
        # Risk profile should reflect portfolio characteristics
        assert risk_profile.concentration_risk < 0.4  # Well diversified
        assert risk_profile.correlation_risk > 0.3  # Some correlation
        assert risk_profile.risk_rating in ["medium", "high"]
        
        # Stress tests should show vulnerability
        market_crash = next(r for r in stress_results if r.scenario_name == "Market Crash")
        assert market_crash.portfolio_impact < -0.15  # Significant impact
        assert market_crash.recovery_time > 60  # Takes time to recover
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_gpu_vs_cpu_performance(self):
        """Compare GPU vs CPU performance for risk assessment"""
        # Create large portfolio
        symbols = [f"STOCK_{i}" for i in range(50)]
        holdings = {symbol: np.random.randint(100, 1000) for symbol in symbols}
        
        portfolio_state = FinancialState(
            portfolio_value=5000000.0,
            holdings=holdings,
            cash_balance=100000.0,
            risk_tolerance=0.6,
            time_horizon=252,
            market_conditions={"volatility": 0.18},
            timestamp=datetime.now()
        )
        
        # Generate large market dataset
        market_data = {}
        for symbol in symbols:
            data_list = []
            price = 100.0
            
            for i in range(500):
                price *= (1 + np.random.normal(0.0008, 0.02))
                data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.now() - timedelta(days=500-i),
                    open=price,
                    high=price * 1.01,
                    low=price * 0.99,
                    close=price,
                    volume=1000000,
                    vwap=price
                ))
            
            market_data[symbol] = data_list
        
        # CPU assessment
        cpu_engine = RiskAssessmentEngine(
            device="cpu",
            monte_carlo_simulations=1000
        )
        
        import time
        cpu_start = time.time()
        cpu_profile = await cpu_engine.assess_portfolio_risk(
            portfolio_state,
            market_data,
            lookback_period=252
        )
        cpu_time = time.time() - cpu_start
        
        # GPU assessment
        gpu_engine = RiskAssessmentEngine(
            device="cuda",
            monte_carlo_simulations=1000
        )
        
        gpu_start = time.time()
        gpu_profile = await gpu_engine.assess_portfolio_risk(
            portfolio_state,
            market_data,
            lookback_period=252
        )
        gpu_time = time.time() - gpu_start
        
        print(f"\nPerformance Comparison:")
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Results should be similar
        assert abs(cpu_profile.portfolio_var - gpu_profile.portfolio_var) < 0.01
        assert abs(cpu_profile.volatility - gpu_profile.volatility) < 0.01
        assert cpu_profile.risk_rating == gpu_profile.risk_rating
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extreme_scenarios(self):
        """Test risk assessment under extreme market conditions"""
        engine = RiskAssessmentEngine(device="cpu")
        
        # Highly leveraged portfolio
        portfolio_state = FinancialState(
            portfolio_value=100000.0,
            holdings={
                "TSLA": 200,  # High volatility
                "GME": 500,   # Meme stock
                "AMC": 1000,  # High volatility
                "COIN": 150   # Crypto-related
            },
            cash_balance=-50000.0,  # Negative cash (margin)
            risk_tolerance=0.9,  # High risk tolerance
            time_horizon=30,  # Short term
            market_conditions={"volatility": 0.40},
            timestamp=datetime.now()
        )
        
        # Generate volatile market data
        market_data = {}
        base_date = datetime.now() - timedelta(days=365)
        
        for symbol in portfolio_state.holdings.keys():
            data_list = []
            price = 100.0
            
            for i in range(365):
                # Extreme volatility
                daily_return = np.random.normal(0, 0.05)  # 5% daily vol
                
                # Add jumps
                if np.random.random() < 0.01:  # 1% chance of jump
                    daily_return += np.random.choice([-0.20, 0.30])
                
                price *= (1 + daily_return)
                price = max(price, 5.0)  # Prevent going to zero
                
                data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=price * 1.10,
                    low=price * 0.90,
                    close=price,
                    volume=np.random.randint(10000000, 50000000),
                    vwap=price
                ))
            
            market_data[symbol] = data_list
        
        # Assess risk
        risk_profile = await engine.assess_portfolio_risk(
            portfolio_state,
            market_data,
            lookback_period=252
        )
        
        # Should identify extreme risk
        assert risk_profile.risk_rating == "extreme"
        assert risk_profile.volatility > 0.30
        assert risk_profile.portfolio_var < -0.20
        assert risk_profile.max_drawdown < -0.50
        assert risk_profile.black_swan_probability > 0.05
        
        # Stress tests
        stress_results = await engine.run_stress_tests(
            portfolio_state,
            market_data
        )
        
        # Should show catastrophic impacts
        market_crash = next(r for r in stress_results if r.scenario_name == "Market Crash")
        assert market_crash.worst_case_loss < -0.70  # Potential wipeout
        
        # Mitigation should be aggressive
        mitigations = await engine.generate_mitigation_strategies(
            risk_profile,
            portfolio_state
        )
        
        assert len(mitigations) > 3  # Multiple strategies needed
        assert any(m.strategy_type == "tail_risk_hedging" for m in mitigations)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
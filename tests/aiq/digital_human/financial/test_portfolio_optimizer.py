"""
Comprehensive tests for Portfolio Optimizer

Tests modern portfolio theory implementation, optimization algorithms,
and GPU acceleration for portfolio management.
"""

import pytest
import numpy as np
import torch
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock

from aiq.digital_human.financial.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult
)


class TestOptimizationConstraints:
    """Test OptimizationConstraints dataclass"""
    
    def test_constraints_creation(self):
        """Test creating optimization constraints"""
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.40,
            target_return=0.08,
            max_risk=0.15,
            sector_limits={"technology": 0.30, "finance": 0.25},
            concentration_limit=0.35
        )
        
        assert constraints.min_weight == 0.05
        assert constraints.max_weight == 0.40
        assert constraints.target_return == 0.08
        assert constraints.max_risk == 0.15
        assert constraints.sector_limits["technology"] == 0.30
        assert constraints.concentration_limit == 0.35
    
    def test_constraints_defaults(self):
        """Test default constraint values"""
        constraints = OptimizationConstraints()
        
        assert constraints.min_weight == 0.0
        assert constraints.max_weight == 1.0
        assert constraints.target_return is None
        assert constraints.max_risk is None
        assert constraints.sector_limits is None
        assert constraints.concentration_limit == 0.4


class TestOptimizationResult:
    """Test OptimizationResult dataclass"""
    
    def test_result_creation(self):
        """Test creating optimization result"""
        result = OptimizationResult(
            weights={"AAPL": 0.3, "GOOGL": 0.4, "MSFT": 0.3},
            expected_return=0.09,
            portfolio_risk=0.12,
            sharpe_ratio=0.85,
            efficient_frontier=[(0.08, 0.05), (0.10, 0.07), (0.12, 0.10)],
            risk_contributions={"AAPL": 0.25, "GOOGL": 0.45, "MSFT": 0.30}
        )
        
        assert result.weights["AAPL"] == 0.3
        assert result.expected_return == 0.09
        assert result.portfolio_risk == 0.12
        assert result.sharpe_ratio == 0.85
        assert len(result.efficient_frontier) == 3
        assert result.risk_contributions["GOOGL"] == 0.45


class TestPortfolioOptimizer:
    """Test Portfolio Optimizer implementation"""
    
    @pytest.fixture
    def optimizer(self):
        """Create optimizer instance"""
        return PortfolioOptimizer(
            device="cpu",  # Use CPU for tests
            risk_free_rate=0.03,
            enable_gpu=False
        )
    
    @pytest.fixture
    def assets(self):
        """Create test asset list"""
        return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
    
    @pytest.fixture
    def historical_returns(self):
        """Create test historical returns data"""
        np.random.seed(42)
        # 252 trading days x 5 assets
        returns = np.random.normal(0.0008, 0.02, (252, 5))
        return returns
    
    @pytest.fixture
    def current_prices(self):
        """Create test current prices"""
        return {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 390.0,
            "AMZN": 170.0,
            "TSLA": 250.0
        }
    
    @pytest.fixture
    def constraints(self):
        """Create test constraints"""
        return OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.40,
            target_return=0.08,
            concentration_limit=0.35
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.device == "cpu"
        assert optimizer.risk_free_rate == 0.03
        assert optimizer.tensor_optimizer is None
    
    def test_optimizer_gpu_initialization(self):
        """Test optimizer with GPU initialization"""
        if torch.cuda.is_available():
            optimizer = PortfolioOptimizer(
                device="cuda",
                enable_gpu=True
            )
            
            assert optimizer.device == "cuda"
            assert optimizer.tensor_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_optimize_portfolio_cpu(
        self, optimizer, assets, historical_returns, current_prices, constraints
    ):
        """Test portfolio optimization on CPU"""
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # Verify result structure
        assert isinstance(result, OptimizationResult)
        assert set(result.weights.keys()) == set(assets)
        assert all(0 <= w <= 1 for w in result.weights.values())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
        assert result.expected_return > 0
        assert result.portfolio_risk > 0
        assert result.sharpe_ratio > -10
        assert len(result.efficient_frontier) > 0
        assert set(result.risk_contributions.keys()) == set(assets)
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_optimize_portfolio_gpu(
        self, assets, historical_returns, current_prices, constraints
    ):
        """Test portfolio optimization on GPU"""
        optimizer = PortfolioOptimizer(
            device="cuda",
            enable_gpu=True
        )
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # Verify GPU calculation
        assert isinstance(result, OptimizationResult)
        assert all(0 <= w <= 1 for w in result.weights.values())
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6
    
    @pytest.mark.asyncio
    async def test_mean_variance_optimization(
        self, optimizer, assets, historical_returns, current_prices, constraints
    ):
        """Test mean-variance optimization method"""
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # Check constraint satisfaction
        assert all(constraints.min_weight <= w <= constraints.max_weight 
                  for w in result.weights.values())
        assert max(result.weights.values()) <= constraints.concentration_limit
        
        # Target return should be approximately met
        if constraints.target_return:
            assert abs(result.expected_return - constraints.target_return) < 0.02
    
    @pytest.mark.asyncio
    async def test_maximum_sharpe_optimization(
        self, optimizer, assets, historical_returns, current_prices
    ):
        """Test maximum Sharpe ratio optimization"""
        constraints = OptimizationConstraints()  # Default constraints
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="maximum_sharpe"
        )
        
        # Should maximize Sharpe ratio
        assert result.sharpe_ratio > 0
        assert all(0 <= w <= 1 for w in result.weights.values())
    
    @pytest.mark.asyncio
    async def test_minimum_variance_optimization(
        self, optimizer, assets, historical_returns, current_prices
    ):
        """Test minimum variance optimization"""
        constraints = OptimizationConstraints()
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="minimum_variance"
        )
        
        # Should minimize risk
        assert result.portfolio_risk > 0
        assert result.portfolio_risk < 0.5  # Reasonable risk level
    
    @pytest.mark.asyncio
    async def test_risk_parity_optimization(
        self, optimizer, assets, historical_returns, current_prices
    ):
        """Test risk parity optimization"""
        constraints = OptimizationConstraints()
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="risk_parity"
        )
        
        # Risk contributions should be approximately equal
        risk_contributions = list(result.risk_contributions.values())
        mean_contribution = np.mean(risk_contributions)
        
        # Allow some deviation
        assert all(abs(c - mean_contribution) < 0.1 for c in risk_contributions)
    
    @pytest.mark.asyncio
    async def test_unknown_optimization_method(
        self, optimizer, assets, historical_returns, current_prices, constraints
    ):
        """Test with unknown optimization method"""
        with pytest.raises(ValueError, match="Unknown optimization method"):
            await optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=historical_returns,
                current_prices=current_prices,
                constraints=constraints,
                optimization_method="invalid_method"
            )
    
    @pytest.mark.asyncio
    async def test_efficient_frontier_calculation(
        self, optimizer, assets, historical_returns, current_prices
    ):
        """Test efficient frontier calculation"""
        constraints = OptimizationConstraints()
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        frontier = result.efficient_frontier
        
        # Verify frontier properties
        assert len(frontier) > 10
        
        # Should be ordered by risk
        risks = [point[0] for point in frontier]
        returns = [point[1] for point in frontier]
        
        # Generally, higher risk should correspond to higher return
        # (though not strictly monotonic due to optimization)
        assert min(risks) < max(risks)
        assert min(returns) < max(returns)
    
    def test_calculate_portfolio_metrics(
        self, optimizer, assets, historical_returns
    ):
        """Test portfolio metrics calculation"""
        weights = {
            "AAPL": 0.2,
            "GOOGL": 0.2,
            "MSFT": 0.2,
            "AMZN": 0.2,
            "TSLA": 0.2
        }
        
        metrics = optimizer.calculate_portfolio_metrics(
            weights=weights,
            returns=historical_returns,
            asset_names=assets
        )
        
        # Verify metrics
        assert "expected_return" in metrics
        assert "volatility" in metrics
        assert "sharpe_ratio" in metrics
        assert "sortino_ratio" in metrics
        assert "max_drawdown" in metrics
        assert "var_95" in metrics
        assert "cvar_95" in metrics
        
        # Check reasonable values
        assert -1 < metrics["expected_return"] < 1
        assert 0 < metrics["volatility"] < 1
        assert -5 < metrics["sharpe_ratio"] < 5
        assert -1 < metrics["max_drawdown"] <= 0
    
    def test_rebalance_portfolio(self, optimizer, current_prices):
        """Test portfolio rebalancing calculation"""
        current_weights = {
            "AAPL": 0.30,
            "GOOGL": 0.25,
            "MSFT": 0.20,
            "AMZN": 0.15,
            "TSLA": 0.10
        }
        
        target_weights = {
            "AAPL": 0.20,
            "GOOGL": 0.20,
            "MSFT": 0.20,
            "AMZN": 0.20,
            "TSLA": 0.20
        }
        
        portfolio_value = 100000.0
        
        trades = optimizer.rebalance_portfolio(
            current_weights=current_weights,
            target_weights=target_weights,
            current_prices=current_prices,
            portfolio_value=portfolio_value,
            transaction_costs=0.001
        )
        
        # Verify trades
        assert isinstance(trades, dict)
        
        # Should generate trades for significant differences
        assert "AAPL" in trades  # Need to sell (0.30 -> 0.20)
        assert trades["AAPL"]["action"] == "sell"
        
        assert "TSLA" in trades  # Need to buy (0.10 -> 0.20)
        assert trades["TSLA"]["action"] == "buy"
        
        # All trades should have required fields
        for asset, trade in trades.items():
            assert "action" in trade
            assert "shares" in trade
            assert "value" in trade
            assert "cost" in trade
            assert trade["cost"] > 0  # Transaction cost
    
    @pytest.mark.asyncio
    async def test_gpu_optimization_methods(self, assets, historical_returns, current_prices):
        """Test GPU-specific optimization methods"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        optimizer = PortfolioOptimizer(device="cuda", enable_gpu=True)
        
        expected_returns = torch.tensor(
            np.mean(historical_returns, axis=0),
            device="cuda",
            dtype=torch.float32
        )
        
        covariance_matrix = torch.tensor(
            np.cov(historical_returns.T),
            device="cuda",
            dtype=torch.float32
        )
        
        constraints = OptimizationConstraints(
            min_weight=0.05,
            max_weight=0.30,
            target_return=0.08
        )
        
        # Test mean-variance on GPU
        weights = await optimizer._mean_variance_optimization_gpu(
            expected_returns,
            covariance_matrix,
            constraints
        )
        
        assert weights.device.type == "cuda"
        assert len(weights) == len(assets)
        assert abs(torch.sum(weights).item() - 1.0) < 1e-5
        
        # Test maximum Sharpe on GPU
        weights = await optimizer._max_sharpe_optimization_gpu(
            expected_returns,
            covariance_matrix,
            constraints
        )
        
        assert weights.device.type == "cuda"
        assert torch.all(weights >= 0)
        
        # Test minimum variance on GPU
        weights = await optimizer._min_variance_optimization_gpu(
            covariance_matrix,
            constraints
        )
        
        assert weights.device.type == "cuda"
        assert torch.all(weights >= constraints.min_weight)
        assert torch.all(weights <= constraints.max_weight)
        
        # Test risk parity on GPU
        weights = await optimizer._risk_parity_optimization_gpu(
            covariance_matrix,
            constraints
        )
        
        assert weights.device.type == "cuda"
        assert abs(torch.sum(weights).item() - 1.0) < 1e-5
    
    def test_calculate_risk_contributions_gpu(self, assets, historical_returns):
        """Test risk contribution calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        optimizer = PortfolioOptimizer(device="cuda", enable_gpu=True)
        
        weights = torch.tensor(
            [0.2, 0.2, 0.2, 0.2, 0.2],
            device="cuda",
            dtype=torch.float32
        )
        
        covariance_matrix = torch.tensor(
            np.cov(historical_returns.T),
            device="cuda",
            dtype=torch.float32
        )
        
        risk_contributions = optimizer._calculate_risk_contributions_gpu(
            weights,
            covariance_matrix
        )
        
        assert risk_contributions.device.type == "cuda"
        assert len(risk_contributions) == len(assets)
        assert abs(torch.sum(risk_contributions).item() - 1.0) < 1e-5
    
    @pytest.mark.asyncio
    async def test_constraints_validation(
        self, optimizer, assets, historical_returns, current_prices
    ):
        """Test constraint validation in optimization"""
        # Test with infeasible constraints
        constraints = OptimizationConstraints(
            min_weight=0.25,  # Forces at least 25% in each asset
            max_weight=0.30,  # But allows max 30%
            # With 5 assets, this is infeasible (5 * 0.25 = 1.25 > 1.0)
        )
        
        # Should handle gracefully
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # Should return some valid portfolio
        assert abs(sum(result.weights.values()) - 1.0) < 1e-5
    
    @pytest.mark.asyncio
    async def test_portfolio_with_single_asset(
        self, optimizer, current_prices
    ):
        """Test optimization with single asset"""
        assets = ["AAPL"]
        returns = np.random.normal(0.0008, 0.02, (252, 1))
        constraints = OptimizationConstraints()
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=returns,
            current_prices={"AAPL": current_prices["AAPL"]},
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # With single asset, weight should be 1.0
        assert result.weights["AAPL"] == 1.0
        assert result.portfolio_risk > 0
    
    @pytest.mark.asyncio
    async def test_historical_data_validation(
        self, optimizer, assets, current_prices, constraints
    ):
        """Test validation of historical data"""
        # Test with too little data
        short_returns = np.random.normal(0.0008, 0.02, (10, 5))
        
        result = await optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=short_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        
        # Should still work with limited data
        assert abs(sum(result.weights.values()) - 1.0) < 1e-5
        
        # Test with mismatched dimensions
        wrong_shape_returns = np.random.normal(0.0008, 0.02, (252, 3))
        
        with pytest.raises(Exception):  # Should raise an error
            await optimizer.optimize_portfolio(
                assets=assets,  # 5 assets
                historical_returns=wrong_shape_returns,  # 3 columns
                current_prices=current_prices,
                constraints=constraints,
                optimization_method="mean_variance"
            )


class TestIntegration:
    """Integration tests for Portfolio Optimizer"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_optimization_workflow(self):
        """Test complete portfolio optimization workflow"""
        # Initialize optimizer
        optimizer = PortfolioOptimizer(
            device="cpu",
            risk_free_rate=0.03
        )
        
        # Real-world like data
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "BRK-B", "JPM", "JNJ", "XOM"]
        
        # Generate realistic returns
        np.random.seed(42)
        years = 3
        trading_days = 252 * years
        
        # Different expected returns and volatilities
        expected_returns = np.array([0.12, 0.15, 0.13, 0.18, 0.10, 0.11, 0.08, 0.09])
        volatilities = np.array([0.18, 0.22, 0.19, 0.25, 0.15, 0.20, 0.12, 0.23])
        
        # Generate correlated returns
        correlation_matrix = np.eye(len(assets))
        # Add some correlations
        correlation_matrix[0, 1] = correlation_matrix[1, 0] = 0.6  # AAPL-GOOGL
        correlation_matrix[0, 2] = correlation_matrix[2, 0] = 0.5  # AAPL-MSFT
        correlation_matrix[1, 2] = correlation_matrix[2, 1] = 0.7  # GOOGL-MSFT
        correlation_matrix[5, 6] = correlation_matrix[6, 5] = 0.4  # JPM-JNJ
        
        # Generate returns
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        annual_returns = np.random.multivariate_normal(
            expected_returns,
            cov_matrix,
            size=trading_days
        )
        daily_returns = annual_returns / 252
        
        current_prices = {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 390.0,
            "AMZN": 170.0,
            "BRK-B": 350.0,
            "JPM": 150.0,
            "JNJ": 160.0,
            "XOM": 110.0
        }
        
        # Test multiple optimization strategies
        strategies = {
            "conservative": OptimizationConstraints(
                min_weight=0.05,
                max_weight=0.20,
                max_risk=0.12,
                concentration_limit=0.20
            ),
            "balanced": OptimizationConstraints(
                min_weight=0.05,
                max_weight=0.30,
                target_return=0.10,
                concentration_limit=0.30
            ),
            "aggressive": OptimizationConstraints(
                min_weight=0.0,
                max_weight=0.40,
                target_return=0.15,
                concentration_limit=0.40
            )
        }
        
        results = {}
        
        for strategy_name, constraints in strategies.items():
            result = await optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=daily_returns,
                current_prices=current_prices,
                constraints=constraints,
                optimization_method="mean_variance"
            )
            
            results[strategy_name] = result
            
            # Verify results
            assert abs(sum(result.weights.values()) - 1.0) < 1e-5
            assert all(w >= constraints.min_weight for w in result.weights.values())
            assert all(w <= constraints.max_weight for w in result.weights.values())
            assert max(result.weights.values()) <= constraints.concentration_limit
            
            # Risk should increase with aggressiveness
            if strategy_name == "conservative":
                conservative_risk = result.portfolio_risk
            elif strategy_name == "aggressive":
                aggressive_risk = result.portfolio_risk
        
        # Aggressive should have higher risk
        assert aggressive_risk > conservative_risk
        
        # Test rebalancing
        current_weights = results["balanced"].weights
        target_weights = results["conservative"].weights
        
        trades = optimizer.rebalance_portfolio(
            current_weights=current_weights,
            target_weights=target_weights,
            current_prices=current_prices,
            portfolio_value=1000000.0,
            transaction_costs=0.001
        )
        
        # Should generate rebalancing trades
        assert len(trades) > 0
        
        # Calculate portfolio metrics
        balanced_portfolio = results["balanced"]
        metrics = optimizer.calculate_portfolio_metrics(
            weights=balanced_portfolio.weights,
            returns=daily_returns,
            asset_names=assets
        )
        
        # Verify comprehensive metrics
        assert metrics["sharpe_ratio"] > 0
        assert metrics["sortino_ratio"] > 0
        assert -1 < metrics["max_drawdown"] < 0
        assert metrics["var_95"] < 0
        assert metrics["cvar_95"] < metrics["var_95"]
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_gpu_performance_comparison(self):
        """Compare GPU vs CPU performance for large portfolios"""
        # Large portfolio
        num_assets = 100
        assets = [f"STOCK_{i}" for i in range(num_assets)]
        
        # Generate returns
        np.random.seed(42)
        historical_returns = np.random.normal(0.0008, 0.02, (252, num_assets))
        
        current_prices = {asset: np.random.uniform(50, 500) for asset in assets}
        
        constraints = OptimizationConstraints(
            min_weight=0.0,
            max_weight=0.10,
            concentration_limit=0.10
        )
        
        # CPU optimization
        cpu_optimizer = PortfolioOptimizer(device="cpu", enable_gpu=False)
        
        import time
        cpu_start = time.time()
        cpu_result = await cpu_optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        cpu_time = time.time() - cpu_start
        
        # GPU optimization
        gpu_optimizer = PortfolioOptimizer(device="cuda", enable_gpu=True)
        
        gpu_start = time.time()
        gpu_result = await gpu_optimizer.optimize_portfolio(
            assets=assets,
            historical_returns=historical_returns,
            current_prices=current_prices,
            constraints=constraints,
            optimization_method="mean_variance"
        )
        gpu_time = time.time() - gpu_start
        
        print(f"Large portfolio optimization - CPU: {cpu_time:.3f}s, GPU: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Results should be similar
        cpu_return = cpu_result.expected_return
        gpu_return = gpu_result.expected_return
        
        assert abs(cpu_return - gpu_return) < 0.01
        
        # Portfolio weights should be similar
        for asset in assets[:10]:  # Check first 10 assets
            cpu_weight = cpu_result.weights[asset]
            gpu_weight = gpu_result.weights[asset]
            assert abs(cpu_weight - gpu_weight) < 0.05
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_extreme_market_conditions(self):
        """Test optimization under extreme market conditions"""
        optimizer = PortfolioOptimizer(device="cpu")
        
        assets = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D"]
        
        # Extreme scenarios
        scenarios = {
            "high_volatility": {
                "returns": np.random.normal(0, 0.05, (252, 4)),  # 5% daily volatility
                "description": "Extreme market volatility"
            },
            "negative_returns": {
                "returns": np.random.normal(-0.002, 0.02, (252, 4)),  # Negative drift
                "description": "Bear market"
            },
            "low_correlation": {
                "returns": np.random.multivariate_normal(
                    [0.0008] * 4,
                    np.eye(4) * 0.0004,  # No correlation
                    size=252
                ),
                "description": "Uncorrelated assets"
            },
            "high_correlation": {
                "returns": np.random.multivariate_normal(
                    [0.0008] * 4,
                    [[0.0004, 0.00038, 0.00038, 0.00038],
                     [0.00038, 0.0004, 0.00038, 0.00038],
                     [0.00038, 0.00038, 0.0004, 0.00038],
                     [0.00038, 0.00038, 0.00038, 0.0004]],  # High correlation
                    size=252
                ),
                "description": "Highly correlated assets"
            }
        }
        
        current_prices = {asset: 100.0 for asset in assets}
        constraints = OptimizationConstraints()
        
        for scenario_name, scenario_data in scenarios.items():
            print(f"Testing {scenario_name}: {scenario_data['description']}")
            
            result = await optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=scenario_data["returns"],
                current_prices=current_prices,
                constraints=constraints,
                optimization_method="minimum_variance"
            )
            
            # Should handle extreme conditions gracefully
            assert abs(sum(result.weights.values()) - 1.0) < 1e-5
            assert all(0 <= w <= 1 for w in result.weights.values())
            
            print(f"  Risk: {result.portfolio_risk:.4f}")
            print(f"  Return: {result.expected_return:.4f}")
            print(f"  Sharpe: {result.sharpe_ratio:.4f}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_objective_optimization(self):
        """Test optimization with multiple objectives"""
        optimizer = PortfolioOptimizer(device="cpu")
        
        assets = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        
        # Generate returns with known characteristics
        np.random.seed(42)
        
        # AAPL: Low risk, moderate return
        aapl_returns = np.random.normal(0.0006, 0.015, 252)
        
        # GOOGL: Moderate risk, moderate return  
        googl_returns = np.random.normal(0.0008, 0.02, 252)
        
        # MSFT: Low risk, moderate return
        msft_returns = np.random.normal(0.0007, 0.016, 252)
        
        # AMZN: High risk, high return
        amzn_returns = np.random.normal(0.0012, 0.025, 252)
        
        # TSLA: Very high risk, high return
        tsla_returns = np.random.normal(0.0015, 0.035, 252)
        
        historical_returns = np.column_stack([
            aapl_returns, googl_returns, msft_returns, amzn_returns, tsla_returns
        ])
        
        current_prices = {
            "AAPL": 175.0,
            "GOOGL": 140.0,
            "MSFT": 390.0,
            "AMZN": 170.0,
            "TSLA": 250.0
        }
        
        # Test different objectives
        objectives = {
            "min_risk": {
                "method": "minimum_variance",
                "constraints": OptimizationConstraints()
            },
            "max_return": {
                "method": "mean_variance",
                "constraints": OptimizationConstraints(
                    target_return=None,  # No target, just maximize
                    max_risk=None
                )
            },
            "max_sharpe": {
                "method": "maximum_sharpe",
                "constraints": OptimizationConstraints()
            },
            "risk_parity": {
                "method": "risk_parity",
                "constraints": OptimizationConstraints()
            }
        }
        
        results = {}
        
        for objective_name, objective_config in objectives.items():
            result = await optimizer.optimize_portfolio(
                assets=assets,
                historical_returns=historical_returns,
                current_prices=current_prices,
                constraints=objective_config["constraints"],
                optimization_method=objective_config["method"]
            )
            
            results[objective_name] = result
            
            print(f"\n{objective_name}:")
            print(f"  Weights: {result.weights}")
            print(f"  Risk: {result.portfolio_risk:.4f}")
            print(f"  Return: {result.expected_return:.4f}")
            print(f"  Sharpe: {result.sharpe_ratio:.4f}")
        
        # Verify expected behavior
        # Min risk should have lowest volatility
        min_risk_vol = results["min_risk"].portfolio_risk
        max_return_vol = results["max_return"].portfolio_risk
        assert min_risk_vol <= max_return_vol
        
        # Risk parity should have balanced risk contributions
        risk_parity_contributions = list(results["risk_parity"].risk_contributions.values())
        contribution_std = np.std(risk_parity_contributions)
        assert contribution_std < 0.05  # Fairly balanced
        
        # Max Sharpe should have highest Sharpe ratio
        max_sharpe_ratio = results["max_sharpe"].sharpe_ratio
        for name, result in results.items():
            if name != "max_sharpe":
                assert max_sharpe_ratio >= result.sharpe_ratio - 0.1  # Allow small tolerance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
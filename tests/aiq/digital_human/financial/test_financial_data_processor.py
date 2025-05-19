"""
Comprehensive tests for Financial Data Processor

Tests market data processing, technical indicators calculation,
and GPU-accelerated financial data operations.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from aiq.digital_human.financial.financial_data_processor import (
    FinancialDataProcessor,
    MarketData,
    FinancialMetrics,
    MarketIndicators
)


class TestMarketData:
    """Test MarketData dataclass"""
    
    def test_market_data_creation(self):
        """Test creating market data"""
        data = MarketData(
            symbol="AAPL",
            timestamp=datetime.now(),
            open=174.50,
            high=176.00,
            low=173.00,
            close=175.50,
            volume=50000000,
            vwap=175.25
        )
        
        assert data.symbol == "AAPL"
        assert data.open == 174.50
        assert data.high == 176.00
        assert data.low == 173.00
        assert data.close == 175.50
        assert data.volume == 50000000
        assert data.vwap == 175.25


class TestFinancialMetrics:
    """Test FinancialMetrics dataclass"""
    
    def test_financial_metrics_creation(self):
        """Test creating financial metrics"""
        metrics = FinancialMetrics(
            symbol="AAPL",
            market_cap=2.8e12,
            pe_ratio=28.5,
            eps=6.15,
            revenue=380e9,
            profit_margin=0.25,
            debt_to_equity=1.2,
            roe=0.15,
            fcf=100e9
        )
        
        assert metrics.symbol == "AAPL"
        assert metrics.market_cap == 2.8e12
        assert metrics.pe_ratio == 28.5
        assert metrics.eps == 6.15
        assert metrics.revenue == 380e9
        assert metrics.profit_margin == 0.25
        assert metrics.debt_to_equity == 1.2
        assert metrics.roe == 0.15
        assert metrics.fcf == 100e9


class TestMarketIndicators:
    """Test MarketIndicators dataclass"""
    
    def test_market_indicators_creation(self):
        """Test creating market indicators"""
        indicators = MarketIndicators(
            symbol="AAPL",
            sma_20=174.50,
            sma_50=172.00,
            sma_200=170.00,
            rsi=65.5,
            macd={"macd": 1.2, "signal": 0.8, "histogram": 0.4},
            bollinger_bands={"upper": 178.0, "middle": 175.0, "lower": 172.0},
            atr=3.5,
            volume_profile={"level_175.00": 10000000, "level_176.00": 8000000}
        )
        
        assert indicators.symbol == "AAPL"
        assert indicators.sma_20 == 174.50
        assert indicators.sma_50 == 172.00
        assert indicators.sma_200 == 170.00
        assert indicators.rsi == 65.5
        assert indicators.macd["macd"] == 1.2
        assert indicators.bollinger_bands["upper"] == 178.0
        assert indicators.atr == 3.5
        assert "level_175.00" in indicators.volume_profile


class TestFinancialDataProcessor:
    """Test Financial Data Processor implementation"""
    
    @pytest.fixture
    def processor(self):
        """Create processor instance"""
        return FinancialDataProcessor(
            device="cpu",  # Use CPU for tests
            cache_size=100,
            enable_gpu=False
        )
    
    @pytest.fixture
    def symbols(self):
        """Create test symbol list"""
        return ["AAPL", "GOOGL", "MSFT"]
    
    @pytest.fixture
    def market_data_list(self):
        """Create test market data list"""
        base_date = datetime.now() - timedelta(days=30)
        data_list = []
        
        for i in range(30):
            data = MarketData(
                symbol="AAPL",
                timestamp=base_date + timedelta(days=i),
                open=170.0 + i * 0.5,
                high=171.0 + i * 0.5,
                low=169.0 + i * 0.5,
                close=170.0 + i * 0.5,
                volume=50000000 + i * 100000,
                vwap=170.0 + i * 0.5
            )
            data_list.append(data)
        
        return data_list
    
    def test_processor_initialization(self, processor):
        """Test processor initialization"""
        assert processor.device == "cpu"
        assert processor.cache_size == 100
        assert processor.tensor_optimizer is None
        assert len(processor.market_data_cache) == 0
        assert len(processor.metrics_cache) == 0
        assert len(processor.indicators_cache) == 0
    
    def test_processor_gpu_initialization(self):
        """Test processor with GPU initialization"""
        if torch.cuda.is_available():
            processor = FinancialDataProcessor(
                device="cuda",
                enable_gpu=True
            )
            
            assert processor.device == "cuda"
            assert processor.tensor_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_fetch_market_data(self, processor, symbols):
        """Test market data fetching"""
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()
        
        # Mock API fetching
        processor._fetch_from_api = AsyncMock(side_effect=lambda s, start, end: [
            {
                "symbol": s,
                "timestamp": start + timedelta(days=i),
                "open": 100.0 + i,
                "high": 101.0 + i,
                "low": 99.0 + i,
                "close": 100.0 + i,
                "volume": 1000000
            }
            for i in range((end - start).days + 1)
        ])
        
        market_data = await processor.fetch_market_data(symbols, start_date, end_date)
        
        assert len(market_data) == len(symbols)
        assert all(symbol in market_data for symbol in symbols)
        
        # Check data structure
        for symbol, data_list in market_data.items():
            assert len(data_list) > 0
            assert all(isinstance(d, MarketData) for d in data_list)
            assert all(d.symbol == symbol for d in data_list)
    
    @pytest.mark.asyncio
    async def test_market_data_caching(self, processor, symbols):
        """Test market data caching"""
        start_date = datetime.now() - timedelta(days=10)
        end_date = datetime.now()
        
        # Mock API
        processor._fetch_from_api = AsyncMock(side_effect=lambda s, start, end: [
            {
                "symbol": s,
                "timestamp": start,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1000000
            }
        ])
        
        # First fetch
        await processor.fetch_market_data(symbols[:1], start_date, end_date)
        assert processor._fetch_from_api.call_count == 1
        
        # Second fetch (should use cache)
        await processor.fetch_market_data(symbols[:1], start_date, end_date)
        assert processor._fetch_from_api.call_count == 1  # No additional calls
    
    def test_process_raw_market_data(self, processor):
        """Test raw market data processing"""
        raw_data = [
            {
                "symbol": "AAPL",
                "timestamp": datetime.now(),
                "open": 175.0,
                "high": 176.0,
                "low": 174.0,
                "close": 175.5,
                "volume": 50000000
            }
        ]
        
        processed = processor._process_raw_market_data(raw_data)
        
        assert len(processed) == 1
        assert isinstance(processed[0], MarketData)
        assert processed[0].symbol == "AAPL"
        assert processed[0].vwap == (176.0 + 174.0 + 175.5) / 3  # Typical price
    
    @pytest.mark.asyncio
    async def test_calculate_indicators_cpu(self, processor, market_data_list):
        """Test technical indicators calculation on CPU"""
        processor.device = "cpu"
        
        indicators = await processor.calculate_indicators(
            symbol="AAPL",
            market_data=market_data_list,
            indicators=["sma", "rsi", "macd", "bollinger", "atr"]
        )
        
        assert isinstance(indicators, MarketIndicators)
        assert indicators.symbol == "AAPL"
        assert indicators.sma_20 > 0
        assert indicators.sma_50 == 0  # Not enough data
        assert 0 <= indicators.rsi <= 100
        assert "macd" in indicators.macd
        assert "upper" in indicators.bollinger_bands
        assert indicators.atr >= 0
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_calculate_indicators_gpu(self, market_data_list):
        """Test technical indicators calculation on GPU"""
        processor = FinancialDataProcessor(
            device="cuda",
            enable_gpu=True
        )
        
        indicators = await processor.calculate_indicators(
            symbol="AAPL",
            market_data=market_data_list,
            indicators=["sma", "rsi", "macd", "bollinger", "atr"]
        )
        
        assert isinstance(indicators, MarketIndicators)
        assert indicators.symbol == "AAPL"
        assert indicators.sma_20 > 0
        assert 0 <= indicators.rsi <= 100
    
    def test_calculate_sma_gpu(self, processor):
        """Test SMA calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.randn(100, device="cuda") * 10 + 100
        
        sma = processor._calculate_sma_gpu(prices, window=20)
        
        assert sma.device.type == "cuda"
        assert len(sma) == 81  # 100 - 20 + 1
        
        # Verify calculation
        expected_first = torch.mean(prices[:20])
        assert torch.abs(sma[0] - expected_first) < 1e-5
    
    def test_calculate_rsi_gpu(self, processor):
        """Test RSI calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.tensor([float(i) for i in range(100)], device="cuda")
        
        rsi = processor._calculate_rsi_gpu(prices, period=14)
        
        assert rsi.device.type == "cuda"
        assert len(rsi) > 0
        assert torch.all((rsi >= 0) & (rsi <= 100))
    
    def test_calculate_macd_gpu(self, processor):
        """Test MACD calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.randn(100, device="cuda") * 10 + 100
        
        macd = processor._calculate_macd_gpu(prices)
        
        assert isinstance(macd, dict)
        assert "macd" in macd
        assert "signal" in macd
        assert "histogram" in macd
    
    def test_calculate_bollinger_bands_gpu(self, processor):
        """Test Bollinger Bands calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.randn(100, device="cuda") * 10 + 100
        
        bb = processor._calculate_bollinger_bands_gpu(prices, period=20)
        
        assert isinstance(bb, dict)
        assert "upper" in bb
        assert "middle" in bb
        assert "lower" in bb
        assert bb["upper"] > bb["middle"] > bb["lower"]
    
    def test_calculate_ema_gpu(self, processor):
        """Test EMA calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.tensor([float(i) for i in range(50)], device="cuda")
        
        ema = processor._calculate_ema_gpu(prices, period=12)
        
        assert ema.device.type == "cuda"
        assert len(ema) == 39  # 50 - 12 + 1
    
    def test_calculate_volume_profile_gpu(self, processor):
        """Test Volume Profile calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        prices = torch.randn(100, device="cuda") * 10 + 100
        volumes = torch.randn(100, device="cuda") * 1000000 + 5000000
        
        profile = processor._calculate_volume_profile_gpu(prices, volumes, num_bins=10)
        
        assert isinstance(profile, dict)
        assert len(profile) <= 10
        assert all(v >= 0 for v in profile.values())
    
    @pytest.mark.asyncio
    async def test_calculate_indicators_cpu_fallback(self, processor, market_data_list):
        """Test CPU fallback for indicators calculation"""
        processor.device = "cpu"
        processor.use_cuda = False
        
        indicators = await processor._cpu_calculate_indicators(
            symbol="AAPL",
            prices=np.array([d.close for d in market_data_list]),
            volumes=np.array([d.volume for d in market_data_list]),
            indicators=["sma", "rsi", "macd", "bollinger", "atr"]
        )
        
        assert isinstance(indicators, MarketIndicators)
        assert indicators.symbol == "AAPL"
        assert indicators.sma_20 > 0
        assert 0 <= indicators.rsi <= 100
    
    @pytest.mark.asyncio
    async def test_fetch_financial_metrics(self, processor, symbols):
        """Test financial metrics fetching"""
        # Mock API
        processor._fetch_fundamentals = AsyncMock(return_value={
            "market_cap": 2.8e12,
            "pe_ratio": 28.5,
            "eps": 6.15,
            "revenue": 380e9,
            "profit_margin": 0.25,
            "debt_to_equity": 1.2,
            "roe": 0.15,
            "fcf": 100e9
        })
        
        metrics = await processor.fetch_financial_metrics(symbols)
        
        assert len(metrics) == len(symbols)
        assert all(symbol in metrics for symbol in symbols)
        
        for symbol, metric in metrics.items():
            assert isinstance(metric, FinancialMetrics)
            assert metric.symbol == symbol
            assert metric.market_cap > 0
            assert metric.pe_ratio > 0
    
    @pytest.mark.asyncio
    async def test_metrics_caching(self, processor, symbols):
        """Test financial metrics caching"""
        # Mock API
        processor._fetch_fundamentals = AsyncMock(return_value={
            "market_cap": 1e12,
            "pe_ratio": 20,
            "eps": 5.0,
            "revenue": 100e9,
            "profit_margin": 0.20,
            "debt_to_equity": 1.0,
            "roe": 0.12,
            "fcf": 50e9
        })
        
        # First fetch
        await processor.fetch_financial_metrics(symbols[:1])
        assert processor._fetch_fundamentals.call_count == 1
        
        # Second fetch (should use cache)
        await processor.fetch_financial_metrics(symbols[:1])
        assert processor._fetch_fundamentals.call_count == 1
    
    def test_calculate_correlation_matrix_cpu(self, processor):
        """Test correlation matrix calculation on CPU"""
        returns_data = {
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.001, 0.02, 252)
        }
        
        correlation_df = processor.calculate_correlation_matrix(returns_data)
        
        assert isinstance(correlation_df, pd.DataFrame)
        assert correlation_df.shape == (3, 3)
        assert list(correlation_df.index) == ["AAPL", "GOOGL", "MSFT"]
        assert list(correlation_df.columns) == ["AAPL", "GOOGL", "MSFT"]
        
        # Diagonal should be 1.0
        for i in range(3):
            assert abs(correlation_df.iloc[i, i] - 1.0) < 1e-10
    
    def test_calculate_correlation_matrix_gpu(self, processor):
        """Test correlation matrix calculation on GPU"""
        if not torch.cuda.is_available():
            pytest.skip("GPU not available")
        
        processor.device = "cuda"
        returns_data = {
            "AAPL": np.random.normal(0.001, 0.02, 252),
            "GOOGL": np.random.normal(0.001, 0.02, 252),
            "MSFT": np.random.normal(0.001, 0.02, 252)
        }
        
        correlation_df = processor._calculate_correlation_gpu(returns_data)
        
        assert isinstance(correlation_df, pd.DataFrame)
        assert correlation_df.shape == (3, 3)
    
    def test_detect_market_regime(self, processor, market_data_list):
        """Test market regime detection"""
        # Test with trending market
        trending_data = []
        base_price = 100.0
        for i in range(252):
            trending_data.append(MarketData(
                symbol="AAPL",
                timestamp=datetime.now() + timedelta(days=i),
                open=base_price + i * 0.5,
                high=base_price + i * 0.5 + 1,
                low=base_price + i * 0.5 - 1,
                close=base_price + i * 0.5,
                volume=1000000,
                vwap=base_price + i * 0.5
            ))
        
        regime = processor.detect_market_regime(trending_data)
        assert regime in ["strong_uptrend", "weak_trend"]
        
        # Test with volatile market
        volatile_data = []
        for i in range(252):
            price = 100 + np.random.normal(0, 5)  # High volatility
            volatile_data.append(MarketData(
                symbol="AAPL",
                timestamp=datetime.now() + timedelta(days=i),
                open=price,
                high=price + 2,
                low=price - 2,
                close=price,
                volume=1000000,
                vwap=price
            ))
        
        regime = processor.detect_market_regime(volatile_data)
        assert regime in ["volatile_ranging", "ranging"]
    
    def test_calculate_risk_metrics(self, processor):
        """Test risk metrics calculation"""
        returns = np.random.normal(0.001, 0.02, 252)
        benchmark_returns = np.random.normal(0.0008, 0.018, 252)
        
        metrics = processor.calculate_risk_metrics(returns, benchmark_returns)
        
        assert "volatility" in metrics
        assert "downside_volatility" in metrics
        assert "var_95" in metrics
        assert "var_99" in metrics
        assert "cvar_95" in metrics
        assert "cvar_99" in metrics
        assert "max_drawdown" in metrics
        assert "beta" in metrics
        assert "alpha" in metrics
        
        # Check reasonable values
        assert metrics["volatility"] > 0
        assert metrics["downside_volatility"] > 0
        assert metrics["var_95"] < 0
        assert metrics["var_99"] < metrics["var_95"]
        assert metrics["cvar_95"] < metrics["var_95"]
        assert metrics["max_drawdown"] < 0
    
    def test_compile_processing_kernels(self, processor):
        """Test CUDA kernel compilation"""
        if not torch.cuda.is_available():
            processor._compile_processing_kernels()
            # Should create kernel strings
            assert hasattr(processor, 'indicator_kernel')
            assert hasattr(processor, 'correlation_kernel')
            assert isinstance(processor.indicator_kernel, str)
            assert isinstance(processor.correlation_kernel, str)


class TestIntegration:
    """Integration tests for Financial Data Processor"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complete_data_processing_workflow(self):
        """Test complete data processing workflow"""
        processor = FinancialDataProcessor(
            device="cpu",
            cache_size=1000
        )
        
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
        start_date = datetime.now() - timedelta(days=365)
        end_date = datetime.now()
        
        # Fetch market data
        market_data = await processor.fetch_market_data(symbols, start_date, end_date)
        
        assert len(market_data) == len(symbols)
        
        # Calculate indicators for each symbol
        indicators = {}
        for symbol, data_list in market_data.items():
            indicator = await processor.calculate_indicators(
                symbol=symbol,
                market_data=data_list,
                indicators=["sma", "rsi", "macd", "bollinger", "atr"]
            )
            indicators[symbol] = indicator
        
        assert len(indicators) == len(symbols)
        
        # Fetch financial metrics
        metrics = await processor.fetch_financial_metrics(symbols)
        
        assert len(metrics) == len(symbols)
        
        # Calculate correlation matrix
        returns_data = {}
        for symbol, data_list in market_data.items():
            prices = np.array([d.close for d in data_list])
            returns = np.diff(prices) / prices[:-1]
            returns_data[symbol] = returns
        
        correlation_matrix = processor.calculate_correlation_matrix(returns_data)
        
        assert correlation_matrix.shape == (len(symbols), len(symbols))
        
        # Detect market regime
        for symbol, data_list in market_data.items():
            regime = processor.detect_market_regime(data_list)
            assert regime in [
                "strong_uptrend", "strong_downtrend", "weak_trend",
                "ranging", "volatile_ranging", "unknown"
            ]
        
        # Calculate risk metrics
        for symbol, returns in returns_data.items():
            risk_metrics = processor.calculate_risk_metrics(returns)
            assert "volatility" in risk_metrics
            assert "var_95" in risk_metrics
            assert "max_drawdown" in risk_metrics
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_gpu_performance_comparison(self):
        """Compare GPU vs CPU performance"""
        cpu_processor = FinancialDataProcessor(device="cpu", enable_gpu=False)
        gpu_processor = FinancialDataProcessor(device="cuda", enable_gpu=True)
        
        # Generate large dataset
        symbols = [f"STOCK_{i}" for i in range(10)]
        data_points = 1000
        
        market_data = {}
        for symbol in symbols:
            data_list = []
            base_price = 100.0
            for i in range(data_points):
                price = base_price + np.random.normal(0, 2)
                data_list.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.now() + timedelta(days=i),
                    open=price - 0.5,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price,
                    volume=1000000,
                    vwap=price
                ))
            market_data[symbol] = data_list
        
        import time
        
        # CPU timing
        cpu_start = time.time()
        cpu_indicators = {}
        for symbol, data_list in market_data.items():
            indicator = await cpu_processor.calculate_indicators(
                symbol=symbol,
                market_data=data_list,
                indicators=["sma", "rsi", "macd", "bollinger", "atr"]
            )
            cpu_indicators[symbol] = indicator
        cpu_time = time.time() - cpu_start
        
        # GPU timing
        gpu_start = time.time()
        gpu_indicators = {}
        for symbol, data_list in market_data.items():
            indicator = await gpu_processor.calculate_indicators(
                symbol=symbol,
                market_data=data_list,
                indicators=["sma", "rsi", "macd", "bollinger", "atr"]
            )
            gpu_indicators[symbol] = indicator
        gpu_time = time.time() - gpu_start
        
        print(f"CPU time: {cpu_time:.3f}s")
        print(f"GPU time: {gpu_time:.3f}s")
        print(f"Speedup: {cpu_time/gpu_time:.2f}x")
        
        # Verify results are similar
        for symbol in symbols:
            cpu_ind = cpu_indicators[symbol]
            gpu_ind = gpu_indicators[symbol]
            
            assert abs(cpu_ind.sma_20 - gpu_ind.sma_20) < 0.01
            assert abs(cpu_ind.rsi - gpu_ind.rsi) < 1.0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_real_world_scenario(self):
        """Test with real-world like data patterns"""
        processor = FinancialDataProcessor(device="cpu")
        
        # Simulate different market scenarios
        scenarios = {
            "bull_market": {
                "trend": 0.0008,  # 0.08% daily
                "volatility": 0.015,
                "description": "Bull market with steady growth"
            },
            "bear_market": {
                "trend": -0.0005,  # -0.05% daily
                "volatility": 0.025,
                "description": "Bear market with high volatility"
            },
            "sideways_market": {
                "trend": 0.0,
                "volatility": 0.018,
                "description": "Sideways market"
            },
            "crash_recovery": {
                "trend": 0.001,
                "volatility": 0.04,
                "description": "Recovery after crash"
            }
        }
        
        for scenario_name, params in scenarios.items():
            print(f"\nTesting {scenario_name}: {params['description']}")
            
            # Generate data
            data_list = []
            base_price = 100.0
            current_price = base_price
            
            for i in range(252):  # One year
                # Add trend and random walk
                daily_return = np.random.normal(params["trend"], params["volatility"])
                current_price *= (1 + daily_return)
                
                # Add intraday volatility
                intraday_range = current_price * params["volatility"] * 0.5
                
                data_list.append(MarketData(
                    symbol="TEST",
                    timestamp=datetime.now() + timedelta(days=i),
                    open=current_price - intraday_range * 0.3,
                    high=current_price + intraday_range,
                    low=current_price - intraday_range,
                    close=current_price,
                    volume=int(1000000 * (1 + abs(daily_return) * 10)),
                    vwap=current_price
                ))
            
            # Calculate indicators
            indicators = await processor.calculate_indicators(
                symbol="TEST",
                market_data=data_list,
                indicators=["sma", "rsi", "macd", "bollinger", "atr"]
            )
            
            # Detect regime
            regime = processor.detect_market_regime(data_list)
            
            # Calculate returns
            prices = np.array([d.close for d in data_list])
            returns = np.diff(prices) / prices[:-1]
            
            # Risk metrics
            risk_metrics = processor.calculate_risk_metrics(returns)
            
            print(f"  Final price: ${current_price:.2f} ({(current_price/base_price - 1)*100:.1f}%)")
            print(f"  RSI: {indicators.rsi:.1f}")
            print(f"  Volatility: {risk_metrics['volatility']:.3f}")
            print(f"  Max Drawdown: {risk_metrics['max_drawdown']:.3f}")
            print(f"  Market Regime: {regime}")
            
            # Verify reasonable results
            if scenario_name == "bull_market":
                assert current_price > base_price
                assert indicators.rsi > 50
            elif scenario_name == "bear_market":
                assert current_price < base_price
                assert risk_metrics["volatility"] > 0.02
            elif scenario_name == "sideways_market":
                assert abs(current_price - base_price) / base_price < 0.2
            elif scenario_name == "crash_recovery":
                assert risk_metrics["max_drawdown"] < -0.1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        processor = FinancialDataProcessor(device="cpu")
        
        # Empty data
        empty_indicators = await processor.calculate_indicators(
            symbol="TEST",
            market_data=[],
            indicators=["sma", "rsi"]
        )
        
        assert empty_indicators.symbol == "TEST"
        assert empty_indicators.sma_20 == 0
        assert empty_indicators.rsi == 50  # Default neutral value
        
        # Insufficient data
        short_data = [
            MarketData(
                symbol="TEST",
                timestamp=datetime.now(),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000000,
                vwap=100.0
            )
            for _ in range(5)
        ]
        
        indicators = await processor.calculate_indicators(
            symbol="TEST",
            market_data=short_data,
            indicators=["sma", "rsi", "macd"]
        )
        
        assert indicators.sma_20 == 0  # Not enough data
        assert indicators.rsi == 50  # Default
        
        # Invalid returns data
        invalid_returns = {
            "STOCK1": np.array([]),  # Empty
            "STOCK2": np.array([0.01])  # Single value
        }
        
        correlation = processor.calculate_correlation_matrix(invalid_returns)
        assert correlation.shape[0] >= 0  # Should handle gracefully
        
        # Risk metrics with no variance
        constant_returns = np.ones(100) * 0.001
        risk_metrics = processor.calculate_risk_metrics(constant_returns)
        
        assert risk_metrics["volatility"] >= 0
        assert "max_drawdown" in risk_metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
"""
Financial Data Processor with Market Integration

Processes market data, financial statements, and real-time feeds
for the digital human financial advisor system.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import torch
import asyncio
import aiohttp
from collections import deque

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    vwap: float  # Volume-weighted average price
    
    
@dataclass
class FinancialMetrics:
    """Company financial metrics"""
    symbol: str
    market_cap: float
    pe_ratio: float
    eps: float
    revenue: float
    profit_margin: float
    debt_to_equity: float
    roe: float  # Return on equity
    fcf: float  # Free cash flow


@dataclass
class MarketIndicators:
    """Technical and market indicators"""
    symbol: str
    sma_20: float
    sma_50: float
    sma_200: float
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    atr: float  # Average true range
    volume_profile: Dict[str, float]


class FinancialDataProcessor:
    """
    Processes financial data with GPU acceleration for real-time analysis.
    Integrates multiple data sources and provides cleaned, normalized data.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        cache_size: int = 10000,
        enable_gpu: bool = True
    ):
        self.device = device if torch.cuda.is_available() and enable_gpu else "cpu"
        self.cache_size = cache_size
        
        # Data caches
        self.market_data_cache = {}
        self.metrics_cache = {}
        self.indicators_cache = {}
        
        # Historical data buffers
        self.price_history = {}
        self.volume_history = {}
        
        # GPU optimization
        if self.device == "cuda":
            self.tensor_optimizer = TensorCoreOptimizer()
            self._compile_processing_kernels()
        
        # API clients (would be configured with actual endpoints)
        self.market_data_api = None
        self.fundamentals_api = None
        self.news_api = None
    
    def _compile_processing_kernels(self):
        """Compile CUDA kernels for data processing"""
        # Technical indicator calculation kernel
        self.indicator_kernel = """
        __global__ void calculate_indicators(
            float* prices,
            float* volumes,
            float* sma_out,
            float* rsi_out,
            float* vwap_out,
            int data_length,
            int window_size
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < data_length - window_size) {
                // Simple Moving Average
                float sum = 0.0f;
                for (int i = 0; i < window_size; i++) {
                    sum += prices[idx + i];
                }
                sma_out[idx] = sum / window_size;
                
                // VWAP calculation
                float price_volume_sum = 0.0f;
                float volume_sum = 0.0f;
                for (int i = 0; i < window_size; i++) {
                    price_volume_sum += prices[idx + i] * volumes[idx + i];
                    volume_sum += volumes[idx + i];
                }
                vwap_out[idx] = price_volume_sum / volume_sum;
                
                // RSI calculation (simplified)
                float gains = 0.0f;
                float losses = 0.0f;
                for (int i = 1; i < window_size; i++) {
                    float change = prices[idx + i] - prices[idx + i - 1];
                    if (change > 0) gains += change;
                    else losses -= change;
                }
                float avg_gain = gains / window_size;
                float avg_loss = losses / window_size;
                float rs = avg_gain / (avg_loss + 1e-10f);
                rsi_out[idx] = 100.0f - (100.0f / (1.0f + rs));
            }
        }
        """
        
        # Correlation matrix calculation
        self.correlation_kernel = """
        __global__ void calculate_correlation(
            float* returns,
            float* correlation_matrix,
            int num_assets,
            int num_samples
        ) {
            int i = blockIdx.x;
            int j = blockIdx.y;
            
            if (i < num_assets && j < num_assets) {
                float sum_i = 0.0f, sum_j = 0.0f;
                float sum_i2 = 0.0f, sum_j2 = 0.0f;
                float sum_ij = 0.0f;
                
                for (int k = 0; k < num_samples; k++) {
                    float val_i = returns[k * num_assets + i];
                    float val_j = returns[k * num_assets + j];
                    
                    sum_i += val_i;
                    sum_j += val_j;
                    sum_i2 += val_i * val_i;
                    sum_j2 += val_j * val_j;
                    sum_ij += val_i * val_j;
                }
                
                float mean_i = sum_i / num_samples;
                float mean_j = sum_j / num_samples;
                
                float cov = (sum_ij - sum_i * sum_j / num_samples) / (num_samples - 1);
                float std_i = sqrtf((sum_i2 - sum_i * sum_i / num_samples) / (num_samples - 1));
                float std_j = sqrtf((sum_j2 - sum_j * sum_j / num_samples) / (num_samples - 1));
                
                correlation_matrix[i * num_assets + j] = cov / (std_i * std_j + 1e-10f);
            }
        }
        """
    
    async def fetch_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> Dict[str, List[MarketData]]:
        """Fetch market data for given symbols and date range"""
        market_data = {}
        
        for symbol in symbols:
            # Check cache first
            cache_key = f"{symbol}_{start_date}_{end_date}"
            if cache_key in self.market_data_cache:
                market_data[symbol] = self.market_data_cache[cache_key]
                continue
            
            # Fetch from API (placeholder implementation)
            data = await self._fetch_from_api(symbol, start_date, end_date)
            
            # Process and cache
            processed_data = self._process_raw_market_data(data)
            self.market_data_cache[cache_key] = processed_data
            market_data[symbol] = processed_data
        
        return market_data
    
    async def _fetch_from_api(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime
    ) -> List[Dict[str, Any]]:
        """Fetch data from market data API (placeholder)"""
        # In production, this would make actual API calls
        # For now, generate synthetic data
        data = []
        current_date = start_date
        current_price = 100.0
        
        while current_date <= end_date:
            # Generate random walk
            change = np.random.normal(0, 0.02)
            current_price *= (1 + change)
            
            volume = int(np.random.uniform(1000000, 5000000))
            
            data.append({
                "symbol": symbol,
                "timestamp": current_date,
                "open": current_price * 0.99,
                "high": current_price * 1.01,
                "low": current_price * 0.98,
                "close": current_price,
                "volume": volume
            })
            
            current_date += timedelta(days=1)
        
        return data
    
    def _process_raw_market_data(self, raw_data: List[Dict[str, Any]]) -> List[MarketData]:
        """Process raw market data into structured format"""
        processed_data = []
        
        for item in raw_data:
            # Calculate VWAP
            typical_price = (item["high"] + item["low"] + item["close"]) / 3
            vwap = typical_price  # Simplified VWAP
            
            market_data = MarketData(
                symbol=item["symbol"],
                timestamp=item["timestamp"],
                open=item["open"],
                high=item["high"],
                low=item["low"],
                close=item["close"],
                volume=item["volume"],
                vwap=vwap
            )
            
            processed_data.append(market_data)
        
        return processed_data
    
    async def calculate_indicators(
        self,
        symbol: str,
        market_data: List[MarketData],
        indicators: List[str] = None
    ) -> MarketIndicators:
        """Calculate technical indicators for a symbol"""
        if not indicators:
            indicators = ["sma", "rsi", "macd", "bollinger", "atr"]
        
        # Convert to arrays
        prices = np.array([d.close for d in market_data])
        volumes = np.array([d.volume for d in market_data])
        
        if self.device == "cuda":
            return await self._gpu_calculate_indicators(
                symbol, prices, volumes, indicators
            )
        else:
            return await self._cpu_calculate_indicators(
                symbol, prices, volumes, indicators
            )
    
    async def _gpu_calculate_indicators(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        indicators: List[str]
    ) -> MarketIndicators:
        """GPU-accelerated indicator calculation"""
        # Convert to tensors
        price_tensor = torch.tensor(prices, device=self.device, dtype=torch.float32)
        volume_tensor = torch.tensor(volumes, device=self.device, dtype=torch.float32)
        
        results = {}
        
        # Simple Moving Averages
        if "sma" in indicators:
            sma_20 = self._calculate_sma_gpu(price_tensor, 20)
            sma_50 = self._calculate_sma_gpu(price_tensor, 50)
            sma_200 = self._calculate_sma_gpu(price_tensor, 200)
            
            results["sma_20"] = sma_20[-1].item() if len(sma_20) > 0 else 0
            results["sma_50"] = sma_50[-1].item() if len(sma_50) > 0 else 0
            results["sma_200"] = sma_200[-1].item() if len(sma_200) > 0 else 0
        
        # RSI
        if "rsi" in indicators:
            rsi = self._calculate_rsi_gpu(price_tensor, 14)
            results["rsi"] = rsi[-1].item() if len(rsi) > 0 else 50
        
        # MACD
        if "macd" in indicators:
            macd_result = self._calculate_macd_gpu(price_tensor)
            results["macd"] = macd_result
        
        # Bollinger Bands
        if "bollinger" in indicators:
            bb = self._calculate_bollinger_bands_gpu(price_tensor, 20, 2)
            results["bollinger_bands"] = bb
        
        # ATR
        if "atr" in indicators:
            # Need high/low prices for ATR
            atr = self._calculate_atr_gpu(price_tensor, price_tensor, price_tensor, 14)
            results["atr"] = atr[-1].item() if len(atr) > 0 else 0
        
        # Volume Profile
        results["volume_profile"] = self._calculate_volume_profile_gpu(
            price_tensor, volume_tensor
        )
        
        return MarketIndicators(
            symbol=symbol,
            sma_20=results.get("sma_20", 0),
            sma_50=results.get("sma_50", 0),
            sma_200=results.get("sma_200", 0),
            rsi=results.get("rsi", 50),
            macd=results.get("macd", {}),
            bollinger_bands=results.get("bollinger_bands", {}),
            atr=results.get("atr", 0),
            volume_profile=results.get("volume_profile", {})
        )
    
    def _calculate_sma_gpu(self, prices: torch.Tensor, window: int) -> torch.Tensor:
        """Calculate Simple Moving Average on GPU"""
        if len(prices) < window:
            return torch.tensor([])
        
        # Use convolution for efficient SMA calculation
        kernel = torch.ones(window, device=self.device) / window
        kernel = kernel.view(1, 1, -1)
        
        prices_padded = torch.nn.functional.pad(
            prices.view(1, 1, -1),
            (window-1, 0),
            mode='constant',
            value=0
        )
        
        sma = torch.nn.functional.conv1d(prices_padded, kernel)
        return sma.squeeze()[window-1:]
    
    def _calculate_rsi_gpu(self, prices: torch.Tensor, period: int = 14) -> torch.Tensor:
        """Calculate RSI on GPU"""
        if len(prices) < period:
            return torch.tensor([])
        
        # Calculate price changes
        deltas = prices[1:] - prices[:-1]
        
        # Separate gains and losses
        gains = torch.where(deltas > 0, deltas, torch.zeros_like(deltas))
        losses = torch.where(deltas < 0, -deltas, torch.zeros_like(deltas))
        
        # Calculate average gains and losses
        avg_gains = self._calculate_sma_gpu(gains, period)
        avg_losses = self._calculate_sma_gpu(losses, period)
        
        # Calculate RSI
        rs = avg_gains / (avg_losses + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd_gpu(self, prices: torch.Tensor) -> Dict[str, float]:
        """Calculate MACD on GPU"""
        # Calculate EMAs
        ema_12 = self._calculate_ema_gpu(prices, 12)
        ema_26 = self._calculate_ema_gpu(prices, 26)
        
        if len(ema_12) == 0 or len(ema_26) == 0:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        # MACD line
        macd_line = ema_12 - ema_26[:len(ema_12)]
        
        # Signal line
        signal_line = self._calculate_ema_gpu(macd_line, 9)
        
        if len(signal_line) == 0:
            return {"macd": 0, "signal": 0, "histogram": 0}
        
        # MACD histogram
        histogram = macd_line[:len(signal_line)] - signal_line
        
        return {
            "macd": macd_line[-1].item(),
            "signal": signal_line[-1].item(),
            "histogram": histogram[-1].item()
        }
    
    def _calculate_ema_gpu(self, prices: torch.Tensor, period: int) -> torch.Tensor:
        """Calculate Exponential Moving Average on GPU"""
        if len(prices) < period:
            return torch.tensor([])
        
        alpha = 2 / (period + 1)
        
        # Initialize with SMA
        ema = torch.zeros_like(prices)
        ema[period-1] = torch.mean(prices[:period])
        
        # Calculate EMA
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema[period-1:]
    
    def _calculate_bollinger_bands_gpu(
        self,
        prices: torch.Tensor,
        period: int = 20,
        num_std: float = 2
    ) -> Dict[str, float]:
        """Calculate Bollinger Bands on GPU"""
        sma = self._calculate_sma_gpu(prices, period)
        
        if len(sma) == 0:
            return {"upper": 0, "middle": 0, "lower": 0}
        
        # Calculate rolling standard deviation
        std_dev = torch.zeros_like(sma)
        for i in range(len(sma)):
            window_prices = prices[i:i+period]
            std_dev[i] = torch.std(window_prices)
        
        upper_band = sma + num_std * std_dev
        lower_band = sma - num_std * std_dev
        
        return {
            "upper": upper_band[-1].item(),
            "middle": sma[-1].item(),
            "lower": lower_band[-1].item()
        }
    
    def _calculate_atr_gpu(
        self,
        high: torch.Tensor,
        low: torch.Tensor,
        close: torch.Tensor,
        period: int = 14
    ) -> torch.Tensor:
        """Calculate Average True Range on GPU"""
        if len(high) < period:
            return torch.tensor([])
        
        # True Range calculation
        tr1 = high - low
        tr2 = torch.abs(high[1:] - close[:-1])
        tr3 = torch.abs(low[1:] - close[:-1])
        
        # Pad to align
        tr2 = torch.cat([torch.tensor([0], device=self.device), tr2])
        tr3 = torch.cat([torch.tensor([0], device=self.device), tr3])
        
        # Maximum of three values
        true_range = torch.maximum(tr1, torch.maximum(tr2, tr3))
        
        # Calculate ATR
        atr = self._calculate_sma_gpu(true_range, period)
        
        return atr
    
    def _calculate_volume_profile_gpu(
        self,
        prices: torch.Tensor,
        volumes: torch.Tensor,
        num_bins: int = 20
    ) -> Dict[str, float]:
        """Calculate Volume Profile on GPU"""
        min_price = torch.min(prices).item()
        max_price = torch.max(prices).item()
        
        # Create price bins
        bins = torch.linspace(min_price, max_price, num_bins + 1, device=self.device)
        
        # Digitize prices into bins
        price_bins = torch.bucketize(prices, bins[:-1])
        
        # Accumulate volume in each bin
        volume_profile = {}
        for i in range(num_bins):
            mask = price_bins == i
            if torch.any(mask):
                total_volume = torch.sum(volumes[mask]).item()
                price_level = (bins[i] + bins[i+1]) / 2
                volume_profile[f"level_{price_level:.2f}"] = total_volume
        
        return volume_profile
    
    async def _cpu_calculate_indicators(
        self,
        symbol: str,
        prices: np.ndarray,
        volumes: np.ndarray,
        indicators: List[str]
    ) -> MarketIndicators:
        """CPU fallback for indicator calculation"""
        results = {}
        
        # Simple Moving Averages
        if "sma" in indicators:
            results["sma_20"] = np.mean(prices[-20:]) if len(prices) >= 20 else 0
            results["sma_50"] = np.mean(prices[-50:]) if len(prices) >= 50 else 0
            results["sma_200"] = np.mean(prices[-200:]) if len(prices) >= 200 else 0
        
        # RSI (simplified)
        if "rsi" in indicators:
            if len(prices) >= 15:
                deltas = np.diff(prices)
                gains = deltas[deltas > 0]
                losses = -deltas[deltas < 0]
                
                avg_gain = np.mean(gains[-14:]) if len(gains) >= 14 else 0
                avg_loss = np.mean(losses[-14:]) if len(losses) >= 14 else 0
                
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    rsi = 100 - (100 / (1 + rs))
                else:
                    rsi = 100 if avg_gain > 0 else 50
            else:
                rsi = 50
            
            results["rsi"] = rsi
        
        # MACD (simplified)
        if "macd" in indicators:
            if len(prices) >= 26:
                ema_12 = self._calculate_ema_cpu(prices, 12)
                ema_26 = self._calculate_ema_cpu(prices, 26)
                macd_line = ema_12 - ema_26
                signal_line = self._calculate_ema_cpu(macd_line, 9)
                
                results["macd"] = {
                    "macd": macd_line[-1],
                    "signal": signal_line[-1],
                    "histogram": macd_line[-1] - signal_line[-1]
                }
            else:
                results["macd"] = {"macd": 0, "signal": 0, "histogram": 0}
        
        # Bollinger Bands
        if "bollinger" in indicators:
            if len(prices) >= 20:
                sma = np.mean(prices[-20:])
                std = np.std(prices[-20:])
                
                results["bollinger_bands"] = {
                    "upper": sma + 2 * std,
                    "middle": sma,
                    "lower": sma - 2 * std
                }
            else:
                results["bollinger_bands"] = {"upper": 0, "middle": 0, "lower": 0}
        
        # ATR (simplified)
        if "atr" in indicators:
            results["atr"] = np.std(prices[-14:]) if len(prices) >= 14 else 0
        
        # Volume Profile
        results["volume_profile"] = {}
        
        return MarketIndicators(
            symbol=symbol,
            sma_20=results.get("sma_20", 0),
            sma_50=results.get("sma_50", 0),
            sma_200=results.get("sma_200", 0),
            rsi=results.get("rsi", 50),
            macd=results.get("macd", {}),
            bollinger_bands=results.get("bollinger_bands", {}),
            atr=results.get("atr", 0),
            volume_profile=results.get("volume_profile", {})
        )
    
    def _calculate_ema_cpu(self, prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate EMA on CPU"""
        if len(prices) < period:
            return np.array([])
        
        alpha = 2 / (period + 1)
        ema = np.zeros_like(prices)
        ema[period-1] = np.mean(prices[:period])
        
        for i in range(period, len(prices)):
            ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
        
        return ema[period-1:]
    
    async def fetch_financial_metrics(
        self,
        symbols: List[str]
    ) -> Dict[str, FinancialMetrics]:
        """Fetch fundamental financial metrics for symbols"""
        metrics = {}
        
        for symbol in symbols:
            # Check cache
            if symbol in self.metrics_cache:
                metrics[symbol] = self.metrics_cache[symbol]
                continue
            
            # Fetch from API (placeholder)
            data = await self._fetch_fundamentals(symbol)
            
            # Create metrics object
            financial_metrics = FinancialMetrics(
                symbol=symbol,
                market_cap=data.get("market_cap", 0),
                pe_ratio=data.get("pe_ratio", 0),
                eps=data.get("eps", 0),
                revenue=data.get("revenue", 0),
                profit_margin=data.get("profit_margin", 0),
                debt_to_equity=data.get("debt_to_equity", 0),
                roe=data.get("roe", 0),
                fcf=data.get("fcf", 0)
            )
            
            self.metrics_cache[symbol] = financial_metrics
            metrics[symbol] = financial_metrics
        
        return metrics
    
    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, float]:
        """Fetch fundamental data (placeholder)"""
        # In production, this would make actual API calls
        return {
            "market_cap": np.random.uniform(1e9, 1e12),
            "pe_ratio": np.random.uniform(10, 40),
            "eps": np.random.uniform(1, 20),
            "revenue": np.random.uniform(1e8, 1e11),
            "profit_margin": np.random.uniform(0.05, 0.3),
            "debt_to_equity": np.random.uniform(0.1, 2.0),
            "roe": np.random.uniform(0.05, 0.3),
            "fcf": np.random.uniform(1e7, 1e10)
        }
    
    def calculate_correlation_matrix(
        self,
        returns_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        if self.device == "cuda":
            return self._calculate_correlation_gpu(returns_data)
        else:
            return self._calculate_correlation_cpu(returns_data)
    
    def _calculate_correlation_gpu(
        self,
        returns_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """GPU-accelerated correlation calculation"""
        symbols = list(returns_data.keys())
        n_assets = len(symbols)
        
        # Stack returns into matrix
        returns_matrix = np.column_stack(
            [returns_data[symbol] for symbol in symbols]
        )
        
        # Convert to tensor
        returns_tensor = torch.tensor(
            returns_matrix,
            device=self.device,
            dtype=torch.float32
        )
        
        # Calculate correlation matrix
        correlation_tensor = torch.corrcoef(returns_tensor.T)
        
        # Convert to DataFrame
        correlation_df = pd.DataFrame(
            correlation_tensor.cpu().numpy(),
            index=symbols,
            columns=symbols
        )
        
        return correlation_df
    
    def _calculate_correlation_cpu(
        self,
        returns_data: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """CPU correlation calculation"""
        symbols = list(returns_data.keys())
        
        # Create DataFrame from returns
        returns_df = pd.DataFrame(returns_data)
        
        # Calculate correlation
        correlation_df = returns_df.corr()
        
        return correlation_df
    
    def detect_market_regime(
        self,
        market_data: List[MarketData],
        window: int = 252
    ) -> str:
        """Detect current market regime (trending, ranging, volatile)"""
        if len(market_data) < window:
            return "unknown"
        
        prices = np.array([d.close for d in market_data])
        
        # Calculate metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns)
        trend_strength = (prices[-1] - prices[-window]) / prices[-window]
        
        # Regime detection rules
        if volatility > 0.02 and abs(trend_strength) < 0.1:
            return "volatile_ranging"
        elif trend_strength > 0.2:
            return "strong_uptrend"
        elif trend_strength < -0.2:
            return "strong_downtrend"
        elif abs(trend_strength) < 0.05:
            return "ranging"
        else:
            return "weak_trend"
    
    def calculate_risk_metrics(
        self,
        returns: np.ndarray,
        benchmark_returns: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Calculate various risk metrics"""
        metrics = {}
        
        # Basic metrics
        metrics["volatility"] = np.std(returns)
        metrics["downside_volatility"] = np.std(returns[returns < 0])
        
        # Value at Risk (VaR)
        metrics["var_95"] = np.percentile(returns, 5)
        metrics["var_99"] = np.percentile(returns, 1)
        
        # Conditional Value at Risk (CVaR)
        metrics["cvar_95"] = np.mean(returns[returns <= metrics["var_95"]])
        metrics["cvar_99"] = np.mean(returns[returns <= metrics["var_99"]])
        
        # Maximum Drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        metrics["max_drawdown"] = np.min(drawdown)
        
        # Beta and Alpha (if benchmark provided)
        if benchmark_returns is not None and len(benchmark_returns) == len(returns):
            covariance = np.cov(returns, benchmark_returns)[0, 1]
            benchmark_variance = np.var(benchmark_returns)
            
            metrics["beta"] = covariance / benchmark_variance
            metrics["alpha"] = np.mean(returns) - metrics["beta"] * np.mean(benchmark_returns)
        
        return metrics
"""
Risk Assessment Engine with Advanced Analytics

Comprehensive risk analysis for financial portfolios using
GPU-accelerated Monte Carlo simulations and machine learning.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
import asyncio

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer
from aiq.digital_human.financial.financial_data_processor import (
    FinancialDataProcessor,
    MarketData,
    FinancialMetrics
)
from aiq.digital_human.financial.mcts_financial_analyzer import FinancialState


@dataclass
class RiskProfile:
    """Comprehensive risk profile"""
    portfolio_var: float  # Value at Risk
    portfolio_cvar: float  # Conditional VaR
    max_drawdown: float
    volatility: float
    beta: float
    correlation_risk: float
    concentration_risk: float
    liquidity_risk: float
    tail_risk: float
    black_swan_probability: float
    risk_rating: str  # low, medium, high, extreme


@dataclass
class StressTestResult:
    """Results from stress testing"""
    scenario_name: str
    portfolio_impact: float
    worst_case_loss: float
    recovery_time: int  # days
    affected_assets: List[str]
    recommendations: List[str]


@dataclass
class RiskMitigation:
    """Risk mitigation strategy"""
    strategy_type: str
    target_risk_level: float
    suggested_actions: List[Dict[str, Any]]
    expected_risk_reduction: float
    implementation_cost: float


class RiskAssessmentEngine:
    """
    Advanced risk assessment engine with GPU acceleration.
    Performs comprehensive risk analysis, stress testing, and mitigation.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        confidence_level: float = 0.95,
        monte_carlo_simulations: int = 10000,
        enable_ml_detection: bool = True
    ):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.confidence_level = confidence_level
        self.monte_carlo_simulations = monte_carlo_simulations
        self.enable_ml_detection = enable_ml_detection
        
        # Initialize components
        self.data_processor = FinancialDataProcessor(device=self.device)
        
        # GPU optimization
        if self.device == "cuda":
            self.tensor_optimizer = TensorCoreOptimizer()
            self._compile_risk_kernels()
        
        # Machine learning models for anomaly detection
        if self.enable_ml_detection:
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                n_estimators=100,
                random_state=42
            )
        
        # Stress test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
    
    def _compile_risk_kernels(self):
        """Compile CUDA kernels for risk calculations"""
        # Monte Carlo simulation kernel
        self.monte_carlo_kernel = """
        __global__ void monte_carlo_var(
            float* returns,
            float* portfolio_weights,
            float* random_numbers,
            float* simulation_results,
            int num_assets,
            int num_simulations,
            int time_horizon
        ) {
            int sim_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (sim_idx < num_simulations) {
                float portfolio_value = 1.0f;
                
                for (int t = 0; t < time_horizon; t++) {
                    float portfolio_return = 0.0f;
                    
                    for (int a = 0; a < num_assets; a++) {
                        int rand_idx = sim_idx * time_horizon * num_assets + t * num_assets + a;
                        float asset_return = returns[a] + random_numbers[rand_idx];
                        portfolio_return += portfolio_weights[a] * asset_return;
                    }
                    
                    portfolio_value *= (1.0f + portfolio_return);
                }
                
                simulation_results[sim_idx] = portfolio_value;
            }
        }
        """
        
        # Tail risk calculation kernel
        self.tail_risk_kernel = """
        __global__ void calculate_tail_risk(
            float* returns,
            float* tail_threshold,
            float* tail_risk_metrics,
            int num_samples
        ) {
            __shared__ float extreme_losses[256];
            int tid = threadIdx.x;
            int idx = blockIdx.x * blockDim.x + tid;
            
            extreme_losses[tid] = 0.0f;
            
            if (idx < num_samples) {
                float ret = returns[idx];
                if (ret < *tail_threshold) {
                    extreme_losses[tid] = ret;
                }
            }
            
            __syncthreads();
            
            // Reduction to find extreme values
            for (int s = blockDim.x / 2; s > 0; s >>= 1) {
                if (tid < s && extreme_losses[tid] > extreme_losses[tid + s]) {
                    extreme_losses[tid] = extreme_losses[tid + s];
                }
                __syncthreads();
            }
            
            if (tid == 0) {
                atomicAdd(&tail_risk_metrics[0], extreme_losses[0]);
                atomicAdd(&tail_risk_metrics[1], 1.0f);
            }
        }
        """
    
    def _initialize_stress_scenarios(self) -> List[Dict[str, Any]]:
        """Initialize stress test scenarios"""
        return [
            {
                "name": "Market Crash",
                "description": "Severe market downturn similar to 2008",
                "parameters": {
                    "market_drop": -0.40,
                    "volatility_spike": 3.0,
                    "correlation_increase": 0.8,
                    "duration_days": 90
                }
            },
            {
                "name": "Interest Rate Shock",
                "description": "Rapid interest rate increase",
                "parameters": {
                    "rate_increase": 0.03,
                    "bond_impact": -0.15,
                    "equity_impact": -0.10,
                    "duration_days": 180
                }
            },
            {
                "name": "Black Swan Event",
                "description": "Unexpected extreme event",
                "parameters": {
                    "probability": 0.001,
                    "impact_range": (-0.50, -0.30),
                    "recovery_time": 365,
                    "correlation_breakdown": True
                }
            },
            {
                "name": "Sector Collapse",
                "description": "Major sector experiencing collapse",
                "parameters": {
                    "affected_sectors": ["technology", "finance"],
                    "sector_drop": -0.60,
                    "contagion_factor": 0.3,
                    "duration_days": 60
                }
            },
            {
                "name": "Liquidity Crisis",
                "description": "Market-wide liquidity freeze",
                "parameters": {
                    "liquidity_reduction": 0.80,
                    "spread_widening": 5.0,
                    "forced_selling": 0.30,
                    "duration_days": 30
                }
            }
        ]
    
    async def assess_portfolio_risk(
        self,
        portfolio_state: FinancialState,
        market_data: Dict[str, List[MarketData]],
        lookback_period: int = 252
    ) -> RiskProfile:
        """
        Perform comprehensive risk assessment on portfolio.
        
        Args:
            portfolio_state: Current portfolio state
            market_data: Historical market data
            lookback_period: Days of history to analyze
            
        Returns:
            Comprehensive risk profile
        """
        # Prepare data
        returns_data = self._prepare_returns_data(market_data, lookback_period)
        
        # Calculate various risk metrics
        if self.device == "cuda":
            risk_metrics = await self._gpu_risk_assessment(
                portfolio_state,
                returns_data
            )
        else:
            risk_metrics = await self._cpu_risk_assessment(
                portfolio_state,
                returns_data
            )
        
        # Detect anomalies
        if self.enable_ml_detection:
            anomaly_score = await self._detect_anomalies(returns_data)
            risk_metrics["anomaly_score"] = anomaly_score
        
        # Calculate risk rating
        risk_rating = self._calculate_risk_rating(risk_metrics)
        
        return RiskProfile(
            portfolio_var=risk_metrics["var"],
            portfolio_cvar=risk_metrics["cvar"],
            max_drawdown=risk_metrics["max_drawdown"],
            volatility=risk_metrics["volatility"],
            beta=risk_metrics["beta"],
            correlation_risk=risk_metrics["correlation_risk"],
            concentration_risk=risk_metrics["concentration_risk"],
            liquidity_risk=risk_metrics["liquidity_risk"],
            tail_risk=risk_metrics["tail_risk"],
            black_swan_probability=risk_metrics["black_swan_probability"],
            risk_rating=risk_rating
        )
    
    def _prepare_returns_data(
        self,
        market_data: Dict[str, List[MarketData]],
        lookback_period: int
    ) -> Dict[str, np.ndarray]:
        """Prepare returns data for analysis"""
        returns_data = {}
        
        for symbol, data in market_data.items():
            prices = np.array([d.close for d in data[-lookback_period:]])
            returns = np.diff(prices) / prices[:-1]
            returns_data[symbol] = returns
        
        return returns_data
    
    async def _gpu_risk_assessment(
        self,
        portfolio_state: FinancialState,
        returns_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """GPU-accelerated risk assessment"""
        # Convert to tensors
        symbols = list(returns_data.keys())
        returns_matrix = torch.tensor(
            np.column_stack([returns_data[s] for s in symbols]),
            device=self.device,
            dtype=torch.float32
        )
        
        weights = torch.tensor(
            [portfolio_state.holdings.get(s, 0) for s in symbols],
            device=self.device,
            dtype=torch.float32
        )
        
        # Normalize weights
        total_value = sum(portfolio_state.holdings.values())
        weights = weights / total_value if total_value > 0 else weights
        
        # Portfolio returns
        portfolio_returns = torch.matmul(returns_matrix, weights)
        
        # VaR and CVaR
        var = self._calculate_var_gpu(portfolio_returns, self.confidence_level)
        cvar = self._calculate_cvar_gpu(portfolio_returns, var)
        
        # Maximum Drawdown
        max_drawdown = self._calculate_max_drawdown_gpu(portfolio_returns)
        
        # Volatility
        volatility = torch.std(portfolio_returns).item()
        
        # Beta (simplified - would need market returns)
        market_returns = torch.mean(returns_matrix, dim=1)
        covariance = torch.cov(torch.stack([portfolio_returns, market_returns]))
        beta = covariance[0, 1] / covariance[1, 1]
        
        # Correlation Risk
        correlation_matrix = torch.corrcoef(returns_matrix.T)
        correlation_risk = torch.std(correlation_matrix).item()
        
        # Concentration Risk
        concentration_risk = torch.max(weights).item()
        
        # Liquidity Risk (simplified)
        volumes = torch.ones_like(weights)  # Placeholder
        liquidity_risk = 1.0 / torch.mean(volumes).item()
        
        # Tail Risk
        tail_risk = await self._calculate_tail_risk_gpu(portfolio_returns)
        
        # Black Swan Probability
        black_swan_prob = await self._estimate_black_swan_probability_gpu(
            portfolio_returns
        )
        
        return {
            "var": var.item(),
            "cvar": cvar.item(),
            "max_drawdown": max_drawdown.item(),
            "volatility": volatility,
            "beta": beta.item(),
            "correlation_risk": correlation_risk,
            "concentration_risk": concentration_risk,
            "liquidity_risk": liquidity_risk,
            "tail_risk": tail_risk,
            "black_swan_probability": black_swan_prob
        }
    
    def _calculate_var_gpu(
        self,
        returns: torch.Tensor,
        confidence_level: float
    ) -> torch.Tensor:
        """Calculate Value at Risk on GPU"""
        percentile = (1 - confidence_level) * 100
        return torch.quantile(returns, percentile / 100)
    
    def _calculate_cvar_gpu(
        self,
        returns: torch.Tensor,
        var: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Conditional Value at Risk on GPU"""
        return torch.mean(returns[returns <= var])
    
    def _calculate_max_drawdown_gpu(
        self,
        returns: torch.Tensor
    ) -> torch.Tensor:
        """Calculate Maximum Drawdown on GPU"""
        cumulative = torch.cumprod(1 + returns, dim=0)
        running_max = torch.cummax(cumulative, dim=0)[0]
        drawdown = (cumulative - running_max) / running_max
        return torch.min(drawdown)
    
    async def _calculate_tail_risk_gpu(
        self,
        returns: torch.Tensor
    ) -> float:
        """Calculate tail risk metrics on GPU"""
        # Define tail as bottom 5%
        tail_threshold = torch.quantile(returns, 0.05)
        tail_returns = returns[returns <= tail_threshold]
        
        if len(tail_returns) == 0:
            return 0.0
        
        # Expected Shortfall in the tail
        tail_risk = torch.mean(tail_returns).item()
        
        # Tail index (simplified)
        if len(tail_returns) > 1:
            log_returns = torch.log(-tail_returns)
            tail_index = 1.0 / torch.std(log_returns).item()
        else:
            tail_index = 1.0
        
        return abs(tail_risk) * tail_index
    
    async def _estimate_black_swan_probability_gpu(
        self,
        returns: torch.Tensor
    ) -> float:
        """Estimate probability of black swan events"""
        # Define black swan as 4+ standard deviation event
        mean_return = torch.mean(returns)
        std_return = torch.std(returns)
        
        z_scores = (returns - mean_return) / std_return
        black_swan_count = torch.sum(torch.abs(z_scores) > 4).item()
        
        probability = black_swan_count / len(returns)
        
        # Adjust for fat tails using kurtosis
        kurtosis = self._calculate_kurtosis_gpu(returns)
        adjusted_probability = probability * (1 + max(0, kurtosis - 3) / 10)
        
        return min(adjusted_probability, 0.1)  # Cap at 10%
    
    def _calculate_kurtosis_gpu(self, returns: torch.Tensor) -> float:
        """Calculate kurtosis on GPU"""
        mean = torch.mean(returns)
        std = torch.std(returns)
        standardized = (returns - mean) / std
        return torch.mean(standardized ** 4).item()
    
    async def _cpu_risk_assessment(
        self,
        portfolio_state: FinancialState,
        returns_data: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """CPU fallback for risk assessment"""
        # Simplified calculations
        symbols = list(returns_data.keys())
        returns_matrix = np.column_stack([returns_data[s] for s in symbols])
        
        weights = np.array([
            portfolio_state.holdings.get(s, 0) for s in symbols
        ])
        
        # Normalize weights
        total_value = sum(portfolio_state.holdings.values())
        weights = weights / total_value if total_value > 0 else weights
        
        # Portfolio returns
        portfolio_returns = np.dot(returns_matrix, weights)
        
        # Basic risk metrics
        var = np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        cvar = np.mean(portfolio_returns[portfolio_returns <= var])
        volatility = np.std(portfolio_returns)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        return {
            "var": var,
            "cvar": cvar,
            "max_drawdown": max_drawdown,
            "volatility": volatility,
            "beta": 1.0,  # Placeholder
            "correlation_risk": 0.5,  # Placeholder
            "concentration_risk": np.max(weights),
            "liquidity_risk": 0.1,  # Placeholder
            "tail_risk": abs(cvar) * 2,  # Simplified
            "black_swan_probability": 0.01  # Placeholder
        }
    
    async def _detect_anomalies(
        self,
        returns_data: Dict[str, np.ndarray]
    ) -> float:
        """Detect anomalies using machine learning"""
        # Prepare data for anomaly detection
        all_returns = []
        for returns in returns_data.values():
            all_returns.extend(returns)
        
        if len(all_returns) < 100:
            return 0.0
        
        # Reshape for sklearn
        X = np.array(all_returns).reshape(-1, 1)
        
        # Fit and predict
        self.anomaly_detector.fit(X)
        anomaly_scores = self.anomaly_detector.decision_function(X)
        
        # Return percentage of anomalies
        anomaly_count = np.sum(anomaly_scores < 0)
        return anomaly_count / len(all_returns)
    
    def _calculate_risk_rating(self, risk_metrics: Dict[str, float]) -> str:
        """Calculate overall risk rating"""
        # Weighted scoring system
        scores = {
            "var": abs(risk_metrics["var"]) * 100,
            "volatility": risk_metrics["volatility"] * 100,
            "concentration": risk_metrics["concentration_risk"] * 50,
            "tail_risk": risk_metrics["tail_risk"] * 200,
            "black_swan": risk_metrics["black_swan_probability"] * 1000
        }
        
        total_score = sum(scores.values())
        
        if total_score < 10:
            return "low"
        elif total_score < 25:
            return "medium"
        elif total_score < 50:
            return "high"
        else:
            return "extreme"
    
    async def run_stress_tests(
        self,
        portfolio_state: FinancialState,
        market_data: Dict[str, List[MarketData]]
    ) -> List[StressTestResult]:
        """Run comprehensive stress tests on portfolio"""
        stress_results = []
        
        for scenario in self.stress_scenarios:
            result = await self._run_single_stress_test(
                portfolio_state,
                market_data,
                scenario
            )
            stress_results.append(result)
        
        return stress_results
    
    async def _run_single_stress_test(
        self,
        portfolio_state: FinancialState,
        market_data: Dict[str, List[MarketData]],
        scenario: Dict[str, Any]
    ) -> StressTestResult:
        """Run single stress test scenario"""
        scenario_name = scenario["name"]
        parameters = scenario["parameters"]
        
        # Simulate scenario impact
        if self.device == "cuda":
            impact_results = await self._gpu_stress_simulation(
                portfolio_state,
                market_data,
                parameters
            )
        else:
            impact_results = await self._cpu_stress_simulation(
                portfolio_state,
                market_data,
                parameters
            )
        
        # Generate recommendations
        recommendations = self._generate_stress_recommendations(
            scenario_name,
            impact_results
        )
        
        return StressTestResult(
            scenario_name=scenario_name,
            portfolio_impact=impact_results["portfolio_impact"],
            worst_case_loss=impact_results["worst_case_loss"],
            recovery_time=impact_results["recovery_time"],
            affected_assets=impact_results["affected_assets"],
            recommendations=recommendations
        )
    
    async def _gpu_stress_simulation(
        self,
        portfolio_state: FinancialState,
        market_data: Dict[str, List[MarketData]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """GPU-accelerated stress simulation"""
        # Prepare tensors
        symbols = list(market_data.keys())
        returns_data = self._prepare_returns_data(market_data, 252)
        
        returns_tensor = torch.tensor(
            np.column_stack([returns_data[s] for s in symbols]),
            device=self.device,
            dtype=torch.float32
        )
        
        # Apply stress scenario
        if "market_drop" in parameters:
            stressed_returns = returns_tensor * (1 + parameters["market_drop"])
        else:
            stressed_returns = returns_tensor
        
        if "volatility_spike" in parameters:
            volatility_mult = parameters["volatility_spike"]
            noise = torch.randn_like(stressed_returns) * volatility_mult
            stressed_returns += noise * torch.std(stressed_returns, dim=0)
        
        # Calculate portfolio impact
        weights = torch.tensor(
            [portfolio_state.holdings.get(s, 0) for s in symbols],
            device=self.device,
            dtype=torch.float32
        )
        weights = weights / torch.sum(weights)
        
        portfolio_returns = torch.matmul(stressed_returns, weights)
        
        # Calculate metrics
        portfolio_impact = torch.mean(portfolio_returns).item()
        worst_case_loss = torch.min(portfolio_returns).item()
        
        # Recovery time estimation
        recovery_periods = torch.where(
            torch.cumsum(portfolio_returns, dim=0) > 0
        )[0]
        recovery_time = recovery_periods[0].item() if len(recovery_periods) > 0 else 365
        
        # Affected assets
        asset_impacts = torch.mean(stressed_returns, dim=0)
        affected_mask = asset_impacts < -0.1
        affected_assets = [s for s, affected in zip(symbols, affected_mask) if affected]
        
        return {
            "portfolio_impact": portfolio_impact,
            "worst_case_loss": worst_case_loss,
            "recovery_time": recovery_time,
            "affected_assets": affected_assets
        }
    
    async def _cpu_stress_simulation(
        self,
        portfolio_state: FinancialState,
        market_data: Dict[str, List[MarketData]],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """CPU fallback for stress simulation"""
        # Simplified stress test
        baseline_value = portfolio_state.portfolio_value
        
        # Apply scenario parameters
        market_impact = parameters.get("market_drop", 0)
        worst_case_impact = market_impact * 1.5  # Amplified worst case
        
        portfolio_impact = baseline_value * market_impact
        worst_case_loss = baseline_value * worst_case_impact
        
        # Estimate recovery time
        recovery_time = parameters.get("duration_days", 90)
        
        # Affected assets (simplified)
        affected_assets = []
        if "affected_sectors" in parameters:
            # Would need sector mapping in production
            affected_assets = list(portfolio_state.holdings.keys())[:3]
        
        return {
            "portfolio_impact": portfolio_impact,
            "worst_case_loss": worst_case_loss,
            "recovery_time": recovery_time,
            "affected_assets": affected_assets
        }
    
    def _generate_stress_recommendations(
        self,
        scenario_name: str,
        impact_results: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on stress test results"""
        recommendations = []
        
        impact = impact_results["portfolio_impact"]
        worst_case = impact_results["worst_case_loss"]
        
        if worst_case < -0.20:
            recommendations.append("Consider reducing portfolio concentration")
            recommendations.append("Implement hedging strategies")
        
        if scenario_name == "Market Crash":
            recommendations.append("Increase allocation to defensive assets")
            recommendations.append("Consider put options for downside protection")
        
        elif scenario_name == "Interest Rate Shock":
            recommendations.append("Reduce duration in fixed income holdings")
            recommendations.append("Consider floating rate instruments")
        
        elif scenario_name == "Liquidity Crisis":
            recommendations.append("Maintain higher cash reserves")
            recommendations.append("Focus on highly liquid assets")
        
        # Recovery-based recommendations
        if impact_results["recovery_time"] > 180:
            recommendations.append("Implement systematic rebalancing strategy")
            recommendations.append("Consider dollar-cost averaging for recovery")
        
        return recommendations
    
    async def generate_mitigation_strategies(
        self,
        risk_profile: RiskProfile,
        portfolio_state: FinancialState
    ) -> List[RiskMitigation]:
        """Generate risk mitigation strategies"""
        strategies = []
        
        # VaR-based mitigation
        if abs(risk_profile.portfolio_var) > 0.10:
            strategies.append(await self._create_var_mitigation(
                risk_profile,
                portfolio_state
            ))
        
        # Concentration risk mitigation
        if risk_profile.concentration_risk > 0.30:
            strategies.append(await self._create_concentration_mitigation(
                risk_profile,
                portfolio_state
            ))
        
        # Correlation risk mitigation
        if risk_profile.correlation_risk > 0.7:
            strategies.append(await self._create_correlation_mitigation(
                risk_profile,
                portfolio_state
            ))
        
        # Tail risk mitigation
        if risk_profile.tail_risk > 0.15:
            strategies.append(await self._create_tail_risk_mitigation(
                risk_profile,
                portfolio_state
            ))
        
        return strategies
    
    async def _create_var_mitigation(
        self,
        risk_profile: RiskProfile,
        portfolio_state: FinancialState
    ) -> RiskMitigation:
        """Create VaR mitigation strategy"""
        current_var = abs(risk_profile.portfolio_var)
        target_var = 0.05  # 5% target VaR
        
        actions = []
        
        # Reduce high-risk positions
        for symbol, shares in portfolio_state.holdings.items():
            # Simplified logic - would need individual asset risk in production
            if shares * 100 > portfolio_state.portfolio_value * 0.20:
                reduction = 0.3  # Reduce by 30%
                actions.append({
                    "action": "reduce_position",
                    "symbol": symbol,
                    "percentage": reduction,
                    "reason": "High concentration contributing to VaR"
                })
        
        # Add diversification
        actions.append({
            "action": "add_diversification",
            "asset_class": "bonds",
            "percentage": 0.15,
            "reason": "Reduce portfolio volatility"
        })
        
        expected_risk_reduction = (current_var - target_var) / current_var
        
        return RiskMitigation(
            strategy_type="var_reduction",
            target_risk_level=target_var,
            suggested_actions=actions,
            expected_risk_reduction=expected_risk_reduction,
            implementation_cost=portfolio_state.portfolio_value * 0.005  # 0.5% cost
        )
    
    async def _create_concentration_mitigation(
        self,
        risk_profile: RiskProfile,
        portfolio_state: FinancialState
    ) -> RiskMitigation:
        """Create concentration risk mitigation strategy"""
        actions = []
        
        # Identify concentrated positions
        total_value = portfolio_state.portfolio_value
        for symbol, shares in portfolio_state.holdings.items():
            position_value = shares * 100  # Placeholder price
            concentration = position_value / total_value
            
            if concentration > 0.15:  # 15% threshold
                excess = concentration - 0.10
                actions.append({
                    "action": "rebalance",
                    "symbol": symbol,
                    "target_weight": 0.10,
                    "current_weight": concentration,
                    "reason": "Reduce concentration risk"
                })
        
        return RiskMitigation(
            strategy_type="concentration_reduction",
            target_risk_level=0.10,
            suggested_actions=actions,
            expected_risk_reduction=0.30,
            implementation_cost=portfolio_state.portfolio_value * 0.002
        )
    
    async def _create_correlation_mitigation(
        self,
        risk_profile: RiskProfile,
        portfolio_state: FinancialState
    ) -> RiskMitigation:
        """Create correlation risk mitigation strategy"""
        actions = [
            {
                "action": "add_uncorrelated_assets",
                "asset_types": ["commodities", "real_estate", "alternatives"],
                "percentage": 0.20,
                "reason": "Reduce portfolio correlation"
            },
            {
                "action": "sector_diversification",
                "sectors": ["utilities", "consumer_staples", "healthcare"],
                "percentage": 0.15,
                "reason": "Add defensive sectors"
            }
        ]
        
        return RiskMitigation(
            strategy_type="correlation_reduction",
            target_risk_level=0.5,
            suggested_actions=actions,
            expected_risk_reduction=0.25,
            implementation_cost=portfolio_state.portfolio_value * 0.003
        )
    
    async def _create_tail_risk_mitigation(
        self,
        risk_profile: RiskProfile,
        portfolio_state: FinancialState
    ) -> RiskMitigation:
        """Create tail risk mitigation strategy"""
        actions = [
            {
                "action": "implement_options_strategy",
                "strategy": "protective_puts",
                "coverage": 0.80,  # 80% of portfolio
                "strike": "10%_out_of_money",
                "reason": "Protect against extreme downside"
            },
            {
                "action": "add_tail_hedge",
                "instrument": "vix_futures",
                "percentage": 0.05,
                "reason": "Volatility hedge for tail events"
            },
            {
                "action": "increase_cash_buffer",
                "percentage": 0.10,
                "reason": "Liquidity for extreme scenarios"
            }
        ]
        
        return RiskMitigation(
            strategy_type="tail_risk_hedging",
            target_risk_level=0.05,
            suggested_actions=actions,
            expected_risk_reduction=0.50,
            implementation_cost=portfolio_state.portfolio_value * 0.015  # Options premium
        )
    
    def calculate_risk_adjusted_returns(
        self,
        returns: np.ndarray,
        risk_free_rate: float = 0.03
    ) -> Dict[str, float]:
        """Calculate risk-adjusted return metrics"""
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        
        metrics = {}
        
        # Sharpe Ratio
        metrics["sharpe_ratio"] = (
            np.mean(excess_returns) / np.std(returns) * np.sqrt(252)
        )
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1
        metrics["sortino_ratio"] = (
            np.mean(excess_returns) / downside_std * np.sqrt(252)
        )
        
        # Calmar Ratio
        max_dd = self._calculate_max_drawdown_cpu(returns)
        metrics["calmar_ratio"] = (
            np.mean(excess_returns) * 252 / abs(max_dd)
            if max_dd != 0 else 0
        )
        
        # Information Ratio (would need benchmark)
        metrics["information_ratio"] = metrics["sharpe_ratio"]  # Simplified
        
        return metrics
    
    def _calculate_max_drawdown_cpu(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown on CPU"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return np.min(drawdown)
"""
Portfolio Optimizer with GPU Acceleration

Implements modern portfolio theory with CUDA optimization for
efficient frontier calculation and risk management.
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
import asyncio

from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer
from aiq.digital_human.financial.mcts_financial_analyzer import FinancialState


@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0
    max_weight: float = 1.0
    target_return: Optional[float] = None
    max_risk: Optional[float] = None
    sector_limits: Optional[Dict[str, float]] = None
    concentration_limit: float = 0.4


@dataclass
class OptimizationResult:
    """Results from portfolio optimization"""
    weights: Dict[str, float]
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    efficient_frontier: List[Tuple[float, float]]
    risk_contributions: Dict[str, float]


class PortfolioOptimizer:
    """
    GPU-accelerated portfolio optimization using modern portfolio theory.
    Implements Markowitz mean-variance optimization with extensions.
    """
    
    def __init__(
        self,
        device: str = "cuda",
        risk_free_rate: float = 0.03,
        enable_gpu: bool = True
    ):
        self.device = device if torch.cuda.is_available() and enable_gpu else "cpu"
        self.risk_free_rate = risk_free_rate
        
        # Initialize GPU optimization
        if self.device == "cuda":
            self.tensor_optimizer = TensorCoreOptimizer()
        else:
            self.tensor_optimizer = None
        
        # Pre-compiled CUDA kernels for optimization
        if self.device == "cuda":
            self._compile_optimization_kernels()
    
    def _compile_optimization_kernels(self):
        """Compile CUDA kernels for portfolio optimization"""
        # Covariance matrix calculation
        self.cov_kernel = """
        __global__ void calculate_covariance(
            float* returns,
            float* covariance,
            int num_assets,
            int num_samples
        ) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            int j = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (i < num_assets && j < num_assets) {
                float cov = 0.0f;
                float mean_i = 0.0f;
                float mean_j = 0.0f;
                
                // Calculate means
                for (int k = 0; k < num_samples; k++) {
                    mean_i += returns[k * num_assets + i];
                    mean_j += returns[k * num_assets + j];
                }
                mean_i /= num_samples;
                mean_j /= num_samples;
                
                // Calculate covariance
                for (int k = 0; k < num_samples; k++) {
                    float diff_i = returns[k * num_assets + i] - mean_i;
                    float diff_j = returns[k * num_assets + j] - mean_j;
                    cov += diff_i * diff_j;
                }
                
                covariance[i * num_assets + j] = cov / (num_samples - 1);
            }
        }
        """
        
        # Efficient frontier calculation
        self.frontier_kernel = """
        __global__ void efficient_frontier(
            float* expected_returns,
            float* covariance,
            float* weights,
            float* frontier_returns,
            float* frontier_risks,
            int num_assets,
            int num_points
        ) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (idx < num_points) {
                // Calculate portfolio for this point on frontier
                float target_return = frontier_returns[idx];
                
                // Simplified optimization (would use more sophisticated method)
                for (int i = 0; i < num_assets; i++) {
                    weights[idx * num_assets + i] = 1.0f / num_assets;
                }
                
                // Calculate risk
                float risk = 0.0f;
                for (int i = 0; i < num_assets; i++) {
                    for (int j = 0; j < num_assets; j++) {
                        float w_i = weights[idx * num_assets + i];
                        float w_j = weights[idx * num_assets + j];
                        risk += w_i * w_j * covariance[i * num_assets + j];
                    }
                }
                
                frontier_risks[idx] = sqrtf(risk);
            }
        }
        """
    
    async def optimize_portfolio(
        self,
        assets: List[str],
        historical_returns: np.ndarray,
        current_prices: Dict[str, float],
        constraints: OptimizationConstraints,
        optimization_method: str = "mean_variance"
    ) -> OptimizationResult:
        """
        Optimize portfolio allocation using specified method.
        
        Args:
            assets: List of asset symbols
            historical_returns: Historical return data (time x assets)
            current_prices: Current asset prices
            constraints: Optimization constraints
            optimization_method: Method to use
            
        Returns:
            Optimization results with weights and metrics
        """
        if self.device == "cuda":
            return await self._gpu_optimize(
                assets,
                historical_returns,
                current_prices,
                constraints,
                optimization_method
            )
        else:
            return await self._cpu_optimize(
                assets,
                historical_returns,
                current_prices,
                constraints,
                optimization_method
            )
    
    async def _gpu_optimize(
        self,
        assets: List[str],
        historical_returns: np.ndarray,
        current_prices: Dict[str, float],
        constraints: OptimizationConstraints,
        method: str
    ) -> OptimizationResult:
        """GPU-accelerated portfolio optimization"""
        # Convert to tensors
        returns_tensor = torch.tensor(
            historical_returns,
            device=self.device,
            dtype=torch.float32
        )
        
        # Calculate expected returns and covariance
        expected_returns = torch.mean(returns_tensor, dim=0)
        covariance_matrix = torch.cov(returns_tensor.T)
        
        # Optimization based on method
        if method == "mean_variance":
            weights = await self._mean_variance_optimization_gpu(
                expected_returns,
                covariance_matrix,
                constraints
            )
        elif method == "maximum_sharpe":
            weights = await self._max_sharpe_optimization_gpu(
                expected_returns,
                covariance_matrix,
                constraints
            )
        elif method == "minimum_variance":
            weights = await self._min_variance_optimization_gpu(
                covariance_matrix,
                constraints
            )
        elif method == "risk_parity":
            weights = await self._risk_parity_optimization_gpu(
                covariance_matrix,
                constraints
            )
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate portfolio metrics
        portfolio_return = torch.sum(weights * expected_returns)
        portfolio_variance = torch.sum(weights @ covariance_matrix @ weights)
        portfolio_risk = torch.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Calculate efficient frontier
        efficient_frontier = await self._calculate_efficient_frontier_gpu(
            expected_returns,
            covariance_matrix
        )
        
        # Risk contributions
        risk_contributions = self._calculate_risk_contributions_gpu(
            weights,
            covariance_matrix
        )
        
        # Convert results
        weight_dict = {
            asset: weight.item()
            for asset, weight in zip(assets, weights)
        }
        
        return OptimizationResult(
            weights=weight_dict,
            expected_return=portfolio_return.item(),
            portfolio_risk=portfolio_risk.item(),
            sharpe_ratio=sharpe_ratio.item(),
            efficient_frontier=efficient_frontier,
            risk_contributions={
                asset: contrib.item()
                for asset, contrib in zip(assets, risk_contributions)
            }
        )
    
    async def _mean_variance_optimization_gpu(
        self,
        expected_returns: torch.Tensor,
        covariance_matrix: torch.Tensor,
        constraints: OptimizationConstraints
    ) -> torch.Tensor:
        """GPU mean-variance optimization"""
        n_assets = len(expected_returns)
        
        # Initialize optimization problem
        class PortfolioOptimizer(nn.Module):
            def __init__(self, n_assets):
                super().__init__()
                self.weights = nn.Parameter(
                    torch.ones(n_assets, device=expected_returns.device) / n_assets
                )
            
            def forward(self):
                # Normalize weights
                w = torch.softmax(self.weights, dim=0)
                return w
        
        model = PortfolioOptimizer(n_assets)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # Optimization loop
        for _ in range(1000):
            optimizer.zero_grad()
            
            # Get normalized weights
            weights = model()
            
            # Calculate portfolio metrics
            portfolio_return = torch.sum(weights * expected_returns)
            portfolio_variance = torch.sum(weights @ covariance_matrix @ weights)
            
            # Objective: maximize return - lambda * variance
            risk_aversion = 2.0  # Risk aversion parameter
            objective = -portfolio_return + risk_aversion * portfolio_variance
            
            # Add constraint penalties
            if constraints.target_return:
                return_penalty = 100 * (portfolio_return - constraints.target_return) ** 2
                objective += return_penalty
            
            if constraints.max_risk:
                risk = torch.sqrt(portfolio_variance)
                risk_penalty = 100 * torch.relu(risk - constraints.max_risk) ** 2
                objective += risk_penalty
            
            # Concentration penalty
            concentration_penalty = 10 * torch.relu(
                torch.max(weights) - constraints.concentration_limit
            ) ** 2
            objective += concentration_penalty
            
            objective.backward()
            optimizer.step()
        
        return model()
    
    async def _max_sharpe_optimization_gpu(
        self,
        expected_returns: torch.Tensor,
        covariance_matrix: torch.Tensor,
        constraints: OptimizationConstraints
    ) -> torch.Tensor:
        """Maximize Sharpe ratio using GPU"""
        n_assets = len(expected_returns)
        
        # Use gradient-based optimization
        weights = torch.ones(n_assets, device=self.device) / n_assets
        weights.requires_grad_(True)
        
        optimizer = torch.optim.LBFGS([weights], lr=0.1)
        
        def closure():
            optimizer.zero_grad()
            
            # Normalize weights
            w = torch.softmax(weights, dim=0)
            
            # Portfolio metrics
            portfolio_return = torch.sum(w * expected_returns)
            portfolio_variance = torch.sum(w @ covariance_matrix @ w)
            portfolio_std = torch.sqrt(portfolio_variance)
            
            # Negative Sharpe ratio (minimize)
            sharpe = -(portfolio_return - self.risk_free_rate) / portfolio_std
            
            sharpe.backward()
            return sharpe
        
        # Optimize
        for _ in range(10):
            optimizer.step(closure)
        
        return torch.softmax(weights, dim=0)
    
    async def _min_variance_optimization_gpu(
        self,
        covariance_matrix: torch.Tensor,
        constraints: OptimizationConstraints
    ) -> torch.Tensor:
        """Minimum variance portfolio using GPU"""
        n_assets = covariance_matrix.shape[0]
        
        # Analytical solution for unconstrained case
        inv_cov = torch.inverse(covariance_matrix)
        ones = torch.ones(n_assets, device=self.device)
        
        weights = inv_cov @ ones
        weights = weights / torch.sum(weights)
        
        # Apply constraints
        weights = torch.clamp(weights, constraints.min_weight, constraints.max_weight)
        weights = weights / torch.sum(weights)
        
        return weights
    
    async def _risk_parity_optimization_gpu(
        self,
        covariance_matrix: torch.Tensor,
        constraints: OptimizationConstraints
    ) -> torch.Tensor:
        """Risk parity optimization using GPU"""
        n_assets = covariance_matrix.shape[0]
        
        # Initialize equal weights
        weights = torch.ones(n_assets, device=self.device) / n_assets
        
        # Iterative optimization
        for _ in range(100):
            # Calculate marginal risk contributions
            portfolio_risk = torch.sqrt(weights @ covariance_matrix @ weights)
            marginal_contrib = covariance_matrix @ weights / portfolio_risk
            
            # Update weights to equalize risk contributions
            weights = weights * (1.0 / marginal_contrib)
            weights = weights / torch.sum(weights)
            
            # Apply constraints
            weights = torch.clamp(weights, constraints.min_weight, constraints.max_weight)
            weights = weights / torch.sum(weights)
        
        return weights
    
    async def _calculate_efficient_frontier_gpu(
        self,
        expected_returns: torch.Tensor,
        covariance_matrix: torch.Tensor,
        n_points: int = 100
    ) -> List[Tuple[float, float]]:
        """Calculate efficient frontier using GPU"""
        min_return = torch.min(expected_returns).item()
        max_return = torch.max(expected_returns).item()
        
        target_returns = torch.linspace(min_return, max_return, n_points, device=self.device)
        frontier_points = []
        
        for target_return in target_returns:
            # Optimize for each target return
            constraints = OptimizationConstraints(target_return=target_return.item())
            weights = await self._mean_variance_optimization_gpu(
                expected_returns,
                covariance_matrix,
                constraints
            )
            
            # Calculate risk
            portfolio_variance = weights @ covariance_matrix @ weights
            portfolio_risk = torch.sqrt(portfolio_variance).item()
            
            frontier_points.append((portfolio_risk, target_return.item()))
        
        return frontier_points
    
    def _calculate_risk_contributions_gpu(
        self,
        weights: torch.Tensor,
        covariance_matrix: torch.Tensor
    ) -> torch.Tensor:
        """Calculate risk contribution of each asset"""
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_risk = torch.sqrt(portfolio_variance)
        
        marginal_contrib = covariance_matrix @ weights / portfolio_risk
        risk_contributions = weights * marginal_contrib
        
        return risk_contributions / torch.sum(risk_contributions)
    
    async def _cpu_optimize(
        self,
        assets: List[str],
        historical_returns: np.ndarray,
        current_prices: Dict[str, float],
        constraints: OptimizationConstraints,
        method: str
    ) -> OptimizationResult:
        """CPU fallback for portfolio optimization"""
        # Calculate expected returns and covariance
        expected_returns = np.mean(historical_returns, axis=0)
        covariance_matrix = np.cov(historical_returns.T)
        
        # Optimization
        n_assets = len(assets)
        initial_weights = np.ones(n_assets) / n_assets
        
        # Constraints for scipy.optimize
        bounds = [(constraints.min_weight, constraints.max_weight)] * n_assets
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if constraints.target_return:
            constraints_list.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * expected_returns) - constraints.target_return
            })
        
        # Objective function
        if method == "mean_variance":
            def objective(w):
                return -np.sum(w * expected_returns) + 2 * np.sqrt(w @ covariance_matrix @ w)
        elif method == "maximum_sharpe":
            def objective(w):
                ret = np.sum(w * expected_returns)
                risk = np.sqrt(w @ covariance_matrix @ w)
                return -(ret - self.risk_free_rate) / risk
        elif method == "minimum_variance":
            def objective(w):
                return w @ covariance_matrix @ w
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Optimize
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )
        
        weights = result.x
        
        # Calculate metrics
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_variance = weights @ covariance_matrix @ weights
        portfolio_risk = np.sqrt(portfolio_variance)
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
        
        # Simple efficient frontier
        frontier_points = []
        for i in range(20):
            target_ret = min(expected_returns) + i * (max(expected_returns) - min(expected_returns)) / 19
            
            def variance_objective(w):
                return w @ covariance_matrix @ w
            
            temp_constraints = constraints_list + [{
                'type': 'eq',
                'fun': lambda w: np.sum(w * expected_returns) - target_ret
            }]
            
            temp_result = minimize(
                variance_objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=temp_constraints
            )
            
            if temp_result.success:
                risk = np.sqrt(temp_result.fun)
                frontier_points.append((risk, target_ret))
        
        # Risk contributions
        portfolio_risk_scalar = np.sqrt(portfolio_variance)
        marginal_contrib = covariance_matrix @ weights / portfolio_risk_scalar
        risk_contributions = weights * marginal_contrib
        risk_contributions = risk_contributions / np.sum(risk_contributions)
        
        return OptimizationResult(
            weights=dict(zip(assets, weights)),
            expected_return=portfolio_return,
            portfolio_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            efficient_frontier=frontier_points,
            risk_contributions=dict(zip(assets, risk_contributions))
        )
    
    def calculate_portfolio_metrics(
        self,
        weights: Dict[str, float],
        returns: np.ndarray,
        asset_names: List[str]
    ) -> Dict[str, float]:
        """Calculate various portfolio performance metrics"""
        # Convert weights to array
        weight_array = np.array([weights[asset] for asset in asset_names])
        
        # Basic metrics
        portfolio_returns = returns @ weight_array
        expected_return = np.mean(portfolio_returns)
        portfolio_std = np.std(portfolio_returns)
        
        # Risk metrics
        sharpe_ratio = (expected_return - self.risk_free_rate) / portfolio_std
        sortino_ratio = (expected_return - self.risk_free_rate) / np.std(portfolio_returns[portfolio_returns < 0])
        
        # Drawdown
        cumulative = np.cumprod(1 + portfolio_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Value at Risk (VaR) and CVaR
        var_95 = np.percentile(portfolio_returns, 5)
        cvar_95 = np.mean(portfolio_returns[portfolio_returns <= var_95])
        
        return {
            "expected_return": expected_return,
            "volatility": portfolio_std,
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "cvar_95": cvar_95
        }
    
    def rebalance_portfolio(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        current_prices: Dict[str, float],
        portfolio_value: float,
        transaction_costs: float = 0.001
    ) -> Dict[str, Dict[str, float]]:
        """Calculate rebalancing trades"""
        trades = {}
        
        for asset in target_weights:
            current_value = current_weights.get(asset, 0) * portfolio_value
            target_value = target_weights[asset] * portfolio_value
            
            difference = target_value - current_value
            if abs(difference) > portfolio_value * 0.01:  # 1% threshold
                shares_to_trade = difference / current_prices[asset]
                cost = abs(difference) * transaction_costs
                
                trades[asset] = {
                    "action": "buy" if difference > 0 else "sell",
                    "shares": abs(shares_to_trade),
                    "value": abs(difference),
                    "cost": cost
                }
        
        return trades
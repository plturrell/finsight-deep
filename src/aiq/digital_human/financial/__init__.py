"""
AIQ Toolkit Digital Human Financial Analysis Module

Provides GPU-accelerated financial analysis capabilities including
portfolio optimization, risk assessment, and Monte Carlo simulations.
"""

from aiq.digital_human.financial.mcts_financial_analyzer import (
    MCTSFinancialAnalyzer,
    FinancialState,
    FinancialAction,
    MCTSNode
)

from aiq.digital_human.financial.portfolio_optimizer import (
    PortfolioOptimizer,
    OptimizationConstraints,
    OptimizationResult
)

from aiq.digital_human.financial.financial_data_processor import (
    FinancialDataProcessor,
    MarketData,
    FinancialMetrics,
    MarketIndicators
)

from aiq.digital_human.financial.risk_assessment_engine import (
    RiskAssessmentEngine,
    RiskProfile,
    StressTestResult,
    RiskMitigation
)

__all__ = [
    # MCTS Analyzer
    'MCTSFinancialAnalyzer',
    'FinancialState',
    'FinancialAction',
    'MCTSNode',
    
    # Portfolio Optimizer
    'PortfolioOptimizer',
    'OptimizationConstraints',
    'OptimizationResult',
    
    # Data Processor
    'FinancialDataProcessor',
    'MarketData',
    'FinancialMetrics',
    'MarketIndicators',
    
    # Risk Assessment
    'RiskAssessmentEngine',
    'RiskProfile',
    'StressTestResult',
    'RiskMitigation'
]

# Module version
__version__ = "1.0.0"
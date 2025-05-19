"""
Comprehensive tests for MCTS Financial Analyzer

Tests Monte Carlo Tree Search implementation, GPU acceleration,
and financial decision making capabilities.
"""

import pytest
import numpy as np
import torch
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from aiq.digital_human.financial.mcts_financial_analyzer import (
    MCTSFinancialAnalyzer,
    FinancialState,
    FinancialAction,
    MCTSNode
)


class TestFinancialState:
    """Test FinancialState dataclass"""
    
    def test_financial_state_creation(self):
        """Test creating a financial state"""
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100, "GOOGL": 50},
            cash_balance=20000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={"volatility": 0.2},
            timestamp=datetime.now()
        )
        
        assert state.portfolio_value == 100000.0
        assert state.holdings["AAPL"] == 100
        assert state.cash_balance == 20000.0
        assert state.risk_tolerance == 0.7
        assert state.time_horizon == 365
        assert "volatility" in state.market_conditions
    
    def test_financial_state_defaults(self):
        """Test financial state with minimal parameters"""
        state = FinancialState(
            portfolio_value=50000.0,
            holdings={},
            cash_balance=50000.0,
            risk_tolerance=0.5,
            time_horizon=180,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        assert state.portfolio_value == 50000.0
        assert len(state.holdings) == 0
        assert state.cash_balance == 50000.0


class TestFinancialAction:
    """Test FinancialAction dataclass"""
    
    def test_financial_action_creation(self):
        """Test creating a financial action"""
        action = FinancialAction(
            action_type="buy",
            symbol="AAPL",
            quantity=10.0,
            price=150.0,
            confidence=0.85,
            reasoning="Strong fundamentals"
        )
        
        assert action.action_type == "buy"
        assert action.symbol == "AAPL"
        assert action.quantity == 10.0
        assert action.price == 150.0
        assert action.confidence == 0.85
        assert action.reasoning == "Strong fundamentals"
    
    def test_financial_action_defaults(self):
        """Test financial action with defaults"""
        action = FinancialAction(action_type="hold")
        
        assert action.action_type == "hold"
        assert action.symbol is None
        assert action.quantity is None
        assert action.price is None
        assert action.confidence == 0.0
        assert action.reasoning == ""


class TestMCTSNode:
    """Test MCTS Node implementation"""
    
    def test_node_creation(self):
        """Test creating an MCTS node"""
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100},
            cash_balance=20000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        node = MCTSNode(state)
        
        assert node.state == state
        assert node.parent is None
        assert node.action is None
        assert len(node.children) == 0
        assert node.visits == 0
        assert node.total_reward == 0.0
        assert len(node.untried_actions) == 0
    
    def test_node_with_parent(self):
        """Test creating a child node"""
        parent_state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100},
            cash_balance=20000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        child_state = FinancialState(
            portfolio_value=101000.0,
            holdings={"AAPL": 110},
            cash_balance=18500.0,
            risk_tolerance=0.7,
            time_horizon=364,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        action = FinancialAction(
            action_type="buy",
            symbol="AAPL",
            quantity=10,
            price=150.0
        )
        
        parent = MCTSNode(parent_state)
        child = MCTSNode(child_state, parent=parent, action=action)
        
        assert child.parent == parent
        assert child.action == action
        assert child.state == child_state
    
    def test_uct_score(self):
        """Test UCT score calculation"""
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={},
            cash_balance=100000.0,
            risk_tolerance=0.5,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        parent = MCTSNode(state)
        child = MCTSNode(state, parent=parent)
        
        # Test unvisited node
        assert child.uct_score == float('inf')
        
        # Test visited node
        parent.visits = 10
        child.visits = 5
        child.total_reward = 2.5
        
        expected_exploitation = 2.5 / 5
        expected_exploration = np.sqrt(2 * np.log(10) / 5)
        expected_uct = expected_exploitation + expected_exploration
        
        assert abs(child.uct_score - expected_uct) < 1e-6
    
    def test_best_child(self):
        """Test best child selection"""
        parent_state = FinancialState(
            portfolio_value=100000.0,
            holdings={},
            cash_balance=100000.0,
            risk_tolerance=0.5,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        parent = MCTSNode(parent_state)
        parent.visits = 100
        
        # Create children with different scores
        children = []
        for i in range(3):
            child = MCTSNode(parent_state, parent=parent)
            child.visits = 10 + i * 5
            child.total_reward = 5.0 + i * 2
            parent.children.append(child)
            children.append(child)
        
        best = parent.best_child(c_param=1.414)
        
        # Verify best child is selected based on UCT
        uct_scores = []
        for child in children:
            exploitation = child.total_reward / child.visits
            exploration = 1.414 * np.sqrt(2 * np.log(parent.visits) / child.visits)
            uct_scores.append(exploitation + exploration)
        
        best_idx = np.argmax(uct_scores)
        assert best == children[best_idx]
    
    def test_expand(self):
        """Test node expansion"""
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100},
            cash_balance=20000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        node = MCTSNode(state)
        
        # Add untried actions
        actions = [
            FinancialAction(action_type="buy", symbol="GOOGL"),
            FinancialAction(action_type="sell", symbol="AAPL"),
            FinancialAction(action_type="hold")
        ]
        node.untried_actions = actions.copy()
        
        # Expand with first action
        new_state = FinancialState(
            portfolio_value=101000.0,
            holdings={"AAPL": 100, "GOOGL": 10},
            cash_balance=18000.0,
            risk_tolerance=0.7,
            time_horizon=364,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        child = node.expand(actions[0], new_state)
        
        assert child.parent == node
        assert child.action == actions[0]
        assert child.state == new_state
        assert len(node.children) == 1
        assert len(node.untried_actions) == 2
        assert actions[0] not in node.untried_actions


class TestMCTSFinancialAnalyzer:
    """Test MCTS Financial Analyzer implementation"""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance"""
        return MCTSFinancialAnalyzer(
            device="cpu",  # Use CPU for tests
            num_simulations=100,
            max_depth=5,
            enable_gpu_optimization=False
        )
    
    @pytest.fixture
    def financial_state(self):
        """Create test financial state"""
        return FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100, "GOOGL": 50},
            cash_balance=20000.0,
            risk_tolerance=0.7,
            time_horizon=365,
            market_conditions={"volatility": 0.2},
            timestamp=datetime.now()
        )
    
    @pytest.fixture
    def available_actions(self):
        """Create test actions"""
        return [
            FinancialAction(
                action_type="buy",
                symbol="MSFT",
                quantity=50,
                price=380.0,
                confidence=0.0,
                reasoning=""
            ),
            FinancialAction(
                action_type="sell",
                symbol="AAPL",
                quantity=20,
                price=175.0,
                confidence=0.0,
                reasoning=""
            ),
            FinancialAction(
                action_type="hold",
                confidence=0.0,
                reasoning=""
            ),
            FinancialAction(
                action_type="rebalance",
                confidence=0.0,
                reasoning=""
            )
        ]
    
    def test_analyzer_initialization(self, analyzer):
        """Test analyzer initialization"""
        assert analyzer.device == "cpu"
        assert analyzer.num_simulations == 100
        assert analyzer.max_depth == 5
        assert analyzer.use_cuda is False
        assert analyzer.risk_free_rate == 0.03
        assert analyzer.market_volatility == 0.16
    
    def test_analyzer_gpu_initialization(self):
        """Test analyzer with GPU initialization"""
        if torch.cuda.is_available():
            analyzer = MCTSFinancialAnalyzer(
                device="cuda",
                enable_gpu_optimization=True
            )
            
            assert analyzer.device == "cuda"
            assert analyzer.use_cuda is True
            assert analyzer.tensor_optimizer is not None
    
    @pytest.mark.asyncio
    async def test_analyze_portfolio(self, analyzer, financial_state, available_actions):
        """Test portfolio analysis"""
        # Mock internal methods for unit testing
        analyzer._select_node = Mock(return_value=MCTSNode(financial_state))
        analyzer._simulate_action = AsyncMock(return_value=financial_state)
        analyzer._simulate_playout = AsyncMock(return_value=0.05)
        analyzer._backpropagate = Mock()
        analyzer._get_best_action = Mock(return_value=available_actions[2])  # hold
        analyzer._generate_analysis = AsyncMock(return_value={
            "recommendation": {
                "action": "hold",
                "symbol": None,
                "quantity": None,
                "confidence": 0.85,
                "reasoning": "Market conditions stable"
            },
            "current_state": {
                "portfolio_value": 100000.0,
                "cash_balance": 20000.0,
                "holdings": {"AAPL": 100, "GOOGL": 50},
                "risk_tolerance": 0.7
            },
            "risk_analysis": {
                "portfolio_volatility": 0.18,
                "value_at_risk_95": 5000.0,
                "sharpe_ratio": 0.85,
                "beta": 1.1,
                "max_drawdown": 0.15
            },
            "performance_projection": {
                "30_days": {"expected_return": 0.02},
                "90_days": {"expected_return": 0.06},
                "365_days": {"expected_return": 0.08}
            },
            "alternative_actions": [],
            "optimization_goal": "maximize_return",
            "simulation_details": {
                "num_simulations": 100,
                "tree_depth": 3,
                "nodes_explored": 150
            },
            "market_conditions": {"volatility": 0.2},
            "timestamp": datetime.now().isoformat()
        })
        
        result = await analyzer.analyze_portfolio(
            financial_state,
            available_actions,
            optimization_goal="maximize_return"
        )
        
        assert result["recommendation"]["action"] == "hold"
        assert result["current_state"]["portfolio_value"] == 100000.0
        assert "risk_analysis" in result
        assert "performance_projection" in result
        assert result["optimization_goal"] == "maximize_return"
    
    def test_select_node(self, analyzer, financial_state):
        """Test node selection in MCTS"""
        root = MCTSNode(financial_state)
        
        # Test leaf node selection
        selected = analyzer._select_node(root)
        assert selected == root
        
        # Test with children
        child1 = MCTSNode(financial_state, parent=root)
        child2 = MCTSNode(financial_state, parent=root)
        root.children = [child1, child2]
        
        child1.visits = 10
        child1.total_reward = 5.0
        child2.visits = 5
        child2.total_reward = 3.0
        
        # Add untried actions to root
        root.untried_actions = [FinancialAction(action_type="hold")]
        
        selected = analyzer._select_node(root)
        assert selected == root  # Should return root due to untried actions
    
    @pytest.mark.asyncio
    async def test_simulate_action_buy(self, analyzer, financial_state):
        """Test simulating buy action"""
        action = FinancialAction(
            action_type="buy",
            symbol="MSFT",
            quantity=50,
            price=380.0
        )
        
        new_state = await analyzer._simulate_action(financial_state, action)
        
        assert new_state.holdings["MSFT"] == 50
        assert new_state.cash_balance == financial_state.cash_balance - (50 * 380)
        assert new_state.holdings["AAPL"] == financial_state.holdings["AAPL"]
        assert new_state.holdings["GOOGL"] == financial_state.holdings["GOOGL"]
    
    @pytest.mark.asyncio
    async def test_simulate_action_sell(self, analyzer, financial_state):
        """Test simulating sell action"""
        action = FinancialAction(
            action_type="sell",
            symbol="AAPL",
            quantity=20,
            price=175.0
        )
        
        new_state = await analyzer._simulate_action(financial_state, action)
        
        assert new_state.holdings["AAPL"] == 80  # 100 - 20
        assert new_state.cash_balance == financial_state.cash_balance + (20 * 175)
        assert new_state.holdings["GOOGL"] == financial_state.holdings["GOOGL"]
    
    @pytest.mark.asyncio
    async def test_simulate_action_insufficient_funds(self, analyzer, financial_state):
        """Test buy action with insufficient funds"""
        action = FinancialAction(
            action_type="buy",
            symbol="MSFT",
            quantity=1000,  # Too many shares
            price=380.0
        )
        
        new_state = await analyzer._simulate_action(financial_state, action)
        
        # Should not execute due to insufficient funds
        assert "MSFT" not in new_state.holdings
        assert new_state.cash_balance == financial_state.cash_balance
    
    @pytest.mark.asyncio
    async def test_simulate_action_insufficient_shares(self, analyzer, financial_state):
        """Test sell action with insufficient shares"""
        action = FinancialAction(
            action_type="sell",
            symbol="AAPL",
            quantity=200,  # More than owned
            price=175.0
        )
        
        new_state = await analyzer._simulate_action(financial_state, action)
        
        # Should not execute due to insufficient shares
        assert new_state.holdings["AAPL"] == financial_state.holdings["AAPL"]
        assert new_state.cash_balance == financial_state.cash_balance
    
    @pytest.mark.asyncio
    async def test_simulate_playout_cpu(self, analyzer, financial_state):
        """Test CPU-based playout simulation"""
        analyzer.use_cuda = False
        
        reward = await analyzer._simulate_playout(
            financial_state,
            "maximize_return"
        )
        
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0  # Reasonable range for returns
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_simulate_playout_gpu(self, financial_state):
        """Test GPU-based playout simulation"""
        analyzer = MCTSFinancialAnalyzer(
            device="cuda",
            enable_gpu_optimization=True
        )
        
        reward = await analyzer._simulate_playout(
            financial_state,
            "maximize_return"
        )
        
        assert isinstance(reward, float)
        assert -1.0 <= reward <= 1.0
    
    def test_backpropagate(self, analyzer, financial_state):
        """Test backpropagation in MCTS tree"""
        # Create tree structure
        root = MCTSNode(financial_state)
        child = MCTSNode(financial_state, parent=root)
        grandchild = MCTSNode(financial_state, parent=child)
        
        # Backpropagate from grandchild
        reward = 0.1
        analyzer._backpropagate(grandchild, reward)
        
        assert grandchild.visits == 1
        assert grandchild.total_reward == reward
        assert child.visits == 1
        assert child.total_reward == reward
        assert root.visits == 1
        assert root.total_reward == reward
    
    def test_get_best_action(self, analyzer, financial_state, available_actions):
        """Test getting best action from MCTS tree"""
        root = MCTSNode(financial_state)
        
        # Create children with different visit counts
        for i, action in enumerate(available_actions):
            child = MCTSNode(financial_state, parent=root, action=action)
            child.visits = (i + 1) * 10
            child.total_reward = (i + 1) * 5
            root.children.append(child)
        
        best_action = analyzer._get_best_action(root)
        
        # Should select action with highest visit count
        assert best_action == available_actions[-1]
    
    def test_get_best_action_no_children(self, analyzer, financial_state):
        """Test getting best action with no children"""
        root = MCTSNode(financial_state)
        
        best_action = analyzer._get_best_action(root)
        
        assert best_action.action_type == "hold"
        assert best_action.reasoning == "No actions evaluated"
    
    @pytest.mark.asyncio
    async def test_generate_analysis(self, analyzer, financial_state, available_actions):
        """Test generating analysis report"""
        root = MCTSNode(financial_state)
        best_action = available_actions[0]
        
        # Mock internal methods
        analyzer._calculate_risk_metrics = AsyncMock(return_value={
            "portfolio_volatility": 0.18,
            "value_at_risk_95": 5000.0,
            "sharpe_ratio": 0.85,
            "beta": 1.1,
            "max_drawdown": 0.15
        })
        
        analyzer._project_performance = AsyncMock(return_value={
            "30_days": {"expected_return": 0.02},
            "90_days": {"expected_return": 0.06},
            "365_days": {"expected_return": 0.08}
        })
        
        analyzer._get_alternative_actions = Mock(return_value=[])
        analyzer._get_tree_depth = Mock(return_value=3)
        analyzer._count_nodes = Mock(return_value=150)
        
        analysis = await analyzer._generate_analysis(
            root,
            best_action,
            financial_state,
            "maximize_return"
        )
        
        assert analysis["recommendation"]["action"] == best_action.action_type
        assert analysis["recommendation"]["symbol"] == best_action.symbol
        assert analysis["current_state"]["portfolio_value"] == financial_state.portfolio_value
        assert "risk_analysis" in analysis
        assert "performance_projection" in analysis
        assert analysis["optimization_goal"] == "maximize_return"
        assert analysis["simulation_details"]["num_simulations"] == analyzer.num_simulations
    
    @pytest.mark.asyncio
    async def test_calculate_risk_metrics(self, analyzer, financial_state):
        """Test risk metrics calculation"""
        risk_metrics = await analyzer._calculate_risk_metrics(financial_state)
        
        assert "portfolio_volatility" in risk_metrics
        assert "value_at_risk_95" in risk_metrics
        assert "sharpe_ratio" in risk_metrics
        assert "beta" in risk_metrics
        assert "max_drawdown" in risk_metrics
        
        # Check reasonable ranges
        assert 0 <= risk_metrics["portfolio_volatility"] <= 1
        assert risk_metrics["value_at_risk_95"] >= 0
        assert -5 <= risk_metrics["sharpe_ratio"] <= 5
        assert 0 <= risk_metrics["beta"] <= 3
        assert -1 <= risk_metrics["max_drawdown"] <= 0
    
    @pytest.mark.asyncio
    async def test_project_performance(self, analyzer, financial_state, available_actions):
        """Test performance projection"""
        action = available_actions[0]  # buy action
        
        projection = await analyzer._project_performance(financial_state, action)
        
        assert "30_days" in projection
        assert "90_days" in projection
        assert "365_days" in projection
        
        # Check structure
        for horizon in ["30_days", "90_days", "365_days"]:
            assert "expected_return" in projection[horizon]
            assert "volatility" in projection[horizon]
            assert "confidence_interval" in projection[horizon]
            assert len(projection[horizon]["confidence_interval"]) == 2
    
    def test_get_alternative_actions(self, analyzer, financial_state, available_actions):
        """Test getting alternative actions"""
        root = MCTSNode(financial_state)
        
        # Create children with different scores
        for i, action in enumerate(available_actions):
            child = MCTSNode(financial_state, parent=root, action=action)
            child.visits = (i + 1) * 10
            child.total_reward = (i + 1) * 5
            root.children.append(child)
        
        root.visits = 100
        
        alternatives = analyzer._get_alternative_actions(root)
        
        assert len(alternatives) <= 5
        assert all("action" in alt for alt in alternatives)
        assert all("visits" in alt for alt in alternatives)
        assert all("average_reward" in alt for alt in alternatives)
        assert all("confidence" in alt for alt in alternatives)
        
        # Check ordering by visits
        for i in range(len(alternatives) - 1):
            assert alternatives[i]["visits"] >= alternatives[i + 1]["visits"]
    
    @pytest.mark.asyncio
    async def test_rebalance_portfolio(self, analyzer, financial_state):
        """Test portfolio rebalancing"""
        # Mock price getter
        analyzer._get_price = Mock(return_value=100.0)
        
        await analyzer._rebalance_portfolio(financial_state)
        
        # Check that rebalancing was attempted
        assert analyzer._get_price.called
    
    def test_get_price(self, analyzer):
        """Test price retrieval"""
        price = analyzer._get_price("AAPL")
        assert price == 175.0
        
        price = analyzer._get_price("UNKNOWN")
        assert price == 100.0  # Default price
    
    @pytest.mark.asyncio
    async def test_calculate_portfolio_value(self, analyzer, financial_state):
        """Test portfolio value calculation"""
        # Mock price getter
        analyzer._get_price = Mock(side_effect=lambda s: {"AAPL": 175.0, "GOOGL": 140.0}.get(s, 100.0))
        
        value = await analyzer._calculate_portfolio_value(financial_state)
        
        expected_value = (100 * 175.0) + (50 * 140.0) + 20000.0
        assert value == expected_value
    
    def test_get_tree_depth(self, analyzer, financial_state):
        """Test tree depth calculation"""
        # Create tree
        root = MCTSNode(financial_state)
        child1 = MCTSNode(financial_state, parent=root)
        child2 = MCTSNode(financial_state, parent=root)
        grandchild = MCTSNode(financial_state, parent=child1)
        
        root.children = [child1, child2]
        child1.children = [grandchild]
        
        depth = analyzer._get_tree_depth(root)
        assert depth == 2
    
    def test_count_nodes(self, analyzer, financial_state):
        """Test node counting"""
        # Create tree
        root = MCTSNode(financial_state)
        child1 = MCTSNode(financial_state, parent=root)
        child2 = MCTSNode(financial_state, parent=root)
        grandchild = MCTSNode(financial_state, parent=child1)
        
        root.children = [child1, child2]
        child1.children = [grandchild]
        
        count = analyzer._count_nodes(root)
        assert count == 4
    
    def test_optimize_gpu_memory(self, analyzer):
        """Test GPU memory optimization"""
        if torch.cuda.is_available():
            analyzer.optimize_gpu_memory()
            # Should not raise any errors
            assert True
    
    def test_get_performance_metrics(self, analyzer):
        """Test performance metrics retrieval"""
        metrics = analyzer.get_performance_metrics()
        
        assert "device" in metrics
        assert "gpu_available" in metrics
        assert "num_simulations" in metrics
        assert "max_depth" in metrics
        
        if torch.cuda.is_available():
            assert "gpu_memory_allocated" in metrics
            assert "gpu_memory_reserved" in metrics
            assert metrics["gpu_memory_allocated"] >= 0
            assert metrics["gpu_memory_reserved"] >= 0


class TestIntegration:
    """Integration tests for MCTS Financial Analyzer"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        analyzer = MCTSFinancialAnalyzer(
            device="cpu",
            num_simulations=50,  # Reduced for testing
            max_depth=3
        )
        
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            cash_balance=10000.0,
            risk_tolerance=0.6,
            time_horizon=252,
            market_conditions={
                "volatility": 0.15,
                "trend": "bullish",
                "vix": 18.5
            },
            timestamp=datetime.now()
        )
        
        actions = [
            FinancialAction(action_type="buy", symbol="AMZN", quantity=20, price=170.0),
            FinancialAction(action_type="sell", symbol="AAPL", quantity=25, price=175.0),
            FinancialAction(action_type="hold"),
            FinancialAction(action_type="rebalance")
        ]
        
        # Run analysis
        result = await analyzer.analyze_portfolio(
            state,
            actions,
            optimization_goal="maximize_return"
        )
        
        # Verify result structure
        assert "recommendation" in result
        assert "current_state" in result
        assert "risk_analysis" in result
        assert "performance_projection" in result
        assert "alternative_actions" in result
        assert "simulation_details" in result
        
        # Verify recommendation
        recommendation = result["recommendation"]
        assert recommendation["action"] in ["buy", "sell", "hold", "rebalance"]
        assert recommendation["confidence"] >= 0
        assert recommendation["reasoning"] != ""
        
        # Verify risk analysis
        risk = result["risk_analysis"]
        assert all(key in risk for key in [
            "portfolio_volatility", "value_at_risk_95", 
            "sharpe_ratio", "beta", "max_drawdown"
        ])
        
        # Verify simulation details
        sim_details = result["simulation_details"]
        assert sim_details["num_simulations"] == 50
        assert sim_details["tree_depth"] >= 0
        assert sim_details["nodes_explored"] >= 1
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU not available")
    async def test_gpu_acceleration(self):
        """Test GPU acceleration performance"""
        cpu_analyzer = MCTSFinancialAnalyzer(
            device="cpu",
            num_simulations=100,
            enable_gpu_optimization=False
        )
        
        gpu_analyzer = MCTSFinancialAnalyzer(
            device="cuda",
            num_simulations=100,
            enable_gpu_optimization=True
        )
        
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100, "GOOGL": 50, "MSFT": 75},
            cash_balance=10000.0,
            risk_tolerance=0.6,
            time_horizon=252,
            market_conditions={"volatility": 0.15},
            timestamp=datetime.now()
        )
        
        actions = [
            FinancialAction(action_type="buy", symbol="AMZN", quantity=20, price=170.0),
            FinancialAction(action_type="hold")
        ]
        
        # Measure CPU performance
        import time
        cpu_start = time.time()
        cpu_result = await cpu_analyzer.analyze_portfolio(state, actions)
        cpu_time = time.time() - cpu_start
        
        # Measure GPU performance  
        gpu_start = time.time()
        gpu_result = await gpu_analyzer.analyze_portfolio(state, actions)
        gpu_time = time.time() - gpu_start
        
        # GPU should be faster for large simulations
        print(f"CPU time: {cpu_time:.3f}s, GPU time: {gpu_time:.3f}s")
        
        # Verify results are similar
        assert cpu_result["recommendation"]["action"] == gpu_result["recommendation"]["action"]
        assert abs(cpu_result["risk_analysis"]["portfolio_volatility"] - 
                  gpu_result["risk_analysis"]["portfolio_volatility"]) < 0.05
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_error_handling(self):
        """Test error handling in various scenarios"""
        analyzer = MCTSFinancialAnalyzer(device="cpu", num_simulations=10)
        
        # Test with empty portfolio
        empty_state = FinancialState(
            portfolio_value=0.0,
            holdings={},
            cash_balance=0.0,
            risk_tolerance=0.5,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        result = await analyzer.analyze_portfolio(empty_state, [])
        assert result["recommendation"]["action"] == "hold"
        
        # Test with invalid action
        normal_state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100},
            cash_balance=10000.0,
            risk_tolerance=0.5,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        invalid_action = FinancialAction(
            action_type="invalid_action",
            symbol="AAPL"
        )
        
        # Should handle gracefully
        result = await analyzer.analyze_portfolio(
            normal_state, 
            [invalid_action]
        )
        assert "recommendation" in result
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_different_optimization_goals(self):
        """Test different optimization objectives"""
        analyzer = MCTSFinancialAnalyzer(device="cpu", num_simulations=50)
        
        state = FinancialState(
            portfolio_value=100000.0,
            holdings={"AAPL": 100, "GOOGL": 50},
            cash_balance=20000.0,
            risk_tolerance=0.5,
            time_horizon=365,
            market_conditions={},
            timestamp=datetime.now()
        )
        
        actions = [
            FinancialAction(action_type="buy", symbol="MSFT", quantity=50, price=380.0),
            FinancialAction(action_type="hold")
        ]
        
        # Test different goals
        goals = ["maximize_return", "minimize_risk", "sharpe_ratio"]
        results = {}
        
        for goal in goals:
            result = await analyzer.analyze_portfolio(state, actions, goal)
            results[goal] = result
        
        # Different goals may lead to different recommendations
        assert all(goal in results for goal in goals)
        assert all("recommendation" in results[goal] for goal in goals)
        
        # Risk minimization might prefer holding
        risk_action = results["minimize_risk"]["recommendation"]["action"]
        return_action = results["maximize_return"]["recommendation"]["action"]
        
        # These could be different based on market conditions
        assert risk_action in ["buy", "sell", "hold", "rebalance"]
        assert return_action in ["buy", "sell", "hold", "rebalance"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
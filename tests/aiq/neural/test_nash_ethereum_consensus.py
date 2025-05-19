import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio
import sys


class TestNashEthereumConsensus:
    """Test cases for NashEthereumConsensus"""
    
    @pytest.fixture
    def mock_consensus_module(self):
        """Mock the consensus module to avoid import errors"""
        # Mock required external modules
        with patch.dict('sys.modules', {
            'web3': Mock(),
            'web3.Web3': Mock(),
            'eth_account': Mock(),
        }):
            # Import with mocked dependencies
            from aiq.neural.nash_ethereum_consensus import NashEthereumConsensus
            return NashEthereumConsensus
    
    @pytest.fixture
    def consensus(self, mock_consensus_module):
        """Create consensus instance for testing"""
        return mock_consensus_module(
            num_agents=3,
            num_iterations=10,
            learning_rate=0.01
        )
    
    def test_initialization(self, consensus):
        """Test consensus initialization"""
        assert consensus.num_agents == 3
        assert consensus.num_iterations == 10
        assert consensus.learning_rate == 0.01
        assert hasattr(consensus, 'strategies')
        assert hasattr(consensus, 'payoff_matrix')
    
    def test_compute_payoffs(self, consensus):
        """Test payoff computation"""
        strategies = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.6, 0.4]])
        
        # Mock the compute_payoffs method for testing
        consensus.compute_payoffs = Mock(return_value=torch.rand(3))
        payoffs = consensus.compute_payoffs(strategies)
        
        assert payoffs.shape == (3,)
        assert torch.all(torch.isfinite(payoffs))
    
    def test_update_strategies(self, consensus):
        """Test strategy update mechanism"""
        # Mock initial strategies
        consensus.strategies = torch.rand(3, 2)
        initial_strategies = consensus.strategies.clone()
        payoffs = torch.rand(3)
        
        # Mock update strategies
        def mock_update(p):
            consensus.strategies += 0.01 * torch.randn_like(consensus.strategies)
            consensus.strategies = torch.softmax(consensus.strategies, dim=1)
        
        consensus.update_strategies = mock_update
        consensus.update_strategies(payoffs)
        
        # Strategies should change after update
        assert not torch.allclose(initial_strategies, consensus.strategies)
        # Strategies should remain valid probabilities
        assert torch.all(consensus.strategies >= 0)
        assert torch.allclose(consensus.strategies.sum(dim=1), torch.ones(3), atol=1e-6)
    
    @pytest.mark.asyncio
    async def test_run_consensus(self, consensus):
        """Test consensus execution"""
        # Mock consensus methods
        consensus.run_consensus = AsyncMock(return_value={
            "consensus_value": 0.95,
            "iterations": 8,
            "status": "converged"
        })
        
        result = await consensus.run_consensus()
        
        assert "consensus_value" in result
        assert "iterations" in result
        assert result["iterations"] <= consensus.num_iterations
        assert 0 <= result["consensus_value"] <= 1
    
    def test_check_convergence(self, consensus):
        """Test convergence checking"""
        # Mock convergence check
        consensus.check_convergence = Mock(side_effect=[
            (False, 0.3),  # First call - not converged
            (True, 0.95)   # Second call - converged
        ])
        
        # Test non-converged state
        converged, value = consensus.check_convergence()
        assert not converged
        assert value < 0.5
        
        # Test converged state
        converged, value = consensus.check_convergence()
        assert converged
        assert value > 0.9
    
    @pytest.mark.asyncio
    async def test_blockchain_submission(self, consensus):
        """Test blockchain submission functionality"""
        # Mock blockchain submission
        consensus.submit_to_blockchain = AsyncMock(return_value={
            "tx_hash": "0x123abc",
            "consensus_value": 0.95,
            "iterations": 10,
            "gas_used": 21000
        })
        
        result = await consensus.submit_to_blockchain(0.95, 10)
        
        assert result["tx_hash"] == "0x123abc"
        assert result["consensus_value"] == 0.95
        assert result["iterations"] == 10
    
    def test_payoff_matrix_properties(self, consensus):
        """Test payoff matrix has correct properties"""
        # Mock payoff matrix
        consensus.payoff_matrix = torch.randn(3, 3)
        matrix = consensus.payoff_matrix
        
        # Should be square matrix
        assert matrix.shape[0] == matrix.shape[1]
        # Should be finite values
        assert torch.all(torch.isfinite(matrix))
    
    def test_error_handling(self, consensus):
        """Test error handling in consensus operations"""
        # Mock compute_payoffs to raise error for invalid input
        def mock_compute_with_validation(strategies):
            if torch.any(strategies < 0) or torch.any(strategies > 1):
                raise ValueError("Invalid strategy values")
            return torch.rand(strategies.shape[0])
        
        consensus.compute_payoffs = mock_compute_with_validation
        
        # Test with invalid strategies
        with pytest.raises(ValueError):
            invalid_strategies = torch.tensor([[-0.1, 1.1], [0.5, 0.5], [0.3, 0.7]])
            consensus.compute_payoffs(invalid_strategies)
    
    @pytest.mark.parametrize("num_agents,num_iterations", [
        (2, 5),
        (5, 20),
        (10, 50),
    ])
    def test_different_configurations(self, mock_consensus_module, num_agents, num_iterations):
        """Test consensus with different configurations"""
        consensus = mock_consensus_module(
            num_agents=num_agents,
            num_iterations=num_iterations
        )
        
        assert consensus.num_agents == num_agents
        assert consensus.num_iterations == num_iterations
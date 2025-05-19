import pytest
import torch
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
import asyncio
from cryptography.fernet import Fernet

from aiq.neural.secure_nash_ethereum import SecureNashEthereumConsensus


class TestSecureNashEthereumConsensus:
    """Test cases for SecureNashEthereumConsensus"""
    
    @pytest.fixture
    def secure_consensus(self):
        """Create secure consensus instance for testing"""
        return SecureNashEthereumConsensus(
            num_agents=3,
            num_iterations=10,
            encryption_key=Fernet.generate_key()
        )
    
    def test_initialization(self, secure_consensus):
        """Test secure consensus initialization"""
        assert secure_consensus.num_agents == 3
        assert secure_consensus.num_iterations == 10
        assert secure_consensus.fernet is not None
        assert secure_consensus.encryption_key is not None
    
    def test_encrypt_decrypt_data(self, secure_consensus):
        """Test data encryption and decryption"""
        original_data = {"consensus": 0.95, "iterations": 10}
        
        # Encrypt data
        encrypted = secure_consensus.encrypt_data(original_data)
        assert isinstance(encrypted, bytes)
        assert encrypted != str(original_data).encode()
        
        # Decrypt data
        decrypted = secure_consensus.decrypt_data(encrypted)
        assert decrypted == original_data
    
    @pytest.mark.asyncio
    async def test_secure_run_consensus(self, secure_consensus):
        """Test secure consensus execution"""
        with patch.object(secure_consensus, 'submit_secure_to_blockchain', new_callable=AsyncMock) as mock_submit:
            mock_submit.return_value = {"tx_hash": "0x456", "status": "success"}
            
            result = await secure_consensus.secure_run_consensus()
            
            assert "consensus_value" in result
            assert "iterations" in result
            assert "encrypted" in result
            assert result["iterations"] <= secure_consensus.num_iterations
            mock_submit.assert_called_once()
    
    def test_validate_agent_credentials(self, secure_consensus):
        """Test agent credential validation"""
        # Valid credentials
        valid_agent = {"id": "agent1", "auth_token": "valid_token_123"}
        assert secure_consensus.validate_agent_credentials(valid_agent)
        
        # Invalid credentials - missing auth token
        invalid_agent1 = {"id": "agent2"}
        assert not secure_consensus.validate_agent_credentials(invalid_agent1)
        
        # Invalid credentials - empty auth token
        invalid_agent2 = {"id": "agent3", "auth_token": ""}
        assert not secure_consensus.validate_agent_credentials(invalid_agent2)
    
    @pytest.mark.asyncio
    async def test_secure_blockchain_submission(self, secure_consensus):
        """Test secure blockchain submission"""
        encrypted_data = secure_consensus.encrypt_data({"value": 0.9})
        
        with patch('aiq.neural.secure_nash_ethereum.Web3') as mock_web3:
            mock_instance = Mock()
            mock_web3.return_value = mock_instance
            mock_instance.eth.gas_price = 1000000000
            mock_instance.eth.get_transaction_count.return_value = 5
            mock_instance.eth.send_raw_transaction.return_value = b'0x789'
            
            result = await secure_consensus.submit_secure_to_blockchain(encrypted_data)
            
            assert result["tx_hash"] == "0x789"
            assert result["data_hash"] is not None
            assert result["encryption_used"] is True
    
    def test_audit_log_creation(self, secure_consensus):
        """Test audit log functionality"""
        action = "consensus_reached"
        details = {"consensus_value": 0.95, "agents": 3}
        
        log_entry = secure_consensus.create_audit_log(action, details)
        
        assert log_entry["action"] == action
        assert log_entry["details"] == details
        assert "timestamp" in log_entry
        assert "agent_signatures" in log_entry
    
    def test_multi_signature_verification(self, secure_consensus):
        """Test multi-signature verification"""
        # Mock signatures from multiple agents
        signatures = [
            {"agent_id": "agent1", "signature": "sig1", "timestamp": "2024-01-01"},
            {"agent_id": "agent2", "signature": "sig2", "timestamp": "2024-01-01"},
            {"agent_id": "agent3", "signature": "sig3", "timestamp": "2024-01-01"}
        ]
        
        # At least 2 out of 3 signatures required
        assert secure_consensus.verify_multi_signature(signatures, required=2)
        
        # Not enough signatures
        assert not secure_consensus.verify_multi_signature(signatures[:1], required=2)
    
    def test_privacy_preserving_aggregation(self, secure_consensus):
        """Test privacy-preserving aggregation"""
        # Individual agent values
        agent_values = [
            torch.tensor([0.8, 0.2]),
            torch.tensor([0.7, 0.3]),
            torch.tensor([0.9, 0.1])
        ]
        
        # Aggregate with privacy
        aggregated = secure_consensus.privacy_preserving_aggregate(agent_values)
        
        assert aggregated.shape == (2,)
        assert torch.allclose(aggregated, torch.tensor([0.8, 0.2]), atol=0.1)
        # Check that noise was added
        assert not torch.allclose(aggregated, torch.mean(torch.stack(agent_values), dim=0))
    
    @pytest.mark.parametrize("encryption_strength", ["AES-128", "AES-256"])
    def test_different_encryption_levels(self, encryption_strength):
        """Test different encryption strength levels"""
        consensus = SecureNashEthereumConsensus(
            encryption_strength=encryption_strength
        )
        
        data = {"test": "data"}
        encrypted = consensus.encrypt_data(data)
        decrypted = consensus.decrypt_data(encrypted)
        
        assert decrypted == data
    
    def test_zero_knowledge_proof(self, secure_consensus):
        """Test zero-knowledge proof generation"""
        value = 0.95
        
        # Generate proof
        proof = secure_consensus.generate_zero_knowledge_proof(value)
        
        assert "commitment" in proof
        assert "challenge" in proof
        assert "response" in proof
        
        # Verify proof
        assert secure_consensus.verify_zero_knowledge_proof(proof, value)
        
        # Should fail with wrong value
        assert not secure_consensus.verify_zero_knowledge_proof(proof, 0.5)
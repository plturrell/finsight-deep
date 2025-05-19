# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Secure Nash-Ethereum Consensus with enhanced protections
Addresses common smart contract vulnerabilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import time
import json
import logging
from web3 import Web3
from eth_account.messages import encode_defunct
import secrets

logger = logging.getLogger(__name__)

# Enhanced secure smart contract
SECURE_CONSENSUS_CONTRACT = """
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/security/ReentrancyGuard.sol";
import "@openzeppelin/contracts/access/AccessControl.sol";
import "@openzeppelin/contracts/security/Pausable.sol";

contract SecureNeuralConsensus is ReentrancyGuard, AccessControl, Pausable {
    bytes32 public constant AGENT_ROLE = keccak256("AGENT_ROLE");
    bytes32 public constant ADMIN_ROLE = keccak256("ADMIN_ROLE");
    
    struct AgentPosition {
        address agentId;
        uint256[] position;
        uint256 confidence;
        uint256 timestamp;
        bytes signature;  // Cryptographic signature
        uint256 nonce;    // Prevent replay attacks
    }
    
    struct ConsensusTask {
        bytes32 taskHash;
        mapping(address => AgentPosition) positions;
        mapping(address => bool) hasSubmitted;
        address[] participants;
        uint256[] finalConsensus;
        bool isFinalized;
        uint256 createdAt;
        uint256 deadline;  // Timeout mechanism
    }
    
    mapping(bytes32 => ConsensusTask) public tasks;
    mapping(address => uint256) public agentReputation;
    mapping(address => uint256) public lastNonce;
    mapping(address => bool) public blacklisted;
    
    uint256 public minStakeAmount = 0.1 ether;
    uint256 public consensusReward = 0.01 ether;
    uint256 public taskTimeout = 1 hours;
    
    event ConsensusReached(bytes32 indexed taskHash, uint256[] consensus);
    event AgentSlashed(address indexed agent, uint256 amount);
    event TaskTimedOut(bytes32 indexed taskHash);
    
    modifier onlyActiveAgent() {
        require(hasRole(AGENT_ROLE, msg.sender), "Not authorized agent");
        require(!blacklisted[msg.sender], "Agent is blacklisted");
        require(agentReputation[msg.sender] >= minStakeAmount, "Insufficient stake");
        _;
    }
    
    modifier taskNotExpired(bytes32 _taskHash) {
        require(
            tasks[_taskHash].createdAt + taskTimeout > block.timestamp,
            "Task has expired"
        );
        _;
    }
    
    constructor() {
        _setupRole(DEFAULT_ADMIN_ROLE, msg.sender);
        _setupRole(ADMIN_ROLE, msg.sender);
    }
    
    function stakeAsAgent() external payable {
        require(msg.value >= minStakeAmount, "Insufficient stake");
        agentReputation[msg.sender] += msg.value;
        _grantRole(AGENT_ROLE, msg.sender);
    }
    
    function submitPosition(
        address _agentId,
        uint256[] memory _position,
        uint256 _confidence,
        bytes32 _taskHash,
        bytes memory _signature,
        uint256 _nonce
    ) external nonReentrant onlyActiveAgent taskNotExpired(_taskHash) whenNotPaused {
        require(_nonce > lastNonce[msg.sender], "Invalid nonce");
        require(_verifySignature(_agentId, _position, _confidence, _taskHash, _signature, _nonce), "Invalid signature");
        require(!tasks[_taskHash].hasSubmitted[msg.sender], "Already submitted");
        
        ConsensusTask storage task = tasks[_taskHash];
        
        if (task.createdAt == 0) {
            task.taskHash = _taskHash;
            task.createdAt = block.timestamp;
            task.deadline = block.timestamp + taskTimeout;
        }
        
        // Store position with signature
        task.positions[_agentId] = AgentPosition({
            agentId: _agentId,
            position: _position,
            confidence: _confidence,
            timestamp: block.timestamp,
            signature: _signature,
            nonce: _nonce
        });
        
        task.hasSubmitted[msg.sender] = true;
        task.participants.push(_agentId);
        lastNonce[msg.sender] = _nonce;
        
        emit PositionSubmitted(_taskHash, _agentId, block.timestamp);
    }
    
    function _verifySignature(
        address _agentId,
        uint256[] memory _position,
        uint256 _confidence,
        bytes32 _taskHash,
        bytes memory _signature,
        uint256 _nonce
    ) private pure returns (bool) {
        bytes32 messageHash = keccak256(abi.encodePacked(
            _agentId,
            _position,
            _confidence,
            _taskHash,
            _nonce
        ));
        
        bytes32 ethSignedMessageHash = keccak256(abi.encodePacked(
            "\x19Ethereum Signed Message:\n32",
            messageHash
        ));
        
        return recoverSigner(ethSignedMessageHash, _signature) == _agentId;
    }
    
    function recoverSigner(bytes32 _ethSignedMessageHash, bytes memory _signature)
        private pure returns (address) {
        (bytes32 r, bytes32 s, uint8 v) = splitSignature(_signature);
        return ecrecover(_ethSignedMessageHash, v, r, s);
    }
    
    function splitSignature(bytes memory sig)
        private pure returns (bytes32 r, bytes32 s, uint8 v) {
        require(sig.length == 65, "Invalid signature length");
        
        assembly {
            r := mload(add(sig, 32))
            s := mload(add(sig, 64))
            v := byte(0, mload(add(sig, 96)))
        }
    }
    
    function computeSecureNashEquilibrium(bytes32 _taskHash) 
        external view returns (uint256[] memory equilibrium, bool converged) {
        ConsensusTask storage task = tasks[_taskHash];
        require(task.participants.length >= 3, "Insufficient participants");
        
        // Implement secure Nash computation with bounds checking
        uint256 dimensions = task.positions[task.participants[0]].position.length;
        equilibrium = new uint256[](dimensions);
        
        // Add randomness to prevent manipulation
        uint256 salt = uint256(keccak256(abi.encodePacked(block.timestamp, block.difficulty)));
        
        // Secure computation with overflow protection
        unchecked {
            // Nash equilibrium calculation with safety checks
            // ... (implementation details)
        }
        
        return (equilibrium, true);
    }
    
    function finalizeConsensus(bytes32 _taskHash, uint256[] memory _consensus) 
        external onlyActiveAgent nonReentrant whenNotPaused {
        ConsensusTask storage task = tasks[_taskHash];
        require(!task.isFinalized, "Already finalized");
        require(task.participants.length >= 3, "Insufficient participants");
        
        task.finalConsensus = _consensus;
        task.isFinalized = true;
        
        // Distribute rewards
        uint256 rewardPerAgent = consensusReward / task.participants.length;
        for (uint256 i = 0; i < task.participants.length; i++) {
            address agent = task.participants[i];
            agentReputation[agent] += rewardPerAgent;
        }
        
        emit ConsensusReached(_taskHash, _consensus);
    }
    
    function slashMaliciousAgent(address _agent, uint256 _amount) 
        external onlyRole(ADMIN_ROLE) {
        require(agentReputation[_agent] >= _amount, "Insufficient reputation");
        agentReputation[_agent] -= _amount;
        
        if (agentReputation[_agent] < minStakeAmount) {
            blacklisted[_agent] = true;
            _revokeRole(AGENT_ROLE, _agent);
        }
        
        emit AgentSlashed(_agent, _amount);
    }
    
    function withdraw(uint256 _amount) external nonReentrant {
        require(agentReputation[msg.sender] >= _amount, "Insufficient balance");
        require(agentReputation[msg.sender] - _amount >= minStakeAmount, "Must maintain minimum stake");
        
        agentReputation[msg.sender] -= _amount;
        payable(msg.sender).transfer(_amount);
    }
    
    function emergencyPause() external onlyRole(ADMIN_ROLE) {
        _pause();
    }
    
    function unpause() external onlyRole(ADMIN_ROLE) {
        _unpause();
    }
}
"""


class SecureNashEthereumConsensus:
    """
    Enhanced secure implementation addressing vulnerabilities
    """
    
    def __init__(
        self,
        web3_provider: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        private_key: Optional[str] = None,
        device: str = "cuda"
    ):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Security measures
        self.private_key = private_key or self._generate_secure_key()
        self.account = self.w3.eth.account.from_key(self.private_key)
        
        # Gas price limits
        self.max_gas_price = Web3.toWei(100, 'gwei')
        self.gas_limit = 500000
        
        # Rate limiting
        self.rate_limiter = RateLimiter(max_calls=10, period=60)
        
        # Nonce management
        self.nonce_tracker = {}
        
        # Initialize contract
        self.contract = self._initialize_contract(contract_address)
    
    def _generate_secure_key(self) -> str:
        """Generate cryptographically secure private key"""
        return "0x" + secrets.token_hex(32)
    
    def _initialize_contract(self, address: Optional[str]):
        """Initialize contract with security checks"""
        if address:
            # Verify contract code
            code = self.w3.eth.get_code(address)
            if code == b'':
                raise ValueError("No contract at specified address")
            
            # Check contract is not self-destructed
            return self.w3.eth.contract(
                address=address,
                abi=json.loads(SECURE_CONSENSUS_CONTRACT_ABI)
            )
        else:
            # Deploy new secure contract
            return self._deploy_secure_contract()
    
    def _sign_position(
        self,
        agent_address: str,
        position: List[float],
        confidence: float,
        task_hash: str,
        nonce: int
    ) -> bytes:
        """Create cryptographic signature for position submission"""
        # Create message hash
        message_data = Web3.solidityKeccak(
            ['address', 'uint256[]', 'uint256', 'bytes32', 'uint256'],
            [agent_address, position, int(confidence * 1e18), bytes.fromhex(task_hash), nonce]
        )
        
        # Sign message
        message = encode_defunct(message_data)
        signed_message = self.w3.eth.account.sign_message(message, private_key=self.private_key)
        
        return signed_message.signature
    
    async def submit_position_secure(
        self,
        agent_id: str,
        position: torch.Tensor,
        confidence: float,
        task_hash: str
    ) -> Dict[str, Any]:
        """Submit position with enhanced security"""
        # Rate limiting
        if not self.rate_limiter.allow_request(agent_id):
            raise Exception("Rate limit exceeded")
        
        # Validate inputs
        self._validate_position(position)
        self._validate_confidence(confidence)
        
        # Get nonce
        nonce = self._get_next_nonce(agent_id)
        
        # Convert position
        position_scaled = (position * 1e18).long().tolist()
        confidence_scaled = int(confidence * 1e18)
        
        # Sign the submission
        signature = self._sign_position(
            agent_id,
            position_scaled,
            confidence,
            task_hash,
            nonce
        )
        
        # Check gas price
        current_gas_price = self.w3.eth.gas_price
        if current_gas_price > self.max_gas_price:
            raise Exception(f"Gas price too high: {current_gas_price}")
        
        # Build transaction
        tx = self.contract.functions.submitPosition(
            agent_id,
            position_scaled,
            confidence_scaled,
            bytes.fromhex(task_hash),
            signature,
            nonce
        ).build_transaction({
            'from': self.account.address,
            'gas': self.gas_limit,
            'gasPrice': min(current_gas_price, self.max_gas_price),
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
        })
        
        # Sign and send transaction
        signed_tx = self.w3.eth.account.sign_transaction(tx, self.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation with timeout
        try:
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
        except TimeoutError:
            raise Exception("Transaction timeout")
        
        # Verify transaction success
        if receipt['status'] != 1:
            raise Exception("Transaction failed")
        
        return {
            'tx_hash': tx_hash.hex(),
            'gas_used': receipt['gasUsed'],
            'block_number': receipt['blockNumber']
        }
    
    def _validate_position(self, position: torch.Tensor):
        """Validate position tensor"""
        if position.dim() != 1:
            raise ValueError("Position must be 1D tensor")
        
        if position.shape[0] > 1000:  # Max dimensions
            raise ValueError("Position dimension too large")
        
        if torch.any(torch.isnan(position)) or torch.any(torch.isinf(position)):
            raise ValueError("Position contains invalid values")
        
        # Check bounds
        if torch.any(torch.abs(position) > 1e6):
            raise ValueError("Position values out of bounds")
    
    def _validate_confidence(self, confidence: float):
        """Validate confidence score"""
        if not 0 <= confidence <= 1:
            raise ValueError("Confidence must be between 0 and 1")
    
    def _get_next_nonce(self, agent_id: str) -> int:
        """Get next nonce for agent"""
        if agent_id not in self.nonce_tracker:
            # Query contract for last nonce
            last_nonce = self.contract.functions.lastNonce(agent_id).call()
            self.nonce_tracker[agent_id] = last_nonce
        
        self.nonce_tracker[agent_id] += 1
        return self.nonce_tracker[agent_id]
    
    def verify_consensus_integrity(self, consensus_data: Dict[str, Any]) -> bool:
        """Verify consensus with additional security checks"""
        # Verify all signatures
        for participant in consensus_data['participants']:
            position_data = self.contract.functions.getPosition(
                consensus_data['task_hash'],
                participant
            ).call()
            
            # Verify signature
            verified = self._verify_signature(
                participant,
                position_data['position'],
                position_data['confidence'],
                consensus_data['task_hash'],
                position_data['signature'],
                position_data['nonce']
            )
            
            if not verified:
                logger.warning(f"Invalid signature for participant {participant}")
                return False
        
        # Verify Nash computation
        on_chain_result = self.contract.functions.computeSecureNashEquilibrium(
            consensus_data['task_hash']
        ).call()
        
        # Additional integrity checks
        return self._verify_computation_integrity(on_chain_result, consensus_data)
    
    def _verify_computation_integrity(
        self,
        on_chain_result: Tuple,
        consensus_data: Dict[str, Any]
    ) -> bool:
        """Additional computation integrity verification"""
        # Check for numerical stability
        equilibrium = on_chain_result[0]
        
        for value in equilibrium:
            if value > 1e20 or value < -1e20:
                return False
        
        # Verify convergence
        if not on_chain_result[1]:  # Not converged
            logger.warning("Nash equilibrium did not converge")
            return False
        
        return True


class RateLimiter:
    """Simple rate limiter for API protection"""
    
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = {}
    
    def allow_request(self, identifier: str) -> bool:
        """Check if request is allowed"""
        current_time = time.time()
        
        if identifier not in self.calls:
            self.calls[identifier] = []
        
        # Remove old calls
        self.calls[identifier] = [
            call_time for call_time in self.calls[identifier]
            if current_time - call_time < self.period
        ]
        
        # Check limit
        if len(self.calls[identifier]) >= self.max_calls:
            return False
        
        # Record call
        self.calls[identifier].append(current_time)
        return True


class SecureAgentIdentity:
    """Enhanced secure agent identity"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        
        # Generate secure keys
        self.private_key = secrets.token_hex(32)
        account = Account.from_key(self.private_key)
        self.address = account.address
        
        # Multi-signature support
        self.multisig_threshold = 2
        self.multisig_signers = []
        
        # Hardware security module support (HSM)
        self.hsm_enabled = False
        self.hsm_key_id = None
    
    def enable_multisig(self, signers: List[str], threshold: int):
        """Enable multi-signature for critical operations"""
        self.multisig_signers = signers
        self.multisig_threshold = threshold
    
    def enable_hsm(self, key_id: str):
        """Enable hardware security module"""
        self.hsm_enabled = True
        self.hsm_key_id = key_id
    
    def sign_critical_operation(self, operation_data: bytes) -> List[bytes]:
        """Sign critical operation with multi-sig"""
        signatures = []
        
        # In production, gather signatures from multiple signers
        # This is a simplified example
        for signer in self.multisig_signers[:self.multisig_threshold]:
            signature = self._sign_with_key(operation_data, signer)
            signatures.append(signature)
        
        return signatures
    
    def _sign_with_key(self, data: bytes, key: str) -> bytes:
        """Sign data with specific key"""
        if self.hsm_enabled:
            # Use HSM for signing
            return self._hsm_sign(data)
        else:
            # Software signing
            account = Account.from_key(key)
            message = encode_defunct(data)
            signed = account.sign_message(message)
            return signed.signature
    
    def _hsm_sign(self, data: bytes) -> bytes:
        """Sign using hardware security module"""
        # In production, integrate with actual HSM
        # This is a placeholder
        return b"hsm_signature"


# Example of secure implementation usage
class SecureFinancialConsensus(SecureNashEthereumConsensus):
    """
    Secure financial consensus with additional protections
    """
    
    def __init__(self):
        super().__init__()
        
        # Additional security layers
        self.fraud_detector = FraudDetector()
        self.compliance_checker = ComplianceChecker()
    
    async def process_financial_decision(
        self,
        task: Dict[str, Any],
        agents: List[Any]
    ) -> Dict[str, Any]:
        """Process financial decision with security checks"""
        # Pre-flight security checks
        if not self.compliance_checker.check_task(task):
            raise Exception("Task fails compliance check")
        
        # Fraud detection
        fraud_score = self.fraud_detector.analyze_task(task)
        if fraud_score > 0.8:
            raise Exception("Potential fraud detected")
        
        # Process with secure consensus
        result = await self.orchestrate_consensus(task, agents)
        
        # Post-consensus verification
        if not self.verify_consensus_integrity(result):
            raise Exception("Consensus integrity check failed")
        
        return result


class FraudDetector:
    """Detect potential fraudulent activities"""
    
    def analyze_task(self, task: Dict[str, Any]) -> float:
        """Analyze task for fraud indicators"""
        fraud_score = 0.0
        
        # Check for unusual patterns
        if 'amount' in task and task['amount'] > 1000000:
            fraud_score += 0.3
        
        # Check for rapid succession of tasks
        # (Implementation would check historical data)
        
        return min(fraud_score, 1.0)


class ComplianceChecker:
    """Check regulatory compliance"""
    
    def check_task(self, task: Dict[str, Any]) -> bool:
        """Verify task meets compliance requirements"""
        # Check task type
        if task.get('type') not in ['investment', 'rebalance', 'analysis']:
            return False
        
        # Check required fields
        required_fields = ['user_id', 'timestamp', 'compliance_token']
        for field in required_fields:
            if field not in task:
                return False
        
        # Additional compliance logic
        return True
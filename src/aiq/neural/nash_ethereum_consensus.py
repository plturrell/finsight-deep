# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nash Equilibrium + Ethereum Blockchain Consensus for Neural Agents
Combines game theory with smart contracts for decentralized AI consensus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
import numpy as np
import asyncio
import hashlib
import time
import json
from enum import Enum
from collections import defaultdict
import logging
from web3 import Web3
from web3.contract import Contract
from eth_account import Account
from eth_utils import to_checksum_address

from aiq.neural.advanced_architectures import (
    FlashAttention, AttentionConfig, NeuralMemoryBank, HybridNeuralSymbolicLayer
)
from aiq.neural.reinforcement_learning import PPO, RLConfig
from aiq.hardware.tensor_core_optimizer import TensorCoreOptimizer


logger = logging.getLogger(__name__)


# Smart contract ABI for neural consensus
CONSENSUS_CONTRACT_ABI = """
[
    {
        "inputs": [],
        "name": "NeuralConsensus",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "constructor"
    },
    {
        "inputs": [
            {"name": "_agentId", "type": "address"},
            {"name": "_position", "type": "uint256[]"},
            {"name": "_confidence", "type": "uint256"},
            {"name": "_taskHash", "type": "bytes32"}
        ],
        "name": "submitPosition",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "_taskHash", "type": "bytes32"}
        ],
        "name": "computeNashEquilibrium",
        "outputs": [
            {"name": "equilibrium", "type": "uint256[]"},
            {"name": "converged", "type": "bool"}
        ],
        "stateMutability": "view",
        "type": "function"
    },
    {
        "inputs": [
            {"name": "_taskHash", "type": "bytes32"},
            {"name": "_consensus", "type": "uint256[]"}
        ],
        "name": "finalizeConsensus",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": false,
        "inputs": [
            {"indexed": true, "name": "taskHash", "type": "bytes32"},
            {"indexed": false, "name": "consensus", "type": "uint256[]"},
            {"indexed": false, "name": "timestamp", "type": "uint256"}
        ],
        "name": "ConsensusReached",
        "type": "event"
    }
]
"""

# Solidity smart contract for on-chain Nash equilibrium
CONSENSUS_CONTRACT_CODE = """
pragma solidity ^0.8.0;

contract NeuralConsensus {
    struct AgentPosition {
        address agentId;
        uint256[] position;
        uint256 confidence;
        uint256 timestamp;
    }
    
    struct ConsensusTask {
        bytes32 taskHash;
        mapping(address => AgentPosition) positions;
        address[] participants;
        uint256[] finalConsensus;
        bool isFinalized;
        uint256 createdAt;
    }
    
    mapping(bytes32 => ConsensusTask) public tasks;
    mapping(address => uint256) public agentReputation;
    
    uint256 constant POSITION_SCALING = 1e18;
    uint256 constant MIN_PARTICIPANTS = 3;
    uint256 constant CONVERGENCE_THRESHOLD = 1e16; // 0.01 in scaled units
    
    event ConsensusReached(bytes32 indexed taskHash, uint256[] consensus, uint256 timestamp);
    event PositionSubmitted(bytes32 indexed taskHash, address indexed agent, uint256 timestamp);
    
    modifier onlyRegisteredAgent() {
        require(agentReputation[msg.sender] > 0, "Not a registered agent");
        _;
    }
    
    function registerAgent(address agent) external {
        agentReputation[agent] = 100; // Initial reputation
    }
    
    function submitPosition(
        address _agentId,
        uint256[] memory _position,
        uint256 _confidence,
        bytes32 _taskHash
    ) external onlyRegisteredAgent {
        ConsensusTask storage task = tasks[_taskHash];
        
        if (task.createdAt == 0) {
            task.taskHash = _taskHash;
            task.createdAt = block.timestamp;
        }
        
        // Store position
        task.positions[_agentId] = AgentPosition({
            agentId: _agentId,
            position: _position,
            confidence: _confidence,
            timestamp: block.timestamp
        });
        
        // Add to participants if new
        bool isNewParticipant = true;
        for (uint i = 0; i < task.participants.length; i++) {
            if (task.participants[i] == _agentId) {
                isNewParticipant = false;
                break;
            }
        }
        if (isNewParticipant) {
            task.participants.push(_agentId);
        }
        
        emit PositionSubmitted(_taskHash, _agentId, block.timestamp);
    }
    
    function computeNashEquilibrium(bytes32 _taskHash) 
        external 
        view 
        returns (uint256[] memory equilibrium, bool converged) 
    {
        ConsensusTask storage task = tasks[_taskHash];
        require(task.participants.length >= MIN_PARTICIPANTS, "Not enough participants");
        
        uint256 dimensions = task.positions[task.participants[0]].position.length;
        equilibrium = new uint256[](dimensions);
        
        // Compute payoff matrix (simplified - in practice, use more sophisticated game theory)
        uint256[][] memory payoffMatrix = new uint256[][](task.participants.length);
        for (uint i = 0; i < task.participants.length; i++) {
            payoffMatrix[i] = new uint256[](task.participants.length);
            for (uint j = 0; j < task.participants.length; j++) {
                if (i != j) {
                    // Compute payoff based on position similarity and confidence
                    payoffMatrix[i][j] = computePayoff(
                        task.positions[task.participants[i]],
                        task.positions[task.participants[j]]
                    );
                }
            }
        }
        
        // Find Nash equilibrium using iterative best response
        uint256[] memory strategies = new uint256[](task.participants.length);
        for (uint i = 0; i < task.participants.length; i++) {
            strategies[i] = POSITION_SCALING / task.participants.length; // Equal initial strategies
        }
        
        converged = false;
        for (uint iteration = 0; iteration < 100; iteration++) {
            uint256[] memory newStrategies = new uint256[](task.participants.length);
            
            for (uint i = 0; i < task.participants.length; i++) {
                // Best response calculation
                uint256 bestPayoff = 0;
                uint256 bestStrategy = 0;
                
                for (uint s = 0; s <= 10; s++) {
                    uint256 strategy = (s * POSITION_SCALING) / 10;
                    uint256 expectedPayoff = 0;
                    
                    for (uint j = 0; j < task.participants.length; j++) {
                        if (i != j) {
                            expectedPayoff += (payoffMatrix[i][j] * strategies[j]) / POSITION_SCALING;
                        }
                    }
                    
                    if (expectedPayoff > bestPayoff) {
                        bestPayoff = expectedPayoff;
                        bestStrategy = strategy;
                    }
                }
                
                newStrategies[i] = bestStrategy;
            }
            
            // Check convergence
            uint256 totalChange = 0;
            for (uint i = 0; i < task.participants.length; i++) {
                uint256 change = strategies[i] > newStrategies[i] 
                    ? strategies[i] - newStrategies[i]
                    : newStrategies[i] - strategies[i];
                totalChange += change;
            }
            
            if (totalChange < CONVERGENCE_THRESHOLD) {
                converged = true;
                break;
            }
            
            strategies = newStrategies;
        }
        
        // Compute equilibrium as weighted average of positions
        for (uint d = 0; d < dimensions; d++) {
            uint256 weightedSum = 0;
            uint256 totalWeight = 0;
            
            for (uint i = 0; i < task.participants.length; i++) {
                uint256[] memory position = task.positions[task.participants[i]].position;
                uint256 weight = (strategies[i] * task.positions[task.participants[i]].confidence) / POSITION_SCALING;
                
                weightedSum += position[d] * weight;
                totalWeight += weight;
            }
            
            equilibrium[d] = weightedSum / totalWeight;
        }
        
        return (equilibrium, converged);
    }
    
    function computePayoff(AgentPosition memory pos1, AgentPosition memory pos2) 
        private 
        pure 
        returns (uint256) 
    {
        // Compute payoff based on position similarity
        uint256 similarity = 0;
        uint256 dimensions = pos1.position.length;
        
        for (uint i = 0; i < dimensions; i++) {
            uint256 diff = pos1.position[i] > pos2.position[i]
                ? pos1.position[i] - pos2.position[i]
                : pos2.position[i] - pos1.position[i];
            
            // Inverse of difference as similarity measure
            similarity += POSITION_SCALING / (1 + diff);
        }
        
        // Weight by confidence levels
        return (similarity * pos1.confidence * pos2.confidence) / (POSITION_SCALING * POSITION_SCALING);
    }
    
    function finalizeConsensus(bytes32 _taskHash, uint256[] memory _consensus) 
        external 
        onlyRegisteredAgent 
    {
        ConsensusTask storage task = tasks[_taskHash];
        require(!task.isFinalized, "Consensus already finalized");
        require(task.participants.length >= MIN_PARTICIPANTS, "Not enough participants");
        
        task.finalConsensus = _consensus;
        task.isFinalized = true;
        
        // Update agent reputations based on consensus
        for (uint i = 0; i < task.participants.length; i++) {
            address agent = task.participants[i];
            uint256 agreement = computeAgreement(
                task.positions[agent].position,
                _consensus
            );
            
            // Update reputation based on agreement with consensus
            if (agreement > 80) {
                agentReputation[agent] += 10;
            } else if (agreement < 20) {
                agentReputation[agent] = agentReputation[agent] > 10 
                    ? agentReputation[agent] - 10 
                    : 0;
            }
        }
        
        emit ConsensusReached(_taskHash, _consensus, block.timestamp);
    }
    
    function computeAgreement(uint256[] memory position, uint256[] memory consensus) 
        private 
        pure 
        returns (uint256) 
    {
        uint256 totalDiff = 0;
        uint256 dimensions = position.length;
        
        for (uint i = 0; i < dimensions; i++) {
            uint256 diff = position[i] > consensus[i]
                ? position[i] - consensus[i]
                : consensus[i] - position[i];
            totalDiff += diff;
        }
        
        // Return agreement as percentage (0-100)
        uint256 avgDiff = totalDiff / dimensions;
        return avgDiff < POSITION_SCALING 
            ? 100 - (avgDiff * 100) / POSITION_SCALING
            : 0;
    }
}
"""


@dataclass
class EthereumAgentIdentity:
    """Ethereum identity for neural agents"""
    address: str
    private_key: str
    public_key: str
    ens_name: Optional[str] = None
    reputation: int = 100


@dataclass
class NashEthereumState:
    """State of Nash-Ethereum consensus"""
    task_hash: str
    contract_address: str
    participants: List[str]
    positions: Dict[str, torch.Tensor]
    nash_equilibrium: Optional[torch.Tensor]
    on_chain_consensus: Optional[List[float]]
    gas_used: int
    block_number: int
    converged: bool


class NashEthereumConsensus:
    """
    Hybrid Nash Equilibrium + Ethereum consensus mechanism
    Combines game theory with smart contracts for trustless AI coordination
    """
    
    def __init__(
        self,
        web3_provider: str = "http://localhost:8545",
        contract_address: Optional[str] = None,
        chain_id: int = 1337,  # Local development chain
        device: str = "cuda"
    ):
        self.w3 = Web3(Web3.HTTPProvider(web3_provider))
        self.chain_id = chain_id
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # Deploy or load consensus contract
        if contract_address:
            self.contract = self.w3.eth.contract(
                address=to_checksum_address(contract_address),
                abi=json.loads(CONSENSUS_CONTRACT_ABI)
            )
        else:
            self.contract = self._deploy_consensus_contract()
        
        # Agent registry
        self.agents: Dict[str, 'EthereumNeuralAgent'] = {}
        
        # GPU optimization for Nash computation
        self.tensor_optimizer = TensorCoreOptimizer()
        
        # Cache for off-chain Nash computations
        self.nash_cache = {}
        
        logger.info(f"Initialized Nash-Ethereum consensus on chain {chain_id}")
    
    def _deploy_consensus_contract(self) -> Contract:
        """Deploy the consensus smart contract"""
        # In production, this would compile and deploy the Solidity contract
        # For now, we'll use a mock deployment
        logger.info("Deploying consensus contract (mock)")
        
        # Mock contract address
        contract_address = "0x" + "0" * 40
        
        return self.w3.eth.contract(
            address=to_checksum_address(contract_address),
            abi=json.loads(CONSENSUS_CONTRACT_ABI)
        )
    
    def create_agent_identity(self, agent_id: str) -> EthereumAgentIdentity:
        """Create Ethereum identity for a neural agent"""
        # Generate new account
        account = Account.create()
        
        identity = EthereumAgentIdentity(
            address=account.address,
            private_key=account.key.hex(),
            public_key=account._key_obj.public_key.to_hex(),
            ens_name=f"{agent_id}.neural.eth"  # Mock ENS name
        )
        
        # Register agent on-chain (in production)
        # self.contract.functions.registerAgent(identity.address).transact()
        
        return identity
    
    def compute_nash_equilibrium_gpu(
        self,
        positions: torch.Tensor,
        confidences: torch.Tensor,
        max_iterations: int = 1000
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """
        Compute Nash equilibrium on GPU for efficiency
        
        Args:
            positions: Agent positions tensor (num_agents, dimensions)
            confidences: Agent confidence scores
            max_iterations: Maximum iterations for convergence
            
        Returns:
            Equilibrium strategies, final consensus, convergence flag
        """
        num_agents, dimensions = positions.shape
        
        # Initialize on GPU
        positions = positions.to(self.device)
        confidences = confidences.to(self.device)
        
        # Compute payoff matrix using GPU acceleration
        payoff_matrix = self._compute_payoff_matrix_gpu(positions, confidences)
        
        # Initialize strategies
        strategies = torch.ones(num_agents, device=self.device) / num_agents
        
        converged = False
        
        # Optimize with Tensor Cores if available
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                for iteration in range(max_iterations):
                    old_strategies = strategies.clone()
                    
                    # Parallel best response computation
                    expected_payoffs = torch.matmul(payoff_matrix, strategies)
                    best_responses = F.softmax(expected_payoffs * 10, dim=0)  # Temperature scaling
                    
                    # Update strategies with momentum
                    strategies = 0.9 * strategies + 0.1 * best_responses
                    
                    # Check convergence
                    if torch.norm(strategies - old_strategies) < 1e-6:
                        converged = True
                        break
        
        # Compute final consensus as weighted average
        weighted_positions = positions * strategies.unsqueeze(1) * confidences.unsqueeze(1)
        consensus = weighted_positions.sum(dim=0) / (strategies * confidences).sum()
        
        return strategies, consensus, converged
    
    def _compute_payoff_matrix_gpu(
        self,
        positions: torch.Tensor,
        confidences: torch.Tensor
    ) -> torch.Tensor:
        """Compute payoff matrix on GPU using batch operations"""
        num_agents = positions.shape[0]
        
        # Compute pairwise distances (vectorized)
        positions_expanded = positions.unsqueeze(0)  # (1, num_agents, dimensions)
        positions_transposed = positions.unsqueeze(1)  # (num_agents, 1, dimensions)
        
        # Cosine similarity as payoff basis
        similarities = F.cosine_similarity(
            positions_expanded,
            positions_transposed,
            dim=2
        )
        
        # Weight by confidence
        confidence_matrix = confidences.unsqueeze(0) * confidences.unsqueeze(1)
        payoff_matrix = similarities * confidence_matrix
        
        # Zero diagonal (no self-payoff)
        payoff_matrix.fill_diagonal_(0)
        
        return payoff_matrix
    
    async def submit_position_on_chain(
        self,
        agent: 'EthereumNeuralAgent',
        position: torch.Tensor,
        confidence: float,
        task_hash: str
    ) -> str:
        """Submit agent position to Ethereum smart contract"""
        # Convert position to integer representation for Solidity
        position_scaled = (position * 1e18).long().tolist()
        confidence_scaled = int(confidence * 1e18)
        
        # Prepare transaction
        tx = self.contract.functions.submitPosition(
            agent.identity.address,
            position_scaled,
            confidence_scaled,
            bytes.fromhex(task_hash)
        ).build_transaction({
            'from': agent.identity.address,
            'gas': 300000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(agent.identity.address),
            'chainId': self.chain_id
        })
        
        # Sign transaction
        signed_tx = Account.sign_transaction(tx, agent.identity.private_key)
        
        # Send transaction
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        logger.info(f"Position submitted on-chain: {tx_hash.hex()}")
        
        return tx_hash.hex()
    
    async def orchestrate_consensus(
        self,
        task: Dict[str, Any],
        agents: List['EthereumNeuralAgent'],
        hybrid_mode: bool = True
    ) -> NashEthereumState:
        """
        Orchestrate consensus using Nash equilibrium and Ethereum
        
        Args:
            task: Task specification
            agents: List of participating agents
            hybrid_mode: Use both off-chain Nash and on-chain consensus
            
        Returns:
            Final consensus state
        """
        # Generate task hash
        task_str = json.dumps(task, sort_keys=True)
        task_hash = hashlib.sha256(task_str.encode()).hexdigest()
        
        # Phase 1: Agents compute positions
        positions = []
        confidences = []
        
        for agent in agents:
            position, confidence = await agent.compute_position(task)
            positions.append(position)
            confidences.append(confidence)
            
            # Submit to blockchain
            await self.submit_position_on_chain(
                agent, position, confidence, task_hash
            )
        
        positions_tensor = torch.stack(positions)
        confidences_tensor = torch.tensor(confidences, device=self.device)
        
        # Phase 2: Compute Nash equilibrium off-chain (fast GPU)
        nash_strategies, nash_consensus, converged = None, None, False
        
        if hybrid_mode:
            nash_strategies, nash_consensus, converged = self.compute_nash_equilibrium_gpu(
                positions_tensor, confidences_tensor
            )
            
            # Cache result
            self.nash_cache[task_hash] = {
                "strategies": nash_strategies,
                "consensus": nash_consensus,
                "converged": converged
            }
        
        # Phase 3: Compute on-chain consensus
        on_chain_result = self.contract.functions.computeNashEquilibrium(
            bytes.fromhex(task_hash)
        ).call()
        
        on_chain_consensus = [x / 1e18 for x in on_chain_result[0]]
        on_chain_converged = on_chain_result[1]
        
        # Phase 4: Finalize consensus
        if hybrid_mode and converged:
            # Use off-chain Nash result if it converged
            final_consensus = nash_consensus
        else:
            # Use on-chain result
            final_consensus = torch.tensor(on_chain_consensus, device=self.device)
        
        # Submit final consensus to blockchain
        final_consensus_scaled = (final_consensus * 1e18).long().tolist()
        
        tx = self.contract.functions.finalizeConsensus(
            bytes.fromhex(task_hash),
            final_consensus_scaled
        ).build_transaction({
            'from': agents[0].identity.address,
            'gas': 500000,
            'gasPrice': self.w3.eth.gas_price,
            'nonce': self.w3.eth.get_transaction_count(agents[0].identity.address),
            'chainId': self.chain_id
        })
        
        signed_tx = Account.sign_transaction(tx, agents[0].identity.private_key)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Create final state
        state = NashEthereumState(
            task_hash=task_hash,
            contract_address=self.contract.address,
            participants=[agent.identity.address for agent in agents],
            positions={agent.agent_id: pos for agent, pos in zip(agents, positions)},
            nash_equilibrium=nash_consensus,
            on_chain_consensus=on_chain_consensus,
            gas_used=receipt['gasUsed'],
            block_number=receipt['blockNumber'],
            converged=converged and on_chain_converged
        )
        
        return state
    
    def verify_consensus_integrity(self, state: NashEthereumState) -> bool:
        """Verify consensus integrity using blockchain"""
        # Check on-chain consensus matches recorded state
        on_chain_result = self.contract.functions.computeNashEquilibrium(
            bytes.fromhex(state.task_hash)
        ).call()
        
        on_chain_consensus = [x / 1e18 for x in on_chain_result[0]]
        
        # Compare with recorded consensus
        recorded_consensus = state.on_chain_consensus
        
        return np.allclose(on_chain_consensus, recorded_consensus, rtol=1e-6)
    
    async def resolve_dispute(
        self,
        disputing_agents: List['EthereumNeuralAgent'],
        disputed_state: NashEthereumState
    ) -> Dict[str, Any]:
        """Resolve disputes using on-chain arbitration"""
        # Get current agent reputations from blockchain
        reputations = {}
        for agent in disputing_agents:
            rep = self.contract.functions.agentReputation(agent.identity.address).call()
            reputations[agent.agent_id] = rep
        
        # Re-compute consensus with reputation weighting
        positions = []
        weights = []
        
        for agent in disputing_agents:
            if agent.agent_id in disputed_state.positions:
                positions.append(disputed_state.positions[agent.agent_id])
                weights.append(reputations[agent.agent_id])
        
        if positions:
            positions_tensor = torch.stack(positions)
            weights_tensor = torch.tensor(weights, device=self.device, dtype=torch.float)
            weights_tensor = F.softmax(weights_tensor, dim=0)
            
            # Reputation-weighted consensus
            weighted_consensus = (positions_tensor * weights_tensor.unsqueeze(1)).sum(dim=0)
            
            return {
                "resolved_consensus": weighted_consensus,
                "reputations": reputations,
                "resolution_method": "reputation_weighted"
            }
        
        return {
            "resolved_consensus": None,
            "reputations": reputations,
            "resolution_method": "failed"
        }


class EthereumNeuralAgent:
    """Neural agent with Ethereum identity and smart contract interaction"""
    
    def __init__(
        self,
        agent_id: str,
        model: nn.Module,
        identity: EthereumAgentIdentity,
        device: str = "cuda"
    ):
        self.agent_id = agent_id
        self.model = model.to(device)
        self.identity = identity
        self.device = device
        
        # Hybrid neural-symbolic layer for Web3 integration
        self.web3_integration = HybridNeuralSymbolicLayer(
            neural_dim=256,
            symbolic_dim=128,
            num_rules=50
        ).to(device)
        
        # Memory of on-chain interactions
        self.on_chain_memory = []
        
        # RL component for learning from consensus outcomes
        rl_config = RLConfig(
            algo="ppo",
            lr_actor=1e-4,
            lr_critic=1e-4
        )
        self.rl_agent = PPO(
            state_dim=256,
            action_dim=256,
            config=rl_config,
            continuous=True
        )
    
    async def compute_position(
        self,
        task: Dict[str, Any]
    ) -> Tuple[torch.Tensor, float]:
        """Compute position on a task with confidence"""
        # Encode task
        task_embedding = self._encode_task(task)
        
        # Neural computation
        with torch.no_grad():
            neural_output = self.model(task_embedding)
            
            # Integrate Web3 context
            position = self.web3_integration(neural_output)
            
            # Compute confidence based on output entropy
            confidence = self._compute_confidence(position)
        
        return position.squeeze(0), confidence
    
    def _encode_task(self, task: Dict[str, Any]) -> torch.Tensor:
        """Encode task to tensor representation"""
        # Convert task to embedding (simplified)
        task_str = json.dumps(task, sort_keys=True)
        task_bytes = task_str.encode('utf-8')
        
        # Use hash as seed for deterministic encoding
        hash_int = int(hashlib.sha256(task_bytes).hexdigest()[:16], 16)
        torch.manual_seed(hash_int)
        
        # Generate structured encoding
        encoding = torch.randn(1, 256, device=self.device)
        encoding = F.normalize(encoding, dim=1)
        
        return encoding
    
    def _compute_confidence(self, position: torch.Tensor) -> float:
        """Compute confidence score from position tensor"""
        # Use entropy as inverse confidence
        if position.dim() > 1:
            position = position.squeeze(0)
        
        # Normalize to probability distribution
        probs = F.softmax(position, dim=0)
        
        # Compute entropy
        entropy = -torch.sum(probs * torch.log(probs + 1e-8))
        max_entropy = torch.log(torch.tensor(position.shape[0], dtype=torch.float))
        
        # Confidence is inverse of normalized entropy
        confidence = 1.0 - (entropy / max_entropy).item()
        
        return confidence
    
    async def learn_from_consensus(
        self,
        consensus_state: NashEthereumState,
        reward: float
    ):
        """Learn from consensus outcome using RL"""
        # Store experience
        if self.agent_id in consensus_state.positions:
            own_position = consensus_state.positions[self.agent_id]
            consensus_position = consensus_state.nash_equilibrium
            
            if consensus_position is not None:
                # Store for RL update
                self.rl_agent.store(
                    state=own_position.cpu().numpy(),
                    action=consensus_position.cpu().numpy(),
                    reward=reward,
                    value=0.0,  # Will be computed by critic
                    log_prob=0.0,  # Will be computed
                    done=True
                )
                
                # Store on-chain interaction
                self.on_chain_memory.append({
                    "task_hash": consensus_state.task_hash,
                    "position": own_position,
                    "consensus": consensus_position,
                    "reward": reward,
                    "gas_used": consensus_state.gas_used,
                    "block_number": consensus_state.block_number
                })
                
                # Periodic RL update
                if len(self.on_chain_memory) % 10 == 0:
                    self.rl_agent.update()
    
    def estimate_gas_cost(self, position_dim: int) -> int:
        """Estimate gas cost for submitting position"""
        # Base cost + cost per dimension
        base_cost = 21000  # Base transaction cost
        storage_cost = 20000  # Per slot storage
        computation_cost = position_dim * 5000  # Per dimension computation
        
        return base_cost + storage_cost + computation_cost


# Example usage for financial consensus
class FinancialNashEthereumConsensus(NashEthereumConsensus):
    """
    Financial market consensus using Nash equilibrium and Ethereum
    """
    
    def __init__(self, num_agents: int = 5):
        super().__init__()
        
        # Create financial agents with different strategies
        self.strategies = ["value", "growth", "momentum", "contrarian", "quant"]
        self._create_financial_agents(num_agents)
    
    def _create_financial_agents(self, num_agents: int):
        """Create financial agents with Ethereum identities"""
        for i in range(num_agents):
            strategy = self.strategies[i % len(self.strategies)]
            
            # Create agent model
            model = nn.Sequential(
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Tanh()
            )
            
            # Create Ethereum identity
            identity = self.create_agent_identity(f"{strategy}_agent_{i}")
            
            # Create agent
            agent = EthereumNeuralAgent(
                agent_id=f"{strategy}_agent_{i}",
                model=model,
                identity=identity,
                device=self.device.type
            )
            
            self.agents[agent.agent_id] = agent
    
    async def make_market_prediction(
        self,
        market_data: Dict[str, Any],
        security: str
    ) -> Dict[str, Any]:
        """
        Make consensus market prediction with blockchain verification
        
        Args:
            market_data: Current market conditions
            security: Security to predict
            
        Returns:
            Consensus prediction with blockchain proof
        """
        task = {
            "type": "market_prediction",
            "security": security,
            "market_data": market_data,
            "timestamp": time.time()
        }
        
        # Get consensus from all agents
        agents_list = list(self.agents.values())
        consensus_state = await self.orchestrate_consensus(
            task, agents_list, hybrid_mode=True
        )
        
        # Interpret consensus as price prediction
        price_prediction = consensus_state.nash_equilibrium
        
        # Verify on blockchain
        verified = self.verify_consensus_integrity(consensus_state)
        
        prediction = {
            "security": security,
            "predicted_price": price_prediction[0].item() if price_prediction is not None else None,
            "confidence": 1.0 - torch.std(price_prediction).item() if price_prediction is not None else 0.0,
            "consensus_method": "nash_ethereum_hybrid",
            "blockchain_verified": verified,
            "block_number": consensus_state.block_number,
            "gas_cost": consensus_state.gas_used,
            "participating_agents": consensus_state.participants,
            "converged": consensus_state.converged
        }
        
        # Record prediction on-chain for accountability
        # In production, this would create an immutable record
        
        return prediction
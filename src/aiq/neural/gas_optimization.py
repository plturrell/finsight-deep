# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Gas optimization strategies for Nash-Ethereum consensus
Includes batch submissions and Layer 2 integration
"""

import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import asyncio
import time
from web3 import Web3
from eth_abi import encode_abi
import logging

from aiq.neural.nash_ethereum_consensus import NashEthereumConsensus


logger = logging.getLogger(__name__)


@dataclass
class BatchSubmission:
    """Batch of positions for gas-efficient submission"""
    task_hash: str
    positions: List[Tuple[str, torch.Tensor, float]]  # (agent_id, position, confidence)
    signatures: List[bytes]
    timestamp: float


@dataclass
class Layer2Config:
    """Configuration for Layer 2 networks"""
    network: str  # polygon, arbitrum, optimism
    bridge_address: str
    rpc_url: str
    chain_id: int
    confirmation_blocks: int = 5


class GasOptimizedConsensus(NashEthereumConsensus):
    """
    Gas-optimized version of Nash-Ethereum consensus
    Implements batching and Layer 2 strategies
    """
    
    def __init__(
        self,
        web3_provider: str,
        layer2_config: Optional[Layer2Config] = None,
        batch_size: int = 10,
        batch_timeout: float = 30.0
    ):
        super().__init__(web3_provider)
        
        self.layer2_config = layer2_config
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        
        # Batch queue
        self.pending_submissions: List[Dict[str, Any]] = []
        self.batch_lock = asyncio.Lock()
        
        # Layer 2 Web3 instance
        if layer2_config:
            self.l2_w3 = Web3(Web3.HTTPProvider(layer2_config.rpc_url))
            self.l2_contract = self._init_l2_contract()
        
        # Gas price oracle
        self.gas_oracle = GasPriceOracle(self.w3)
        
        # Start batch processor
        asyncio.create_task(self._batch_processor())
    
    def _init_l2_contract(self):
        """Initialize Layer 2 contract"""
        # Deploy or connect to L2 consensus contract
        # This would use the same ABI but deployed on L2
        return self.l2_w3.eth.contract(
            address=self.layer2_config.bridge_address,
            abi=self.contract.abi
        )
    
    async def submit_position_optimized(
        self,
        agent_id: str,
        position: torch.Tensor,
        confidence: float,
        task_hash: str,
        priority: str = "medium"
    ) -> Dict[str, Any]:
        """
        Submit position with gas optimization
        
        Args:
            agent_id: Agent identifier
            position: Position tensor
            confidence: Confidence score
            task_hash: Task hash
            priority: Submission priority (low, medium, high)
        
        Returns:
            Submission status
        """
        # Check if we should use L2
        if self.layer2_config and self._should_use_l2(priority):
            return await self._submit_to_l2(agent_id, position, confidence, task_hash)
        
        # Otherwise, add to batch queue
        submission = {
            "agent_id": agent_id,
            "position": position,
            "confidence": confidence,
            "task_hash": task_hash,
            "timestamp": time.time(),
            "priority": priority
        }
        
        async with self.batch_lock:
            self.pending_submissions.append(submission)
            
            # Check if we should process batch immediately
            if len(self.pending_submissions) >= self.batch_size:
                await self._process_batch()
        
        return {
            "status": "queued",
            "batch_size": len(self.pending_submissions),
            "estimated_gas_savings": self._estimate_gas_savings()
        }
    
    def _should_use_l2(self, priority: str) -> bool:
        """Determine if submission should use Layer 2"""
        if not self.layer2_config:
            return False
        
        # Use L2 for low priority or when mainnet is congested
        gas_price = self.gas_oracle.get_current_price()
        
        if priority == "low":
            return True
        elif priority == "medium" and gas_price > Web3.toWei(100, 'gwei'):
            return True
        
        return False
    
    async def _submit_to_l2(
        self,
        agent_id: str,
        position: torch.Tensor,
        confidence: float,
        task_hash: str
    ) -> Dict[str, Any]:
        """Submit position to Layer 2 network"""
        # Convert position for L2
        position_scaled = (position * 1e18).long().tolist()
        confidence_scaled = int(confidence * 1e18)
        
        # Build L2 transaction
        tx = self.l2_contract.functions.submitPosition(
            agent_id,
            position_scaled,
            confidence_scaled,
            bytes.fromhex(task_hash)
        ).build_transaction({
            'from': self.account.address,
            'gas': 100000,  # Much lower gas on L2
            'gasPrice': self.l2_w3.eth.gas_price,
            'nonce': self.l2_w3.eth.get_transaction_count(self.account.address),
            'chainId': self.layer2_config.chain_id
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.l2_w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        # Wait for L2 confirmation
        receipt = self.l2_w3.eth.wait_for_transaction_receipt(tx_hash)
        
        return {
            "status": "submitted_l2",
            "tx_hash": tx_hash.hex(),
            "network": self.layer2_config.network,
            "gas_used": receipt['gasUsed'],
            "block_number": receipt['blockNumber']
        }
    
    async def _batch_processor(self):
        """Background task to process batches"""
        while True:
            await asyncio.sleep(self.batch_timeout)
            
            async with self.batch_lock:
                if self.pending_submissions:
                    await self._process_batch()
    
    async def _process_batch(self):
        """Process a batch of submissions"""
        if not self.pending_submissions:
            return
        
        batch = self.pending_submissions[:self.batch_size]
        self.pending_submissions = self.pending_submissions[self.batch_size:]
        
        # Group by task hash
        task_batches = {}
        for submission in batch:
            task_hash = submission['task_hash']
            if task_hash not in task_batches:
                task_batches[task_hash] = []
            task_batches[task_hash].append(submission)
        
        # Submit each task batch
        for task_hash, submissions in task_batches.items():
            await self._submit_batch(task_hash, submissions)
    
    async def _submit_batch(self, task_hash: str, submissions: List[Dict[str, Any]]):
        """Submit a batch of positions for a single task"""
        # Prepare batch data
        agent_ids = []
        positions = []
        confidences = []
        
        for sub in submissions:
            agent_ids.append(sub['agent_id'])
            positions.append((sub['position'] * 1e18).long().tolist())
            confidences.append(int(sub['confidence'] * 1e18))
        
        # Encode batch data
        batch_data = encode_abi(
            ['address[]', 'uint256[][]', 'uint256[]'],
            [agent_ids, positions, confidences]
        )
        
        # Use multicall contract for batch submission
        multicall_address = self.config.get('multicall_address')
        multicall = self.w3.eth.contract(
            address=multicall_address,
            abi=MULTICALL_ABI
        )
        
        # Build batch transaction
        tx = multicall.functions.aggregate([
            (self.contract.address, self.contract.encodeABI(
                fn_name='submitPosition',
                args=[agent_id, pos, conf, bytes.fromhex(task_hash)]
            ))
            for agent_id, pos, conf in zip(agent_ids, positions, confidences)
        ]).build_transaction({
            'from': self.account.address,
            'gas': 500000,  # Adjust based on batch size
            'gasPrice': self.gas_oracle.get_optimal_price(),
            'nonce': self.w3.eth.get_transaction_count(self.account.address)
        })
        
        # Sign and send
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Batch submitted: {len(submissions)} positions in tx {tx_hash.hex()}")
        
        # Wait for confirmation
        receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
        
        # Notify agents of batch result
        for sub in submissions:
            # Implement notification mechanism
            pass
    
    def _estimate_gas_savings(self) -> Dict[str, float]:
        """Estimate gas savings from batching"""
        if not self.pending_submissions:
            return {"percentage": 0, "amount_wei": 0}
        
        # Individual submission cost
        individual_gas = 150000  # Estimated gas per submission
        individual_total = individual_gas * len(self.pending_submissions)
        
        # Batch submission cost
        batch_base_gas = 50000
        batch_per_item_gas = 30000
        batch_total = batch_base_gas + (batch_per_item_gas * len(self.pending_submissions))
        
        # Savings
        savings_wei = individual_total - batch_total
        savings_percentage = (savings_wei / individual_total) * 100
        
        return {
            "percentage": savings_percentage,
            "amount_wei": savings_wei,
            "amount_gwei": savings_wei / 1e9
        }


class GasPriceOracle:
    """Oracle for optimal gas price determination"""
    
    def __init__(self, w3: Web3):
        self.w3 = w3
        self.price_history = []
        self.last_update = 0
        self.update_interval = 60  # seconds
    
    def get_current_price(self) -> int:
        """Get current gas price"""
        return self.w3.eth.gas_price
    
    def get_optimal_price(self, speed: str = "medium") -> int:
        """
        Get optimal gas price based on desired speed
        
        Args:
            speed: Transaction speed (slow, medium, fast)
        
        Returns:
            Optimal gas price in wei
        """
        current_price = self.get_current_price()
        
        # Adjust based on speed preference
        multipliers = {
            "slow": 0.8,
            "medium": 1.0,
            "fast": 1.5
        }
        
        return int(current_price * multipliers.get(speed, 1.0))
    
    def should_wait(self) -> bool:
        """Determine if we should wait for better gas prices"""
        current_price = self.get_current_price()
        
        # Wait if price is above threshold
        threshold = Web3.toWei(150, 'gwei')
        return current_price > threshold
    
    def predict_best_time(self) -> float:
        """Predict best time to submit transaction"""
        # Simplified prediction - in production use ML model
        # Based on historical patterns
        current_hour = time.localtime().tm_hour
        
        # Gas is typically cheaper during off-peak hours
        off_peak_hours = [2, 3, 4, 5, 6]  # UTC
        
        if current_hour in off_peak_hours:
            return 0  # Submit now
        else:
            # Wait until next off-peak
            next_off_peak = min(h for h in off_peak_hours if h > current_hour)
            hours_to_wait = next_off_peak - current_hour
            return hours_to_wait * 3600  # Convert to seconds


# Multicall ABI for batch operations
MULTICALL_ABI = [
    {
        "inputs": [
            {
                "components": [
                    {"name": "target", "type": "address"},
                    {"name": "callData", "type": "bytes"}
                ],
                "name": "calls",
                "type": "tuple[]"
            }
        ],
        "name": "aggregate",
        "outputs": [
            {"name": "blockNumber", "type": "uint256"},
            {"name": "returnData", "type": "bytes[]"}
        ],
        "stateMutability": "nonpayable",
        "type": "function"
    }
]


# Example usage for practical demonstrations
class SimpleTaskConsensus(GasOptimizedConsensus):
    """
    Simplified consensus for practical examples
    Lower stakes than financial decisions
    """
    
    async def collaborative_content_ranking(
        self,
        content_items: List[Dict[str, Any]],
        agents: List[Any]
    ) -> Dict[str, Any]:
        """
        Example: Collaborative content ranking/curation
        Multiple AI agents rank content for quality
        """
        task = {
            "type": "content_ranking",
            "items": content_items,
            "criteria": ["quality", "relevance", "originality"],
            "timestamp": time.time()
        }
        
        # Get agent rankings
        rankings = []
        for agent in agents:
            position, confidence = await agent.rank_content(content_items)
            
            # Submit with gas optimization
            await self.submit_position_optimized(
                agent.agent_id,
                position,
                confidence,
                task_hash=self._hash_task(task),
                priority="low"  # Not time-critical
            )
            
            rankings.append((agent.agent_id, position))
        
        # Wait for consensus
        await self._wait_for_batch_processing()
        
        # Get consensus ranking
        consensus_state = await self.get_consensus_state(self._hash_task(task))
        
        return {
            "consensus_ranking": consensus_state.consensus_value,
            "individual_rankings": rankings,
            "gas_saved": self._estimate_gas_savings()
        }
    
    async def distributed_model_evaluation(
        self,
        model_outputs: List[torch.Tensor],
        evaluation_agents: List[Any]
    ) -> Dict[str, Any]:
        """
        Example: Distributed ML model evaluation
        Multiple agents evaluate model outputs for quality
        """
        task = {
            "type": "model_evaluation",
            "model_id": "example_model_v1",
            "outputs": len(model_outputs),
            "timestamp": time.time()
        }
        
        evaluations = []
        
        # Batch evaluations for gas efficiency
        async with self.batch_lock:
            for agent in evaluation_agents:
                score, confidence = await agent.evaluate_model(model_outputs)
                
                self.pending_submissions.append({
                    "agent_id": agent.agent_id,
                    "position": score,
                    "confidence": confidence,
                    "task_hash": self._hash_task(task),
                    "priority": "medium"
                })
                
                evaluations.append((agent.agent_id, score))
        
        # Process batch
        await self._process_batch()
        
        return {
            "consensus_score": await self.get_consensus_score(task),
            "individual_scores": evaluations,
            "evaluation_cost": self._calculate_gas_cost()
        }
    
    def _hash_task(self, task: Dict[str, Any]) -> str:
        """Generate task hash"""
        import hashlib
        import json
        
        task_str = json.dumps(task, sort_keys=True)
        return hashlib.sha256(task_str.encode()).hexdigest()
    
    async def _wait_for_batch_processing(self):
        """Wait for current batch to be processed"""
        while self.pending_submissions:
            await asyncio.sleep(1)
    
    def _calculate_gas_cost(self) -> Dict[str, float]:
        """Calculate actual gas cost in USD"""
        gas_price = self.gas_oracle.get_current_price()
        eth_price = 2000  # Mock ETH price, use oracle in production
        
        gas_used = 50000  # Estimated for batch
        cost_eth = (gas_used * gas_price) / 1e18
        cost_usd = cost_eth * eth_price
        
        return {
            "gas_used": gas_used,
            "cost_eth": cost_eth,
            "cost_usd": cost_usd
        }
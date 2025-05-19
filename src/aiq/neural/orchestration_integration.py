# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Integration of Nash-Ethereum consensus with existing Digital Human orchestration
Provides seamless multi-agent coordination for financial advisory system
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
import asyncio
import time
import json
import logging
from dataclasses import dataclass

from aiq.neural.nash_ethereum_consensus import (
    NashEthereumConsensus, EthereumNeuralAgent, EthereumAgentIdentity,
    NashEthereumState
)
from aiq.digital_human.orchestrator.digital_human_orchestrator import (
    DigitalHumanOrchestrator, InteractionSession
)
from aiq.digital_human.conversation.conversation_orchestrator import (
    ConversationOrchestrator
)
from aiq.digital_human.persistence.jena_database import JenaPersistenceManager
from aiq.neural.knowledge_integration import NeuralJenaIntegration


logger = logging.getLogger(__name__)


@dataclass
class MultiAgentDecision:
    """Decision made by multiple AI agents"""
    decision_id: str
    decision_type: str
    consensus_value: torch.Tensor
    individual_positions: Dict[str, torch.Tensor]
    confidence: float
    blockchain_proof: str
    jena_uri: str
    timestamp: float


class EnhancedDigitalHumanOrchestrator(DigitalHumanOrchestrator):
    """
    Enhanced orchestrator with Nash-Ethereum consensus for multi-agent decisions
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        enable_multi_agent: bool = True,
        num_reasoning_agents: int = 5
    ):
        super().__init__(config)
        
        self.enable_multi_agent = enable_multi_agent
        self.num_reasoning_agents = num_reasoning_agents
        
        if enable_multi_agent:
            # Initialize Nash-Ethereum consensus
            self.consensus_system = NashEthereumConsensus(
                web3_provider=config.get("web3_provider", "http://localhost:8545"),
                device=self.device
            )
            
            # Initialize Neural-Jena integration
            jena_config = {
                "jena_endpoint": config.get("jena_endpoint", "http://localhost:3030"),
                "dataset_name": "digital_human_consensus",
                "embedding_dim": 768,
                "use_gpu": True
            }
            self.neural_jena = NeuralJenaIntegration(jena_config)
            
            # Create reasoning agents
            self._create_reasoning_agents()
            
            # Decision history
            self.decision_history: List[MultiAgentDecision] = []
    
    def _create_reasoning_agents(self):
        """Create specialized reasoning agents with different perspectives"""
        agent_types = [
            "risk_analyst",
            "growth_optimizer", 
            "tax_strategist",
            "market_analyst",
            "portfolio_manager"
        ]
        
        for i in range(self.num_reasoning_agents):
            agent_type = agent_types[i % len(agent_types)]
            
            # Create specialized model for each agent type
            model = self._create_agent_model(agent_type)
            
            # Create Ethereum identity
            identity = self.consensus_system.create_agent_identity(
                f"{agent_type}_{i}"
            )
            
            # Create agent
            agent = EthereumNeuralAgent(
                agent_id=f"{agent_type}_{i}",
                model=model,
                identity=identity,
                device=self.device
            )
            
            self.consensus_system.agents[agent.agent_id] = agent
    
    def _create_agent_model(self, agent_type: str) -> nn.Module:
        """Create specialized model based on agent type"""
        # Base architecture
        model = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        
        # Specialized heads based on agent type
        if agent_type == "risk_analyst":
            model.add_module("risk_head", nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Sigmoid()  # Risk scores between 0-1
            ))
        elif agent_type == "growth_optimizer":
            model.add_module("growth_head", nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.Softplus()  # Positive growth predictions
            ))
        else:
            model.add_module("general_head", nn.Sequential(
                nn.Linear(512, 256),
                nn.Tanh()  # General bounded output
            ))
        
        return model
    
    async def process_complex_decision(
        self,
        session_id: str,
        decision_request: Dict[str, Any]
    ) -> MultiAgentDecision:
        """
        Process complex decision using multi-agent consensus
        
        Args:
            session_id: Current session ID
            decision_request: Decision request details
            
        Returns:
            Multi-agent consensus decision
        """
        start_time = time.time()
        
        # Prepare task for agents
        task = {
            "session_id": session_id,
            "type": decision_request["type"],
            "context": await self._gather_context(session_id),
            "parameters": decision_request.get("parameters", {}),
            "timestamp": start_time
        }
        
        # Get consensus from agents
        agents_list = list(self.consensus_system.agents.values())
        consensus_state = await self.consensus_system.orchestrate_consensus(
            task, agents_list, hybrid_mode=True
        )
        
        # Store decision in Jena
        decision_uri = await self._store_decision_in_jena(
            consensus_state, decision_request
        )
        
        # Create decision object
        decision = MultiAgentDecision(
            decision_id=consensus_state.task_hash,
            decision_type=decision_request["type"],
            consensus_value=consensus_state.nash_equilibrium,
            individual_positions=consensus_state.positions,
            confidence=self._calculate_confidence(consensus_state),
            blockchain_proof=consensus_state.task_hash,
            jena_uri=decision_uri,
            timestamp=time.time()
        )
        
        # Store in history
        self.decision_history.append(decision)
        
        # Update agents with outcome
        await self._update_agents_with_outcome(consensus_state, decision)
        
        return decision
    
    async def _gather_context(self, session_id: str) -> Dict[str, Any]:
        """Gather relevant context for decision making"""
        context = {}
        
        # Get user profile from Jena
        user_query = f"""
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        PREFIX user: <http://aiqtoolkit.com/user/>
        
        SELECT ?property ?value
        WHERE {{
            ?session dh:sessionId "{session_id}" ;
                     dh:userId ?user .
            ?user ?property ?value .
        }}
        """
        
        user_data = await self.neural_jena.jena_manager.execute_sparql_query(user_query)
        context["user_profile"] = user_data
        
        # Get recent interactions
        interaction_query = f"""
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        
        SELECT ?content ?timestamp
        WHERE {{
            ?session dh:sessionId "{session_id}" ;
                     dh:hasInteraction ?interaction .
            ?interaction dh:content ?content ;
                        dh:timestamp ?timestamp .
        }}
        ORDER BY DESC(?timestamp)
        LIMIT 10
        """
        
        interactions = await self.neural_jena.jena_manager.execute_sparql_query(
            interaction_query
        )
        context["recent_interactions"] = interactions
        
        # Get portfolio data if available
        portfolio_query = f"""
        PREFIX finance: <http://aiqtoolkit.com/finance/>
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        
        SELECT ?portfolio ?holdings ?value
        WHERE {{
            ?session dh:sessionId "{session_id}" ;
                     dh:userId ?user .
            ?user dh:hasPortfolio ?portfolio .
            ?portfolio finance:holdings ?holdings ;
                      finance:totalValue ?value .
        }}
        """
        
        portfolio_data = await self.neural_jena.jena_manager.execute_sparql_query(
            portfolio_query
        )
        context["portfolio"] = portfolio_data
        
        return context
    
    async def _store_decision_in_jena(
        self,
        consensus_state: NashEthereumState,
        decision_request: Dict[str, Any]
    ) -> str:
        """Store consensus decision in Jena knowledge graph"""
        decision_uri = f"http://aiqtoolkit.com/decisions/{consensus_state.task_hash}"
        
        # Create RDF triples for decision
        insert_query = f"""
        PREFIX dh: <http://aiqtoolkit.com/digital-human/>
        PREFIX decision: <http://aiqtoolkit.com/decision/>
        PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
        
        INSERT DATA {{
            <{decision_uri}> a decision:ConsensusDecision ;
                decision:taskHash "{consensus_state.task_hash}" ;
                decision:type "{decision_request['type']}" ;
                decision:timestamp "{time.time()}"^^xsd:dateTime ;
                decision:converged {str(consensus_state.converged).lower()} ;
                decision:blockNumber {consensus_state.block_number} ;
                decision:gasUsed {consensus_state.gas_used} .
        """
        
        # Add individual agent positions
        for agent_id, position in consensus_state.positions.items():
            position_uri = f"{decision_uri}/position/{agent_id}"
            insert_query += f"""
                <{decision_uri}> decision:hasPosition <{position_uri}> .
                <{position_uri}> a decision:AgentPosition ;
                    decision:agentId "{agent_id}" ;
                    decision:position "{position.tolist()}" .
            """
        
        # Add consensus value
        if consensus_state.nash_equilibrium is not None:
            insert_query += f"""
                <{decision_uri}> decision:consensusValue "{consensus_state.nash_equilibrium.tolist()}" .
            """
        
        insert_query += "}"
        
        await self.neural_jena.jena_manager.sparql_update.setQuery(insert_query)
        await self.neural_jena.jena_manager.sparql_update.query()
        
        return decision_uri
    
    def _calculate_confidence(self, consensus_state: NashEthereumState) -> float:
        """Calculate confidence score from consensus state"""
        if not consensus_state.converged:
            return 0.0
        
        # Calculate variance among positions
        positions = list(consensus_state.positions.values())
        if not positions:
            return 0.0
        
        positions_tensor = torch.stack(positions)
        variance = torch.var(positions_tensor, dim=0).mean()
        
        # Lower variance = higher confidence
        confidence = 1.0 / (1.0 + variance.item())
        
        return confidence
    
    async def _update_agents_with_outcome(
        self,
        consensus_state: NashEthereumState,
        decision: MultiAgentDecision
    ):
        """Update agents with decision outcome for learning"""
        # Calculate reward based on decision confidence
        reward = decision.confidence
        
        # Additional reward if consensus converged quickly
        if consensus_state.converged:
            reward += 0.1
        
        # Update each agent
        for agent_id, agent in self.consensus_system.agents.items():
            await agent.learn_from_consensus(consensus_state, reward)
    
    async def explain_decision(
        self,
        decision: MultiAgentDecision
    ) -> str:
        """Generate human-readable explanation of multi-agent decision"""
        explanation_parts = []
        
        # Overall consensus
        explanation_parts.append(
            f"Decision reached with {decision.confidence:.1%} confidence."
        )
        
        # Individual agent perspectives
        explanation_parts.append("\nAgent perspectives:")
        for agent_id, position in decision.individual_positions.items():
            agent_type = agent_id.split('_')[0]
            explanation_parts.append(f"- {agent_type}: {self._interpret_position(position)}")
        
        # Blockchain verification
        explanation_parts.append(
            f"\nDecision verified on blockchain (hash: {decision.blockchain_proof[:8]}...)"
        )
        
        # Query Jena for additional context
        context_query = f"""
        PREFIX decision: <http://aiqtoolkit.com/decision/>
        
        SELECT ?property ?value
        WHERE {{
            <{decision.jena_uri}> ?property ?value .
        }}
        """
        
        context_data = await self.neural_jena.jena_manager.execute_sparql_query(
            context_query
        )
        
        if context_data.get("results", {}).get("bindings"):
            explanation_parts.append("\nAdditional context from knowledge base:")
            for binding in context_data["results"]["bindings"][:3]:
                prop = binding["property"]["value"].split("/")[-1]
                val = binding["value"]["value"]
                explanation_parts.append(f"- {prop}: {val}")
        
        return "\n".join(explanation_parts)
    
    def _interpret_position(self, position: torch.Tensor) -> str:
        """Interpret agent position tensor as human-readable text"""
        # Simplified interpretation - in practice, use more sophisticated mapping
        position_np = position.cpu().numpy()
        
        # Assume first dimension is risk preference
        risk_level = position_np[0] if len(position_np) > 0 else 0.5
        
        if risk_level < 0.3:
            risk_str = "Conservative"
        elif risk_level < 0.7:
            risk_str = "Moderate"
        else:
            risk_str = "Aggressive"
        
        return f"{risk_str} stance"
    
    async def handle_user_query_with_consensus(
        self,
        session_id: str,
        query: str
    ) -> Dict[str, Any]:
        """Handle user query using multi-agent consensus when appropriate"""
        # Determine if query requires multi-agent consensus
        requires_consensus = self._requires_consensus(query)
        
        if requires_consensus and self.enable_multi_agent:
            # Complex decision requiring consensus
            decision_request = {
                "type": "user_query",
                "query": query,
                "parameters": {"original_query": query}
            }
            
            decision = await self.process_complex_decision(
                session_id, decision_request
            )
            
            # Generate response based on consensus
            response_text = await self._generate_consensus_response(
                query, decision
            )
            
            # Add explanation
            explanation = await self.explain_decision(decision)
            
            return {
                "response": response_text,
                "explanation": explanation,
                "consensus_used": True,
                "decision_id": decision.decision_id,
                "confidence": decision.confidence
            }
        else:
            # Simple query - use standard processing
            response = await self.conversation_engine.generate_response(
                query, session_id
            )
            
            return {
                "response": response,
                "consensus_used": False,
                "confidence": 0.8  # Default confidence
            }
    
    def _requires_consensus(self, query: str) -> bool:
        """Determine if query requires multi-agent consensus"""
        # Keywords that indicate complex decisions
        consensus_keywords = [
            "invest", "portfolio", "risk", "allocation",
            "strategy", "recommend", "advise", "best",
            "should i", "what about", "compare"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in consensus_keywords)
    
    async def _generate_consensus_response(
        self,
        query: str,
        decision: MultiAgentDecision
    ) -> str:
        """Generate response based on multi-agent consensus"""
        # Use consensus value to guide response generation
        consensus_embedding = decision.consensus_value
        
        # Generate response using consensus-guided generation
        prompt = f"""
        Based on multi-agent consensus analysis with {decision.confidence:.1%} confidence:
        
        User Query: {query}
        
        Consensus indicates: {self._interpret_position(consensus_embedding)}
        
        Please provide a helpful response that reflects this consensus while addressing the user's specific question.
        """
        
        response = await self.conversation_engine.generate_response(
            prompt, decision.decision_id
        )
        
        return response


# Example usage
async def demonstrate_consensus_orchestration():
    """Demonstrate the enhanced orchestration system"""
    config = {
        "model_name": "meta-llama/Llama-3.1-70B-Instruct",
        "web3_provider": "http://localhost:8545",
        "jena_endpoint": "http://localhost:3030",
        "temperature": 0.7
    }
    
    # Create enhanced orchestrator
    orchestrator = EnhancedDigitalHumanOrchestrator(
        config=config,
        enable_multi_agent=True,
        num_reasoning_agents=5
    )
    
    # Example session
    session_id = "demo_session_001"
    
    # Complex investment query requiring consensus
    query = "Should I reallocate my portfolio given the current market volatility?"
    
    # Process with consensus
    result = await orchestrator.handle_user_query_with_consensus(
        session_id, query
    )
    
    print(f"Response: {result['response']}")
    print(f"Consensus used: {result['consensus_used']}")
    print(f"Confidence: {result['confidence']:.1%}")
    
    if result['consensus_used']:
        print(f"\nExplanation:\n{result['explanation']}")
    
    # Verify blockchain integrity
    if 'decision_id' in result:
        decision = next(
            d for d in orchestrator.decision_history 
            if d.decision_id == result['decision_id']
        )
        
        integrity = orchestrator.consensus_system.verify_consensus_integrity(
            NashEthereumState(
                task_hash=decision.decision_id,
                contract_address=orchestrator.consensus_system.contract.address,
                participants=[],
                positions=decision.individual_positions,
                nash_equilibrium=decision.consensus_value,
                on_chain_consensus=None,
                gas_used=0,
                block_number=0,
                converged=True
            )
        )
        
        print(f"Blockchain integrity verified: {integrity}")


if __name__ == "__main__":
    asyncio.run(demonstrate_consensus_orchestration())
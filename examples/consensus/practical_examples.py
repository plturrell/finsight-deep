# SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Practical examples of Nash-Ethereum consensus with lower stakes
Demonstrates real-world applications beyond financial decisions
"""

import torch
import torch.nn as nn
import asyncio
import time
from typing import Dict, List, Any
import numpy as np

from aiq.neural.gas_optimization import SimpleTaskConsensus
from aiq.neural.consensus_monitoring import ConsensusMonitor


class ContentModerationConsensus(SimpleTaskConsensus):
    """
    Example: Distributed content moderation
    Multiple AI agents decide on content appropriateness
    """
    
    async def moderate_content(
        self,
        content_batch: List[Dict[str, Any]],
        moderation_agents: List['ModerationAgent']
    ) -> Dict[str, Any]:
        """
        Moderate content using consensus of AI agents
        
        Args:
            content_batch: List of content items to moderate
            moderation_agents: List of moderation agents
            
        Returns:
            Consensus moderation decisions
        """
        task = {
            "type": "content_moderation",
            "batch_id": f"batch_{int(time.time())}",
            "content_count": len(content_batch),
            "timestamp": time.time()
        }
        
        decisions = {}
        
        for content in content_batch:
            item_decisions = []
            
            # Get each agent's assessment
            for agent in moderation_agents:
                score, confidence = await agent.assess_content(content)
                
                # Submit to consensus (low priority)
                await self.submit_position_optimized(
                    agent_id=agent.agent_id,
                    position=score,  # 0-1 score (0=inappropriate, 1=appropriate)
                    confidence=confidence,
                    task_hash=f"{task['batch_id']}_{content['id']}",
                    priority="low"
                )
                
                item_decisions.append({
                    "agent": agent.agent_id,
                    "score": score.item(),
                    "confidence": confidence
                })
            
            decisions[content['id']] = item_decisions
        
        # Process batch for gas efficiency
        await self._wait_for_batch_processing()
        
        # Get consensus decisions
        consensus_results = {}
        for content in content_batch:
            consensus_state = await self.get_consensus_state(
                f"{task['batch_id']}_{content['id']}"
            )
            
            if consensus_state:
                consensus_score = consensus_state.consensus_value[0].item()
                consensus_results[content['id']] = {
                    "action": "approve" if consensus_score > 0.7 else "flag",
                    "consensus_score": consensus_score,
                    "individual_assessments": decisions[content['id']]
                }
        
        return {
            "batch_id": task['batch_id'],
            "results": consensus_results,
            "gas_efficiency": self._estimate_gas_savings(),
            "processing_time": time.time() - task['timestamp']
        }


class CollaborativeRecommendationConsensus(SimpleTaskConsensus):
    """
    Example: Collaborative recommendation system
    Multiple AI agents suggest personalized recommendations
    """
    
    async def generate_recommendations(
        self,
        user_profile: Dict[str, Any],
        item_pool: List[Dict[str, Any]],
        recommendation_agents: List['RecommendationAgent']
    ) -> Dict[str, Any]:
        """
        Generate personalized recommendations using agent consensus
        
        Args:
            user_profile: User preferences and history
            item_pool: Available items to recommend
            recommendation_agents: List of recommendation agents
            
        Returns:
            Consensus-based recommendations
        """
        task = {
            "type": "recommendation",
            "user_id": user_profile['user_id'],
            "item_count": len(item_pool),
            "timestamp": time.time()
        }
        
        # Get rankings from each agent
        agent_rankings = []
        
        for agent in recommendation_agents:
            rankings, confidence = await agent.rank_items(user_profile, item_pool)
            
            # Submit ranking as position vector
            await self.submit_position_optimized(
                agent_id=agent.agent_id,
                position=rankings,
                confidence=confidence,
                task_hash=f"rec_{task['user_id']}_{int(task['timestamp'])}",
                priority="medium"
            )
            
            agent_rankings.append({
                "agent": agent.agent_id,
                "rankings": rankings,
                "confidence": confidence
            })
        
        # Wait for consensus
        await self._wait_for_batch_processing()
        
        # Get consensus rankings
        consensus_state = await self.get_consensus_state(
            f"rec_{task['user_id']}_{int(task['timestamp'])}"
        )
        
        if consensus_state and consensus_state.consensus_value is not None:
            # Sort items by consensus ranking
            consensus_rankings = consensus_state.consensus_value
            sorted_indices = torch.argsort(consensus_rankings, descending=True)
            
            recommendations = []
            for idx in sorted_indices[:10]:  # Top 10 recommendations
                item = item_pool[idx]
                recommendations.append({
                    "item_id": item['id'],
                    "title": item['title'],
                    "consensus_score": consensus_rankings[idx].item(),
                    "agent_agreement": self._calculate_agreement(
                        idx, agent_rankings
                    )
                })
            
            return {
                "user_id": user_profile['user_id'],
                "recommendations": recommendations,
                "consensus_confidence": self._calculate_consensus_confidence(
                    consensus_state
                ),
                "processing_metrics": {
                    "agents_used": len(recommendation_agents),
                    "gas_saved": self._estimate_gas_savings()['percentage'],
                    "latency_ms": (time.time() - task['timestamp']) * 1000
                }
            }
        
        return {"error": "Failed to reach consensus"}
    
    def _calculate_agreement(self, item_idx: int, agent_rankings: List[Dict]) -> float:
        """Calculate agent agreement on specific item"""
        ranks = []
        for ranking in agent_rankings:
            item_rank = torch.where(
                torch.argsort(ranking['rankings'], descending=True) == item_idx
            )[0]
            if len(item_rank) > 0:
                ranks.append(item_rank[0].item())
        
        if not ranks:
            return 0.0
        
        # Lower variance = higher agreement
        variance = np.var(ranks)
        return 1.0 / (1.0 + variance)


class DataValidationConsensus(SimpleTaskConsensus):
    """
    Example: Distributed data validation
    Multiple agents validate data quality and accuracy
    """
    
    async def validate_dataset(
        self,
        data_batch: List[Dict[str, Any]],
        validation_agents: List['ValidationAgent']
    ) -> Dict[str, Any]:
        """
        Validate data quality using agent consensus
        
        Args:
            data_batch: Batch of data to validate
            validation_agents: List of validation agents
            
        Returns:
            Consensus validation results
        """
        task = {
            "type": "data_validation",
            "batch_id": f"validation_{int(time.time())}",
            "size": len(data_batch),
            "timestamp": time.time()
        }
        
        validation_results = []
        
        # Process each data item
        for i, data_item in enumerate(data_batch):
            item_validations = []
            
            # Get validation from each agent
            for agent in validation_agents:
                validation_score, issues = await agent.validate_data(data_item)
                
                # Create position vector encoding validation aspects
                position = torch.tensor([
                    validation_score,  # Overall score
                    float(len(issues)),  # Number of issues
                    float('accuracy' in issues),  # Accuracy flag
                    float('completeness' in issues),  # Completeness flag
                    float('consistency' in issues)  # Consistency flag
                ])
                
                await self.submit_position_optimized(
                    agent_id=agent.agent_id,
                    position=position,
                    confidence=agent.confidence,
                    task_hash=f"{task['batch_id']}_item_{i}",
                    priority="low"
                )
                
                item_validations.append({
                    "agent": agent.agent_id,
                    "score": validation_score,
                    "issues": issues
                })
            
            validation_results.append({
                "item_id": data_item.get('id', i),
                "validations": item_validations
            })
        
        # Process batch
        await self._wait_for_batch_processing()
        
        # Compile consensus results
        consensus_results = []
        
        for i, item_result in enumerate(validation_results):
            consensus_state = await self.get_consensus_state(
                f"{task['batch_id']}_item_{i}"
            )
            
            if consensus_state and consensus_state.consensus_value is not None:
                consensus_vector = consensus_state.consensus_value
                
                consensus_results.append({
                    "item_id": item_result['item_id'],
                    "consensus_score": consensus_vector[0].item(),
                    "issue_count": int(consensus_vector[1].item()),
                    "has_accuracy_issues": consensus_vector[2].item() > 0.5,
                    "has_completeness_issues": consensus_vector[3].item() > 0.5,
                    "has_consistency_issues": consensus_vector[4].item() > 0.5,
                    "individual_validations": item_result['validations']
                })
        
        # Summary statistics
        avg_score = np.mean([r['consensus_score'] for r in consensus_results])
        issues_found = sum(r['issue_count'] for r in consensus_results)
        
        return {
            "batch_id": task['batch_id'],
            "results": consensus_results,
            "summary": {
                "average_quality_score": avg_score,
                "total_issues": issues_found,
                "pass_rate": sum(1 for r in consensus_results if r['consensus_score'] > 0.8) / len(consensus_results)
            },
            "efficiency": {
                "processing_time_ms": (time.time() - task['timestamp']) * 1000,
                "gas_saved": self._estimate_gas_savings()['percentage']
            }
        }


# Example Agents for demonstrations

class ModerationAgent:
    """Simple content moderation agent"""
    
    def __init__(self, agent_id: str, model: nn.Module):
        self.agent_id = agent_id
        self.model = model
        self.confidence = 0.8
    
    async def assess_content(self, content: Dict[str, Any]) -> Tuple[torch.Tensor, float]:
        """Assess content appropriateness"""
        # Simplified assessment
        text = content.get('text', '')
        
        # Mock assessment based on content features
        features = torch.tensor([
            len(text),
            text.count('!'),
            text.count('?'),
            float(any(word in text.lower() for word in ['spam', 'scam']))
        ])
        
        with torch.no_grad():
            score = torch.sigmoid(self.model(features))
        
        return score, self.confidence


class RecommendationAgent:
    """Simple recommendation agent"""
    
    def __init__(self, agent_id: str, strategy: str = "collaborative"):
        self.agent_id = agent_id
        self.strategy = strategy
        self.confidence = 0.85
    
    async def rank_items(
        self,
        user_profile: Dict[str, Any],
        items: List[Dict[str, Any]]
    ) -> Tuple[torch.Tensor, float]:
        """Rank items for user"""
        rankings = []
        
        for item in items:
            # Mock ranking based on strategy
            if self.strategy == "collaborative":
                score = np.random.random() * user_profile.get('collaborative_weight', 1.0)
            elif self.strategy == "content":
                score = np.random.random() * user_profile.get('content_weight', 1.0)
            else:
                score = np.random.random()
            
            rankings.append(score)
        
        return torch.tensor(rankings), self.confidence


class ValidationAgent:
    """Simple data validation agent"""
    
    def __init__(self, agent_id: str, strictness: float = 0.7):
        self.agent_id = agent_id
        self.strictness = strictness
        self.confidence = 0.9
    
    async def validate_data(self, data_item: Dict[str, Any]) -> Tuple[float, List[str]]:
        """Validate data item"""
        issues = []
        score = 1.0
        
        # Mock validation checks
        if 'required_field' not in data_item:
            issues.append('completeness')
            score -= 0.3
        
        if not isinstance(data_item.get('value', 0), (int, float)):
            issues.append('type_error')
            score -= 0.2
        
        if np.random.random() < self.strictness:
            if np.random.random() < 0.2:
                issues.append('accuracy')
                score -= 0.2
        
        return max(0, score), issues


# Main demonstration
async def run_practical_examples():
    """Run practical consensus examples"""
    
    print("=== Nash-Ethereum Consensus Practical Examples ===\n")
    
    # Initialize monitoring
    monitor = ConsensusMonitor()
    
    # Example 1: Content Moderation
    print("1. Content Moderation Example")
    moderation_consensus = ContentModerationConsensus()
    
    # Create moderation agents
    moderation_agents = []
    for i in range(5):
        model = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        agent = ModerationAgent(f"mod_agent_{i}", model)
        moderation_agents.append(agent)
    
    # Test content
    content_batch = [
        {"id": "content_1", "text": "This is a normal post about cats."},
        {"id": "content_2", "text": "URGENT!!! CLICK HERE FOR FREE MONEY!!!"},
        {"id": "content_3", "text": "Check out this amazing recipe for cookies."}
    ]
    
    moderation_result = await moderation_consensus.moderate_content(
        content_batch, moderation_agents
    )
    
    print(f"Moderation Results:")
    for content_id, result in moderation_result['results'].items():
        print(f"  {content_id}: {result['action']} (score: {result['consensus_score']:.3f})")
    print(f"Gas saved: {moderation_result['gas_efficiency']['percentage']:.1f}%\n")
    
    # Example 2: Collaborative Recommendations
    print("2. Recommendation System Example")
    recommendation_consensus = CollaborativeRecommendationConsensus()
    
    # Create recommendation agents
    rec_agents = [
        RecommendationAgent(f"rec_agent_{i}", strategy)
        for i, strategy in enumerate(['collaborative', 'content', 'hybrid'])
    ]
    
    # User profile
    user_profile = {
        "user_id": "user_123",
        "collaborative_weight": 0.7,
        "content_weight": 0.3
    }
    
    # Items to recommend
    items = [
        {"id": f"item_{i}", "title": f"Product {i}"}
        for i in range(20)
    ]
    
    rec_result = await recommendation_consensus.generate_recommendations(
        user_profile, items, rec_agents
    )
    
    print("Top Recommendations:")
    for i, rec in enumerate(rec_result['recommendations'][:5]):
        print(f"  {i+1}. {rec['title']} (score: {rec['consensus_score']:.3f})")
    print(f"Gas saved: {rec_result['processing_metrics']['gas_saved']:.1f}%\n")
    
    # Example 3: Data Validation
    print("3. Data Validation Example")
    validation_consensus = DataValidationConsensus()
    
    # Create validation agents
    val_agents = [
        ValidationAgent(f"val_agent_{i}", strictness=0.6 + i*0.1)
        for i in range(4)
    ]
    
    # Data to validate
    data_batch = [
        {"id": "data_1", "required_field": "value", "value": 42},
        {"id": "data_2", "value": "string instead of number"},
        {"id": "data_3", "required_field": "exists", "value": 3.14}
    ]
    
    val_result = await validation_consensus.validate_dataset(
        data_batch, val_agents
    )
    
    print("Validation Results:")
    for result in val_result['results']:
        print(f"  {result['item_id']}: score={result['consensus_score']:.3f}, issues={result['issue_count']}")
    print(f"Overall pass rate: {val_result['summary']['pass_rate']:.1%}")
    print(f"Gas saved: {val_result['efficiency']['gas_saved']:.1f}%\n")
    
    # Display monitoring summary
    print("=== Monitoring Summary ===")
    summary = monitor.get_summary_stats()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    asyncio.run(run_practical_examples())
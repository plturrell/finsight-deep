# Multi-Agent Configuration Guide

## Overview

This guide covers how to configure and optimize multi-agent systems in the Nash-Ethereum consensus framework. Proper agent configuration is crucial for achieving efficient consensus and high-quality decisions.

## Agent Types

### 1. Specialized Agents

Different agent types optimized for specific domains:

```python
from aiq.neural import EthereumNeuralAgent, AgentType

# Risk Analysis Agent
class RiskAnalysisAgent(EthereumNeuralAgent):
    def __init__(self, agent_id, identity, device="cuda"):
        model = self._build_risk_model()
        super().__init__(agent_id, model, identity, device)
        self.agent_type = AgentType.RISK_ANALYST
    
    def _build_risk_model(self):
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Sigmoid()  # Risk scores between 0-1
        )

# Growth Optimization Agent
class GrowthOptimizerAgent(EthereumNeuralAgent):
    def __init__(self, agent_id, identity, device="cuda"):
        model = self._build_growth_model()
        super().__init__(agent_id, model, identity, device)
        self.agent_type = AgentType.GROWTH_OPTIMIZER
    
    def _build_growth_model(self):
        return nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.Softplus()  # Positive growth predictions
        )
```

### 2. Hybrid Agents

Agents that combine multiple capabilities:

```python
class HybridFinancialAgent(EthereumNeuralAgent):
    def __init__(self, agent_id, identity, capabilities=None):
        self.capabilities = capabilities or [
            "risk_assessment",
            "growth_analysis",
            "market_prediction"
        ]
        
        model = self._build_hybrid_model()
        super().__init__(agent_id, model, identity)
    
    def _build_hybrid_model(self):
        # Multi-head architecture
        base = nn.Sequential(
            nn.Linear(768, 1024),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        heads = nn.ModuleDict({
            capability: nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 256)
            )
            for capability in self.capabilities
        })
        
        return nn.ModuleDict({
            'base': base,
            'heads': heads
        })
    
    async def generate_position(self, task):
        # Use appropriate head based on task
        capability = self._select_capability(task)
        features = self.model['base'](task.embedding)
        position = self.model['heads'][capability](features)
        return position
```

## Configuration Strategies

### 1. Agent Pool Configuration

```python
class AgentPoolConfig:
    def __init__(self):
        self.pool_size = 10
        self.diversity_factor = 0.7
        self.specialization_ratio = {
            "risk_analyst": 0.2,
            "growth_optimizer": 0.2,
            "market_analyst": 0.2,
            "portfolio_manager": 0.2,
            "generalist": 0.2
        }
    
    def create_agent_pool(self, consensus_system):
        agents = []
        
        for agent_type, ratio in self.specialization_ratio.items():
            count = int(self.pool_size * ratio)
            
            for i in range(count):
                agent_id = f"{agent_type}_{i}"
                identity = consensus_system.create_agent_identity(agent_id)
                
                if agent_type == "risk_analyst":
                    agent = RiskAnalysisAgent(agent_id, identity)
                elif agent_type == "growth_optimizer":
                    agent = GrowthOptimizerAgent(agent_id, identity)
                else:
                    agent = self._create_generic_agent(
                        agent_id, identity, agent_type
                    )
                
                agents.append(agent)
        
        return agents
```

### 2. Dynamic Agent Allocation

```python
class DynamicAgentAllocator:
    def __init__(self, agent_pool):
        self.agent_pool = agent_pool
        self.task_history = {}
        self.performance_metrics = {}
    
    def allocate_agents_for_task(self, task):
        """
        Dynamically allocate agents based on task requirements
        """
        # Analyze task requirements
        required_capabilities = self._analyze_task_requirements(task)
        
        # Score agents based on capabilities and past performance
        agent_scores = {}
        for agent in self.agent_pool:
            score = self._score_agent_for_task(agent, task, required_capabilities)
            agent_scores[agent.agent_id] = score
        
        # Select top agents
        sorted_agents = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        num_agents = self._determine_optimal_agent_count(task)
        selected_agents = [
            self._get_agent_by_id(agent_id)
            for agent_id, _ in sorted_agents[:num_agents]
        ]
        
        return selected_agents
    
    def _score_agent_for_task(self, agent, task, required_capabilities):
        score = 0.0
        
        # Capability match
        for capability in required_capabilities:
            if hasattr(agent, capability):
                score += 1.0
        
        # Historical performance
        if agent.agent_id in self.performance_metrics:
            performance = self.performance_metrics[agent.agent_id]
            score += performance.get('success_rate', 0) * 0.5
            score += performance.get('convergence_speed', 0) * 0.3
        
        # Specialization bonus
        if agent.agent_type == task.preferred_agent_type:
            score += 0.5
        
        return score
```

### 3. Learning Configuration

```python
class AgentLearningConfig:
    def __init__(self):
        self.learning_params = {
            "learning_rate": 0.001,
            "experience_replay_size": 10000,
            "batch_size": 32,
            "update_frequency": 10,
            "exploration_rate": 0.1,
            "exploration_decay": 0.995,
            "min_exploration": 0.01
        }
        
        self.reward_structure = {
            "consensus_contribution": 0.4,
            "accuracy": 0.3,
            "speed": 0.2,
            "gas_efficiency": 0.1
        }
    
    def configure_agent_learning(self, agent):
        # Set up reinforcement learning
        agent.rl_config = RLConfig(
            algorithm="PPO",
            learning_rate=self.learning_params["learning_rate"],
            gamma=0.99,
            clip_param=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01
        )
        
        # Configure experience replay
        agent.experience_replay = ExperienceReplay(
            capacity=self.learning_params["experience_replay_size"],
            batch_size=self.learning_params["batch_size"]
        )
        
        # Set exploration parameters
        agent.exploration_rate = self.learning_params["exploration_rate"]
        agent.exploration_decay = self.learning_params["exploration_decay"]
        
        return agent
```

## Communication Protocols

### 1. Agent Communication Layer

```python
class AgentCommunicationProtocol:
    def __init__(self, encryption_enabled=True):
        self.encryption_enabled = encryption_enabled
        self.message_types = {
            "PROPOSAL": "proposal",
            "VOTE": "vote",
            "OBJECTION": "objection",
            "INFORMATION": "information",
            "NEGOTIATION": "negotiation"
        }
    
    async def broadcast_message(self, sender_agent, message_type, content):
        """
        Broadcast message to all agents
        """
        message = {
            "sender": sender_agent.agent_id,
            "type": message_type,
            "content": content,
            "timestamp": time.time()
        }
        
        if self.encryption_enabled:
            message = self._encrypt_message(message, sender_agent.private_key)
        
        # Send to all agents
        for agent in self.get_all_agents():
            if agent.agent_id != sender_agent.agent_id:
                await agent.receive_message(message)
    
    async def private_message(self, sender, recipient, content):
        """
        Send private message between agents
        """
        message = {
            "sender": sender.agent_id,
            "recipient": recipient.agent_id,
            "content": content,
            "timestamp": time.time(),
            "private": True
        }
        
        if self.encryption_enabled:
            # Encrypt with recipient's public key
            message = self._encrypt_for_recipient(
                message,
                recipient.public_key
            )
        
        await recipient.receive_message(message)
```

### 2. Negotiation Protocols

```python
class NegotiationProtocol:
    def __init__(self):
        self.negotiation_rounds = 5
        self.concession_rate = 0.1
    
    async def negotiate_consensus(self, agents, initial_positions):
        """
        Multi-round negotiation between agents
        """
        current_positions = initial_positions.copy()
        
        for round_num in range(self.negotiation_rounds):
            # Each agent proposes adjustments
            proposals = {}
            for agent in agents:
                proposal = await agent.propose_adjustment(
                    current_positions,
                    round_num
                )
                proposals[agent.agent_id] = proposal
            
            # Agents vote on proposals
            votes = await self._collect_votes(agents, proposals)
            
            # Update positions based on votes
            current_positions = self._update_positions(
                current_positions,
                proposals,
                votes
            )
            
            # Check if consensus reached
            if self._check_consensus(current_positions):
                break
        
        return current_positions
```

## Performance Optimization

### 1. GPU Distribution

```python
class MultiGPUAgentManager:
    def __init__(self, num_gpus=None):
        self.num_gpus = num_gpus or torch.cuda.device_count()
        self.device_assignments = {}
        self.load_balancer = GPULoadBalancer(self.num_gpus)
    
    def assign_agent_to_gpu(self, agent):
        """
        Assign agent to optimal GPU based on load
        """
        # Get current GPU loads
        gpu_loads = self.load_balancer.get_current_loads()
        
        # Find least loaded GPU
        optimal_gpu = min(gpu_loads, key=gpu_loads.get)
        
        # Assign agent
        agent.to(f"cuda:{optimal_gpu}")
        self.device_assignments[agent.agent_id] = optimal_gpu
        
        # Update load balancer
        self.load_balancer.update_assignment(agent, optimal_gpu)
    
    def rebalance_agents(self):
        """
        Rebalance agents across GPUs for optimal performance
        """
        all_agents = list(self.device_assignments.keys())
        
        # Sort agents by computational requirements
        agent_loads = {}
        for agent_id in all_agents:
            agent = self.get_agent(agent_id)
            agent_loads[agent_id] = self._estimate_agent_load(agent)
        
        # Redistribute using bin packing algorithm
        new_assignments = self._bin_packing_assignment(
            agent_loads,
            self.num_gpus
        )
        
        # Apply new assignments
        for agent_id, gpu_id in new_assignments.items():
            if self.device_assignments[agent_id] != gpu_id:
                agent = self.get_agent(agent_id)
                agent.to(f"cuda:{gpu_id}")
                self.device_assignments[agent_id] = gpu_id
```

### 2. Memory Optimization

```python
class AgentMemoryOptimizer:
    def __init__(self, max_memory_per_agent_mb=512):
        self.max_memory_per_agent = max_memory_per_agent_mb * 1024 * 1024
        self.memory_tracking = {}
    
    def optimize_agent_memory(self, agent):
        """
        Optimize agent's memory usage
        """
        # 1. Enable gradient checkpointing
        if hasattr(agent.model, 'enable_gradient_checkpointing'):
            agent.model.enable_gradient_checkpointing()
        
        # 2. Use mixed precision
        agent.use_mixed_precision = True
        agent.scaler = torch.cuda.amp.GradScaler()
        
        # 3. Limit replay buffer
        if hasattr(agent, 'experience_replay'):
            current_size = agent.experience_replay.capacity
            optimal_size = self._calculate_optimal_replay_size(agent)
            if optimal_size < current_size:
                agent.experience_replay.resize(optimal_size)
        
        # 4. Clear unnecessary caches
        torch.cuda.empty_cache()
        
        # 5. Monitor memory usage
        self.memory_tracking[agent.agent_id] = self._get_agent_memory_usage(agent)
```

## Agent Coordination

### 1. Role-Based Coordination

```python
class RoleBasedCoordinator:
    def __init__(self):
        self.role_hierarchy = {
            "lead_analyst": 1,
            "specialist": 2,
            "validator": 3,
            "observer": 4
        }
        
        self.role_responsibilities = {
            "lead_analyst": ["propose_initial", "aggregate_results"],
            "specialist": ["analyze_domain", "provide_expertise"],
            "validator": ["verify_proposals", "check_consistency"],
            "observer": ["monitor_process", "report_anomalies"]
        }
    
    def assign_roles(self, agents, task):
        """
        Assign roles based on agent capabilities and task requirements
        """
        role_assignments = {}
        
        # Assign lead analyst
        lead_agent = self._select_lead_agent(agents, task)
        role_assignments[lead_agent.agent_id] = "lead_analyst"
        
        # Assign specialists
        specialists = self._select_specialists(agents, task)
        for agent in specialists:
            role_assignments[agent.agent_id] = "specialist"
        
        # Assign validators
        validators = self._select_validators(agents)
        for agent in validators:
            role_assignments[agent.agent_id] = "validator"
        
        # Remaining agents are observers
        for agent in agents:
            if agent.agent_id not in role_assignments:
                role_assignments[agent.agent_id] = "observer"
        
        return role_assignments
```

### 2. Consensus Strategies

```python
class ConsensusStrategyManager:
    def __init__(self):
        self.strategies = {
            "unanimous": self._unanimous_consensus,
            "majority": self._majority_consensus,
            "weighted": self._weighted_consensus,
            "hierarchical": self._hierarchical_consensus
        }
    
    async def apply_strategy(self, strategy_name, agents, positions):
        """
        Apply specified consensus strategy
        """
        strategy_func = self.strategies.get(strategy_name)
        if not strategy_func:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        return await strategy_func(agents, positions)
    
    async def _unanimous_consensus(self, agents, positions):
        """
        Require unanimous agreement
        """
        if len(set(tuple(p.tolist()) for p in positions.values())) == 1:
            return positions[agents[0].agent_id], True
        return None, False
    
    async def _weighted_consensus(self, agents, positions):
        """
        Weight positions by agent reputation
        """
        weighted_sum = torch.zeros_like(next(iter(positions.values())))
        total_weight = 0
        
        for agent_id, position in positions.items():
            agent = self._get_agent_by_id(agent_id)
            weight = agent.reputation_score
            weighted_sum += position * weight
            total_weight += weight
        
        consensus = weighted_sum / total_weight
        return consensus, True
```

## Monitoring and Debugging

### 1. Agent Performance Metrics

```python
class AgentPerformanceMonitor:
    def __init__(self):
        self.metrics = defaultdict(lambda: {
            'consensus_participation': [],
            'convergence_contribution': [],
            'accuracy_scores': [],
            'response_times': [],
            'gas_efficiency': []
        })
    
    def record_agent_performance(self, agent_id, consensus_round):
        """
        Record comprehensive performance metrics
        """
        metrics = self.metrics[agent_id]
        
        # Participation rate
        metrics['consensus_participation'].append(
            consensus_round.agent_participated(agent_id)
        )
        
        # Convergence contribution
        contribution = self._calculate_convergence_contribution(
            agent_id,
            consensus_round
        )
        metrics['convergence_contribution'].append(contribution)
        
        # Accuracy score
        if consensus_round.has_ground_truth:
            accuracy = self._calculate_accuracy(
                agent_id,
                consensus_round
            )
            metrics['accuracy_scores'].append(accuracy)
        
        # Response time
        metrics['response_times'].append(
            consensus_round.get_response_time(agent_id)
        )
        
        # Gas efficiency
        metrics['gas_efficiency'].append(
            consensus_round.get_gas_usage(agent_id)
        )
    
    def generate_agent_report(self, agent_id):
        """
        Generate comprehensive performance report
        """
        metrics = self.metrics[agent_id]
        
        report = {
            'agent_id': agent_id,
            'participation_rate': np.mean(metrics['consensus_participation']),
            'avg_convergence_contribution': np.mean(metrics['convergence_contribution']),
            'accuracy': np.mean(metrics['accuracy_scores']) if metrics['accuracy_scores'] else None,
            'avg_response_time': np.mean(metrics['response_times']),
            'gas_efficiency': np.mean(metrics['gas_efficiency']),
            'performance_trend': self._calculate_trend(metrics)
        }
        
        return report
```

### 2. Debugging Tools

```python
class AgentDebugger:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.debug_logs = []
    
    def debug_agent_decision(self, agent, task, position):
        """
        Debug individual agent decision-making
        """
        debug_info = {
            'agent_id': agent.agent_id,
            'task': task,
            'timestamp': time.time(),
            'model_state': self._capture_model_state(agent.model),
            'input_features': self._capture_input_features(task),
            'intermediate_activations': self._capture_activations(agent, task),
            'final_position': position,
            'confidence': agent.calculate_confidence(position)
        }
        
        if self.verbose:
            self._print_debug_info(debug_info)
        
        self.debug_logs.append(debug_info)
        return debug_info
    
    def analyze_consensus_failure(self, consensus_round):
        """
        Analyze why consensus failed
        """
        analysis = {
            'round_id': consensus_round.id,
            'iteration_count': consensus_round.iterations,
            'agent_positions': consensus_round.final_positions,
            'position_variance': self._calculate_position_variance(
                consensus_round.final_positions
            ),
            'communication_graph': self._analyze_communication(
                consensus_round.communication_log
            ),
            'bottlenecks': self._identify_bottlenecks(consensus_round)
        }
        
        return analysis
```

## Advanced Configurations

### 1. Adaptive Agent Configuration

```python
class AdaptiveAgentConfigurator:
    def __init__(self):
        self.configuration_history = []
        self.performance_tracker = AgentPerformanceMonitor()
    
    def adapt_configuration(self, agent, recent_performance):
        """
        Dynamically adapt agent configuration based on performance
        """
        current_config = agent.get_configuration()
        
        # Analyze performance trends
        trends = self._analyze_performance_trends(
            agent.agent_id,
            recent_performance
        )
        
        # Recommend adjustments
        adjustments = {}
        
        if trends['accuracy_declining']:
            adjustments['learning_rate'] = current_config['learning_rate'] * 1.1
            adjustments['exploration_rate'] = min(
                current_config['exploration_rate'] * 1.2,
                0.3
            )
        
        if trends['slow_convergence']:
            adjustments['confidence_threshold'] = (
                current_config['confidence_threshold'] * 0.95
            )
        
        if trends['high_gas_usage']:
            adjustments['batch_size'] = min(
                current_config['batch_size'] * 2,
                64
            )
        
        # Apply adjustments
        if adjustments:
            agent.update_configuration(adjustments)
            self.configuration_history.append({
                'agent_id': agent.agent_id,
                'timestamp': time.time(),
                'adjustments': adjustments,
                'reason': trends
            })
```

### 2. Ensemble Agent Configuration

```python
class EnsembleAgentConfig:
    def __init__(self, base_agents):
        self.base_agents = base_agents
        self.ensemble_strategies = {
            "voting": self._voting_ensemble,
            "stacking": self._stacking_ensemble,
            "boosting": self._boosting_ensemble
        }
    
    def create_ensemble_agent(self, strategy="voting"):
        """
        Create ensemble agent from base agents
        """
        ensemble_agent = EnsembleAgent(
            agent_id=f"ensemble_{strategy}",
            base_agents=self.base_agents,
            strategy=self.ensemble_strategies[strategy]
        )
        
        # Configure ensemble-specific parameters
        ensemble_agent.aggregation_weights = self._optimize_weights(
            self.base_agents
        )
        ensemble_agent.confidence_combination = "multiplicative"
        
        return ensemble_agent
    
    def _voting_ensemble(self, positions):
        """
        Simple voting ensemble
        """
        return torch.median(torch.stack(positions), dim=0).values
    
    def _stacking_ensemble(self, positions, meta_model):
        """
        Stacking ensemble with meta-learner
        """
        stacked_input = torch.cat(positions, dim=-1)
        return meta_model(stacked_input)
```
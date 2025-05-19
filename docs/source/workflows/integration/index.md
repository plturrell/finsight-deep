# Integration Guide

## Overview

This guide explains how the various AIQToolkit components integrate with each other to create powerful AI systems. The modular architecture allows flexible combination of verification, consensus, research, and digital human components.

## Component Integration Map

```
┌────────────────────────────────────────────────────────────────┐
│                    AIQToolkit Integration                       │
├────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│  │ Verification │←→│  Consensus   │←→│   Research   │        │
│  │   System     │  │   System     │  │  Components  │        │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘        │
│         │                 │                 │                  │
│         └─────────────────┼─────────────────┘                  │
│                           │                                    │
│                    ┌──────┴───────┐                           │
│                    │ Digital Human│                           │
│                    │  Orchestrator│                           │
│                    └──────────────┘                           │
└────────────────────────────────────────────────────────────────┘
```

## Core Integration Patterns

### 1. Verification + Consensus

Combine real-time verification with multi-agent consensus:

```python
from aiq.verification import VerificationSystem
from aiq.neural import NashEthereumConsensus

class VerifiedConsensus:
    def __init__(self):
        self.verifier = VerificationSystem({
            'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer']
        })
        self.consensus = NashEthereumConsensus()
    
    async def verified_decision(self, claim, sources, agents):
        # Step 1: Verify claim
        verification = await self.verifier.verify_claim(claim, sources)
        
        # Step 2: Only proceed to consensus if confidence is high
        if verification.confidence > 0.7:
            task = {
                'type': 'verified_decision',
                'claim': claim,
                'verification': verification
            }
            
            consensus_result = await self.consensus.orchestrate_consensus(
                task, agents, hybrid_mode=True
            )
            
            return {
                'decision': consensus_result.nash_equilibrium,
                'verification_confidence': verification.confidence,
                'consensus_achieved': consensus_result.converged
            }
        else:
            return {
                'decision': None,
                'verification_confidence': verification.confidence,
                'consensus_achieved': False,
                'reason': 'Insufficient verification confidence'
            }
```

### 2. Research + Self-Correction + Verification

Complete research pipeline with error correction and verification:

```python
from aiq.research import ResearchTaskExecutor, ResearchTask, TaskType
from aiq.correction import SelfCorrectingResearchSystem
from aiq.verification import VerificationSystem

class IntelligentResearchPipeline:
    def __init__(self):
        self.executor = ResearchTaskExecutor()
        self.corrector = SelfCorrectingResearchSystem()
        self.verifier = VerificationSystem()
    
    async def execute_verified_research(self, query):
        # Step 1: Execute research task
        task = ResearchTask(
            task_id="research_001",
            task_type=TaskType.SYNTHESIS,
            query=query
        )
        
        research_result = await self.executor.execute_task(task)
        
        # Step 2: Apply self-correction
        correction_result = await self.corrector.process_query(
            query=query,
            initial_content=research_result.result_data['content'],
            content_type=ContentType.FACTUAL_REPORT
        )
        
        # Step 3: Verify corrected content
        verification_result = await self.verifier.verify_claim(
            claim=correction_result.corrected_content,
            sources=research_result.result_data.get('sources', [])
        )
        
        return {
            'content': correction_result.corrected_content,
            'errors_corrected': correction_result.error_count,
            'confidence': verification_result.confidence,
            'sources': verification_result.sources_verified,
            'total_time': sum([
                research_result.execution_time_ms,
                correction_result.processing_time_ms,
                verification_result.verification_time_ms
            ])
        }
```

### 3. Digital Human + All Components

Complete digital human system with integrated intelligence:

```python
from aiq.digital_human.orchestrator import DigitalHumanOrchestrator
from aiq.digital_human.conversation import SgLangConversationEngine
from aiq.verification import VerificationSystem
from aiq.neural import NashEthereumConsensus
from aiq.research import ResearchTaskExecutor

class IntelligentDigitalHuman(DigitalHumanOrchestrator):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize all components
        self.verifier = VerificationSystem()
        self.consensus = NashEthereumConsensus()
        self.researcher = ResearchTaskExecutor()
        
        # Enhanced conversation engine
        self.conversation_engine = SgLangConversationEngine(
            enable_research=True,
            enable_verification=True
        )
    
    async def process_user_input_with_intelligence(self, user_input):
        # Step 1: Analyze user intent
        intent = await self._classify_intent(user_input)
        
        if intent == "fact_check":
            # Use verification system
            result = await self._handle_fact_check(user_input)
        elif intent == "complex_decision":
            # Use consensus system
            result = await self._handle_consensus_decision(user_input)
        elif intent == "research":
            # Use research components
            result = await self._handle_research_query(user_input)
        else:
            # Standard conversation
            result = await self._handle_general_query(user_input)
        
        # Generate response with avatar
        avatar_response = await self._generate_avatar_response(
            result['response'],
            result.get('emotion', 'neutral')
        )
        
        return avatar_response
```

## Integration Architectures

### Microservices Architecture

```python
# API Gateway for integrated services
from fastapi import FastAPI
from aiq.verification import VerificationSystem
from aiq.neural import NashEthereumConsensus
from aiq.research import ResearchTaskExecutor

app = FastAPI()

# Initialize services
verifier = VerificationSystem()
consensus = NashEthereumConsensus()
researcher = ResearchTaskExecutor()

@app.post("/intelligent/query")
async def intelligent_query(request: QueryRequest):
    # Route to appropriate service
    if request.requires_verification:
        verification = await verifier.verify_claim(
            request.claim,
            request.sources
        )
        
    if request.requires_consensus:
        consensus_result = await consensus.orchestrate_consensus(
            request.task,
            request.agents
        )
    
    if request.requires_research:
        research_result = await researcher.execute_task(
            request.research_task
        )
    
    # Combine results
    return {
        "verification": verification if request.requires_verification else None,
        "consensus": consensus_result if request.requires_consensus else None,
        "research": research_result if request.requires_research else None
    }
```

### Event-Driven Architecture

```python
import asyncio
from typing import Dict, Any

class EventDrivenIntegration:
    def __init__(self):
        self.event_bus = EventBus()
        self.components = {
            'verifier': VerificationSystem(),
            'consensus': NashEthereumConsensus(),
            'researcher': ResearchTaskExecutor(),
            'corrector': SelfCorrectingResearchSystem()
        }
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        self.event_bus.on('research_completed', self._handle_research_completed)
        self.event_bus.on('correction_completed', self._handle_correction_completed)
        self.event_bus.on('verification_completed', self._handle_verification_completed)
        self.event_bus.on('consensus_needed', self._handle_consensus_needed)
    
    async def _handle_research_completed(self, event: Dict[str, Any]):
        # Trigger correction after research
        content = event['data']['content']
        correction_result = await self.components['corrector'].process_query(
            query=event['data']['query'],
            initial_content=content
        )
        
        self.event_bus.emit('correction_completed', {
            'original_event': event,
            'correction': correction_result
        })
    
    async def _handle_correction_completed(self, event: Dict[str, Any]):
        # Trigger verification after correction
        corrected_content = event['data']['correction'].corrected_content
        sources = event['data']['original_event']['data'].get('sources', [])
        
        verification_result = await self.components['verifier'].verify_claim(
            claim=corrected_content,
            sources=sources
        )
        
        self.event_bus.emit('verification_completed', {
            'original_event': event,
            'verification': verification_result
        })
```

## Data Flow Patterns

### Sequential Pipeline

```python
class SequentialPipeline:
    def __init__(self):
        self.stages = [
            ('research', ResearchTaskExecutor()),
            ('correction', SelfCorrectingResearchSystem()),
            ('verification', VerificationSystem()),
            ('consensus', NashEthereumConsensus())
        ]
    
    async def execute(self, initial_input):
        current_data = initial_input
        results = {}
        
        for stage_name, component in self.stages:
            if stage_name == 'research':
                result = await component.execute_task(current_data['task'])
                current_data['content'] = result.result_data
            elif stage_name == 'correction':
                result = await component.process_query(
                    query=current_data['query'],
                    initial_content=current_data['content']
                )
                current_data['corrected'] = result.corrected_content
            elif stage_name == 'verification':
                result = await component.verify_claim(
                    claim=current_data['corrected'],
                    sources=current_data.get('sources', [])
                )
                current_data['verified'] = result
            elif stage_name == 'consensus':
                result = await component.orchestrate_consensus(
                    task=current_data['consensus_task'],
                    agents=current_data['agents']
                )
                current_data['consensus'] = result
            
            results[stage_name] = result
        
        return results
```

### Parallel Processing

```python
class ParallelIntegration:
    async def execute_parallel_verification_consensus(self, claim, sources, agents):
        # Execute verification and prepare consensus in parallel
        verification_task = self.verifier.verify_claim(claim, sources)
        consensus_prep_task = self.consensus.prepare_agents(agents)
        
        # Wait for both to complete
        verification_result, prepared_agents = await asyncio.gather(
            verification_task,
            consensus_prep_task
        )
        
        # Proceed with consensus if verification passes
        if verification_result.confidence > 0.7:
            consensus_result = await self.consensus.orchestrate_consensus(
                task={'claim': claim, 'verification': verification_result},
                agents=prepared_agents
            )
            return consensus_result
        
        return None
```

## Component Communication

### Message Passing

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class ComponentMessage:
    sender: str
    receiver: str
    message_type: str
    payload: Any
    correlation_id: str

class MessageBus:
    def __init__(self):
        self.subscribers = {}
        self.message_queue = asyncio.Queue()
    
    async def publish(self, message: ComponentMessage):
        await self.message_queue.put(message)
    
    async def subscribe(self, component_id: str, handler):
        self.subscribers[component_id] = handler
    
    async def process_messages(self):
        while True:
            message = await self.message_queue.get()
            if message.receiver in self.subscribers:
                await self.subscribers[message.receiver](message)
```

### Shared State Management

```python
class SharedStateManager:
    def __init__(self):
        self.state = {}
        self.locks = {}
    
    async def update_state(self, key: str, value: Any):
        if key not in self.locks:
            self.locks[key] = asyncio.Lock()
        
        async with self.locks[key]:
            self.state[key] = value
            await self._notify_observers(key, value)
    
    async def get_state(self, key: str) -> Any:
        return self.state.get(key)
    
    async def _notify_observers(self, key: str, value: Any):
        # Notify components watching this state
        pass
```

## Performance Optimization

### GPU Resource Sharing

```python
class GPUResourceManager:
    def __init__(self, num_gpus: int):
        self.num_gpus = num_gpus
        self.gpu_allocation = {}
        self.gpu_locks = [asyncio.Lock() for _ in range(num_gpus)]
    
    async def allocate_gpu(self, component_id: str) -> int:
        # Find least loaded GPU
        gpu_loads = await self._get_gpu_loads()
        best_gpu = min(range(self.num_gpus), key=lambda i: gpu_loads[i])
        
        async with self.gpu_locks[best_gpu]:
            self.gpu_allocation[component_id] = best_gpu
            return best_gpu
    
    async def release_gpu(self, component_id: str):
        if component_id in self.gpu_allocation:
            gpu_id = self.gpu_allocation[component_id]
            async with self.gpu_locks[gpu_id]:
                del self.gpu_allocation[component_id]
```

### Caching Strategy

```python
class IntegratedCacheManager:
    def __init__(self):
        self.verification_cache = LRUCache(maxsize=1000)
        self.research_cache = LRUCache(maxsize=500)
        self.consensus_cache = LRUCache(maxsize=100)
    
    async def get_or_compute(
        self,
        cache_key: str,
        component: str,
        compute_func
    ):
        # Check appropriate cache
        cache = getattr(self, f"{component}_cache")
        
        if cache_key in cache:
            return cache[cache_key]
        
        # Compute and cache
        result = await compute_func()
        cache[cache_key] = result
        return result
```

## Error Handling and Recovery

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure = None
        self.state = "closed"  # closed, open, half-open
    
    async def call(self, func, *args, **kwargs):
        if self.state == "open":
            if time.time() - self.last_failure > self.timeout:
                self.state = "half-open"
            else:
                raise Exception("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            if self.state == "half-open":
                self.state = "closed"
                self.failure_count = 0
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "open"
            
            raise e
```

### Fallback Strategies

```python
class IntegrationFallbackManager:
    async def execute_with_fallback(self, primary_func, fallback_func):
        try:
            return await primary_func()
        except Exception as e:
            logger.warning(f"Primary function failed: {e}")
            return await fallback_func()
    
    async def verification_with_fallback(self, claim, sources):
        async def primary():
            return await self.verifier.verify_claim(claim, sources)
        
        async def fallback():
            # Simplified verification without GPU
            simple_verifier = VerificationSystem({'device': 'cpu'})
            return await simple_verifier.verify_claim(claim, sources[:5])
        
        return await self.execute_with_fallback(primary, fallback)
```

## Production Deployment

### Health Monitoring

```python
class IntegratedHealthMonitor:
    def __init__(self, components):
        self.components = components
        self.health_checks = {}
    
    async def check_health(self):
        results = {}
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'health_check'):
                    results[name] = await component.health_check()
                else:
                    results[name] = {'status': 'unknown'}
            except Exception as e:
                results[name] = {'status': 'unhealthy', 'error': str(e)}
        
        return results
    
    @app.get("/health")
    async def health_endpoint():
        health_status = await health_monitor.check_health()
        overall_status = all(
            c.get('status') == 'healthy' 
            for c in health_status.values()
        )
        
        return {
            'status': 'healthy' if overall_status else 'degraded',
            'components': health_status,
            'timestamp': datetime.now().isoformat()
        }
```

## Next Steps

- [Verification-Consensus Integration](verification-consensus.md)
- [Knowledge Graph Integration](knowledge-graph.md)
- [System Architecture](architecture.md)
- [Production Deployment](deployment.md)
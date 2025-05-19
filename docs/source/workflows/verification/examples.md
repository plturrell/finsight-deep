# Verification System Examples

## Basic Examples

### Simple Claim Verification

```python
import asyncio
from aiq.verification import VerificationSystem

async def verify_simple_claim():
    # Initialize system
    verifier = VerificationSystem({
        'enable_source_validation': True,
        'confidence_methods': ['bayesian']
    })
    
    # Verify a factual claim
    claim = "The Eiffel Tower is 330 meters tall"
    sources = [
        {
            'url': 'https://www.eiffel-tower.com/facts',
            'type': 'website',
            'title': 'Official Eiffel Tower Website'
        }
    ]
    
    result = await verifier.verify_claim(claim, sources)
    print(f"Confidence: {result.confidence:.2%}")
    return result

# Run verification
asyncio.run(verify_simple_claim())
```

### Multi-Source Verification

```python
async def verify_with_multiple_sources():
    verifier = VerificationSystem({
        'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer']
    })
    
    claim = "Bitcoin's market cap exceeds $1 trillion"
    sources = [
        {
            'url': 'https://coinmarketcap.com/currencies/bitcoin',
            'type': 'api',
            'title': 'CoinMarketCap API'
        },
        {
            'url': 'https://www.coingecko.com/en/coins/bitcoin',
            'type': 'api',
            'title': 'CoinGecko API'
        },
        {
            'url': 'https://finance.yahoo.com/quote/BTC-USD',
            'type': 'website',
            'title': 'Yahoo Finance'
        }
    ]
    
    result = await verifier.verify_claim(claim, sources)
    
    # Analyze results by method
    print(f"Overall confidence: {result.confidence:.2%}")
    for method, score in result.method_scores.items():
        print(f"{method.value}: {score:.2%}")
```

## Advanced Examples

### Custom Source Validation

```python
from aiq.verification import SourceValidator, Source, SourceType

class AcademicSourceValidator(SourceValidator):
    def __init__(self, device='cuda'):
        super().__init__(device)
        self.journal_rankings = self._load_journal_rankings()
    
    async def validate_source(self, source: Source) -> float:
        if source.type == SourceType.PAPER:
            # Custom validation for academic papers
            score = 0.5  # Base score
            
            # Check journal ranking
            if source.title:
                journal = self._extract_journal(source.title)
                if journal in self.journal_rankings:
                    score += self.journal_rankings[journal] * 0.3
            
            # Check citation count (would need API call)
            citations = await self._get_citation_count(source.url)
            if citations > 100:
                score += 0.2
            
            return min(score, 0.95)
        
        return await super().validate_source(source)

# Use custom validator
async def verify_academic_claim():
    verifier = VerificationSystem({'device': 'cuda'})
    verifier.source_validator = AcademicSourceValidator()
    
    claim = "Transformer models outperform RNNs on machine translation"
    sources = [
        {
            'url': 'https://arxiv.org/abs/1706.03762',
            'type': 'paper',
            'title': 'Attention Is All You Need - NIPS 2017'
        }
    ]
    
    result = await verifier.verify_claim(claim, sources)
    return result
```

### Batch Processing

```python
async def batch_verify_claims():
    verifier = VerificationSystem({
        'device': 'cuda',
        'confidence_methods': ['ensemble']
    })
    
    # Multiple claims to verify
    claims_and_sources = [
        (
            "Tesla is the world's most valuable automaker",
            [{'url': 'https://finance.yahoo.com/quote/TSLA', 'type': 'api'}]
        ),
        (
            "SpaceX successfully landed a rocket",
            [{'url': 'https://www.spacex.com/launches', 'type': 'website'}]
        ),
        (
            "Apple released the iPhone 15",
            [{'url': 'https://www.apple.com/newsroom', 'type': 'website'}]
        )
    ]
    
    # Verify all claims concurrently
    tasks = [
        verifier.verify_claim(claim, sources)
        for claim, sources in claims_and_sources
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Summarize results
    for i, result in enumerate(results):
        claim = claims_and_sources[i][0]
        print(f"\nClaim: {claim[:50]}...")
        print(f"Confidence: {result.confidence:.2%}")
        print(f"Time: {result.verification_time_ms:.1f}ms")
    
    return results
```

### Real-time Monitoring

```python
import time
from typing import AsyncGenerator

async def real_time_verification_monitor(
    claim_stream: AsyncGenerator[str, None]
) -> None:
    verifier = VerificationSystem({
        'device': 'cuda',
        'confidence_methods': ['bayesian']
    })
    
    async for claim in claim_stream:
        start_time = time.time()
        
        # Quick source lookup (in practice, use proper source discovery)
        sources = await discover_sources(claim)
        
        # Verify claim
        result = await verifier.verify_claim(claim, sources)
        
        # Alert on low confidence
        if result.confidence < 0.5:
            await send_alert(
                f"Low confidence claim detected: {claim[:50]}... "
                f"(confidence: {result.confidence:.2%})"
            )
        
        # Log metrics
        await log_verification_metrics(
            claim=claim,
            confidence=result.confidence,
            latency=time.time() - start_time,
            gpu_usage=result.gpu_utilization
        )
```

## Integration Examples

### With Nash-Ethereum Consensus

```python
from aiq.neural import NashEthereumConsensus
from aiq.verification import VerificationSystem

async def consensus_with_verification():
    # Initialize systems
    consensus = NashEthereumConsensus()
    verifier = VerificationSystem({
        'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer']
    })
    
    # Financial claim to verify and reach consensus on
    claim = "Fed will raise interest rates by 0.25%"
    sources = [
        {'url': 'https://www.federalreserve.gov/news', 'type': 'website'},
        {'url': 'https://www.bloomberg.com/markets', 'type': 'website'},
        {'url': 'https://api.fred.stlouisfed.org/series', 'type': 'api'}
    ]
    
    # Step 1: Verify claim
    verification = await verifier.verify_claim(claim, sources)
    
    if verification.confidence > 0.7:
        # Step 2: High confidence - proceed to consensus
        task = {
            'type': 'market_prediction',
            'claim': claim,
            'verification': {
                'confidence': verification.confidence,
                'sources': len(verification.sources_verified),
                'methods': list(verification.method_scores.keys())
            }
        }
        
        agents = list(consensus.agents.values())
        consensus_result = await consensus.orchestrate_consensus(
            task, agents, hybrid_mode=True
        )
        
        return {
            'claim': claim,
            'verification_confidence': verification.confidence,
            'consensus_reached': consensus_result.converged,
            'final_position': consensus_result.nash_equilibrium
        }
    else:
        return {
            'claim': claim,
            'verification_confidence': verification.confidence,
            'consensus_reached': False,
            'reason': 'Insufficient verification confidence'
        }
```

### With Digital Human

```python
from aiq.digital_human.conversation import SgLangConversationEngine
from aiq.verification import VerificationSystem

class VerifiedConversationEngine(SgLangConversationEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = VerificationSystem({
            'confidence_methods': ['ensemble']
        })
    
    async def generate_verified_response(
        self,
        user_query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Generate initial response
        response = await self.generate_response(user_query, context)
        
        # Extract claims from response
        claims = await self.extract_claims(response)
        
        # Verify each claim
        verification_results = []
        for claim in claims:
            sources = await self.find_sources_for_claim(claim)
            result = await self.verifier.verify_claim(claim, sources)
            verification_results.append(result)
        
        # Annotate response with verification
        annotated_response = self.annotate_with_verification(
            response, claims, verification_results
        )
        
        return {
            'response': annotated_response,
            'verifications': verification_results,
            'overall_confidence': np.mean([r.confidence for r in verification_results])
        }
```

### With Research Components

```python
from aiq.research import ResearchTaskExecutor, TaskType
from aiq.verification import VerificationSystem

async def research_with_verification():
    executor = ResearchTaskExecutor(enable_optimization=True)
    verifier = VerificationSystem({'device': 'cuda'})
    
    # Research task
    research_task = ResearchTask(
        task_id="research_001",
        task_type=TaskType.SYNTHESIS,
        query="Latest advances in quantum computing applications"
    )
    
    # Execute research
    research_result = await executor.execute_task(research_task)
    
    # Extract and verify key claims
    synthesis = research_result.result_data.get('synthesis', '')
    claims = extract_key_claims(synthesis)
    
    verified_claims = []
    for claim in claims:
        # Find sources from research
        sources = find_sources_in_research(claim, research_result)
        
        # Verify claim
        verification = await verifier.verify_claim(claim, sources)
        
        verified_claims.append({
            'claim': claim,
            'confidence': verification.confidence,
            'sources': len(verification.sources_verified),
            'provenance': verification.provenance_chain
        })
    
    return {
        'research_task': research_task.task_id,
        'synthesis': synthesis,
        'verified_claims': verified_claims,
        'high_confidence_claims': [
            c for c in verified_claims if c['confidence'] > 0.8
        ]
    }
```

## Performance Examples

### GPU Optimization

```python
import torch
import time

async def benchmark_gpu_verification():
    # CPU verification
    cpu_verifier = VerificationSystem({'device': 'cpu'})
    
    # GPU verification
    gpu_verifier = VerificationSystem({'device': 'cuda'})
    
    # Test claims
    claims = [f"Test claim {i}" for i in range(100)]
    sources = [[{'url': f'http://example.com/{i}', 'type': 'website'}] 
               for i in range(100)]
    
    # Benchmark CPU
    cpu_start = time.time()
    cpu_results = []
    for claim, source in zip(claims, sources):
        result = await cpu_verifier.verify_claim(claim, source)
        cpu_results.append(result)
    cpu_time = time.time() - cpu_start
    
    # Benchmark GPU
    gpu_start = time.time()
    gpu_results = []
    for claim, source in zip(claims, sources):
        result = await gpu_verifier.verify_claim(claim, source)
        gpu_results.append(result)
    gpu_time = time.time() - gpu_start
    
    print(f"CPU time: {cpu_time:.2f}s")
    print(f"GPU time: {gpu_time:.2f}s")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Compare GPU utilization
    avg_gpu_util = np.mean([r.gpu_utilization for r in gpu_results])
    print(f"Average GPU utilization: {avg_gpu_util:.1%}")
```

### Caching Example

```python
async def cached_verification_example():
    verifier = VerificationSystem({
        'cache_embeddings': True,
        'confidence_methods': ['bayesian']
    })
    
    claim = "The Earth is round"
    sources = [{'url': 'https://nasa.gov/facts', 'type': 'website'}]
    
    # First call - computes and caches
    start_time = time.time()
    result1 = await verifier.verify_claim(claim, sources)
    first_call_time = time.time() - start_time
    
    # Second call - uses cache
    start_time = time.time()
    result2 = await verifier.verify_claim(claim, sources)
    second_call_time = time.time() - start_time
    
    print(f"First call: {first_call_time*1000:.1f}ms")
    print(f"Second call: {second_call_time*1000:.1f}ms")
    print(f"Speedup: {first_call_time/second_call_time:.1f}x")
    
    # Results should be identical
    assert result1.confidence == result2.confidence
```

## Production Examples

### Error Handling

```python
async def robust_verification():
    verifier = VerificationSystem({
        'confidence_methods': ['bayesian', 'fuzzy']
    })
    
    try:
        # Verify with timeout
        result = await asyncio.wait_for(
            verifier.verify_claim("Complex claim", sources),
            timeout=5.0  # 5 second timeout
        )
    except asyncio.TimeoutError:
        logger.error("Verification timed out")
        result = VerificationResult(
            claim="Complex claim",
            confidence=0.0,
            sources_verified=[],
            method_scores={},
            provenance_chain=[],
            verification_time_ms=5000,
            gpu_utilization=0.0,
            explanations={"error": "Timeout"}
        )
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        result = None
    
    return result
```

### Monitoring and Metrics

```python
from prometheus_client import Counter, Histogram, Gauge

# Prometheus metrics
verification_counter = Counter('verifications_total', 'Total verifications')
verification_time = Histogram('verification_duration_seconds', 'Verification duration')
confidence_gauge = Gauge('verification_confidence', 'Current confidence level')

async def monitored_verification(claim: str, sources: List[Dict]):
    verifier = VerificationSystem({'device': 'cuda'})
    
    # Increment counter
    verification_counter.inc()
    
    # Time the verification
    with verification_time.time():
        result = await verifier.verify_claim(claim, sources)
    
    # Update confidence gauge
    confidence_gauge.set(result.confidence)
    
    # Log to monitoring system
    await send_to_monitoring({
        'claim': claim[:100],
        'confidence': result.confidence,
        'time_ms': result.verification_time_ms,
        'gpu_usage': result.gpu_utilization,
        'sources': len(result.sources_verified)
    })
    
    return result
```
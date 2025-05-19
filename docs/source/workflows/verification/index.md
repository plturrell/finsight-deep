# Real-time Citation Verification System

## Overview

The AIQToolkit Verification System provides real-time citation verification and fact-checking capabilities to prevent hallucinations and ensure information accuracy. It implements W3C PROV-compliant provenance tracking with multi-method confidence scoring, all optimized for GPU acceleration.

## Key Features

- **Multi-Method Confidence Scoring**: Bayesian, Fuzzy Logic, and Dempster-Shafer evidence theory
- **W3C PROV-Compliant Provenance**: Complete audit trail of verification activities
- **Source Validation**: Automatic credibility assessment of sources
- **GPU Acceleration**: Optimized for NVIDIA hardware
- **Real-time Performance**: Sub-second verification latency
- **Blockchain Integration**: Immutable verification records

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   Verification System                        │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐│
│  │ Source        │    │ Confidence    │    │ Provenance    ││
│  │ Validator     │    │ Scorers       │    │ Tracker       ││
│  │               │    │ - Bayesian    │    │ (W3C PROV)    ││
│  │               │    │ - Fuzzy       │    │               ││
│  │               │    │ - D-S Theory  │    │               ││
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘│
│          │                    │                    │         │
│  ┌───────┴───────────────────┴────────────────────┴───────┐ │
│  │                    GPU Acceleration Layer              │ │
│  │                 (CUDA, TensorRT, Triton)              │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from aiq.verification import (
    VerificationSystem,
    ConfidenceMethod,
    SourceType,
    Source
)

# Initialize verification system
config = {
    'enable_source_validation': True,
    'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer'],
    'device': 'cuda'  # Use GPU acceleration
}

verifier = VerificationSystem(config)

# Verify a claim
claim = "GPT-3 has 175 billion parameters"
sources = [
    {
        'url': 'https://arxiv.org/abs/2005.14165',
        'type': 'paper',
        'title': 'Language Models are Few-Shot Learners'
    }
]

result = await verifier.verify_claim(claim, sources)

print(f"Confidence: {result.confidence:.2%}")
print(f"Method scores: {result.method_scores}")
print(f"Verification time: {result.verification_time_ms}ms")
```

### Integration with Consensus System

```python
from aiq.neural import NashEthereumConsensus
from aiq.verification import VerificationSystem

# Create verification-enabled consensus
consensus = NashEthereumConsensus()
verifier = VerificationSystem(config)

# Verify before consensus
claim = "Market will rise 10% next quarter"
sources = await gather_market_sources()

verification = await verifier.verify_claim(claim, sources)

if verification.confidence > 0.8:
    # High confidence - proceed with consensus
    consensus_result = await consensus.process_with_verification(
        claim, 
        verification
    )
```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_source_validation` | bool | True | Whether to validate source credibility |
| `confidence_methods` | List[str] | ['bayesian'] | Confidence calculation methods |
| `device` | str | 'cuda' | Device for computation ('cuda' or 'cpu') |
| `cache_embeddings` | bool | True | Cache source embeddings for performance |
| `provenance_enabled` | bool | True | Track W3C PROV provenance |

## Source Types

The system supports various source types with different reliability weights:

- `PAPER`: Academic papers (0.8 base reliability)
- `BOOK`: Published books (0.75)
- `DATABASE`: Structured databases (0.9)
- `API`: API endpoints (0.85)
- `WEBSITE`: General websites (0.6)
- `EXPERT`: Domain experts (0.7)

## Confidence Methods

### Bayesian Inference

Uses prior beliefs and evidence to calculate posterior confidence:

```python
P(claim|evidence) = P(evidence|claim) × P(claim) / P(evidence)
```

### Fuzzy Logic

Applies linguistic variables and membership functions for uncertainty:

```python
confidence = fuzzy_rules(evidence_strength, source_reliability)
```

### Dempster-Shafer Theory

Combines evidence from multiple sources, handling conflicting information:

```python
m₁₂(A) = (Σ m₁(B)m₂(C)) / (1 - Σ m₁(B)m₂(C))
```

## Best Practices

1. **Use Multiple Sources**: Verify claims against multiple independent sources
2. **Enable GPU Acceleration**: Use CUDA-enabled devices for better performance
3. **Configure Confidence Methods**: Choose methods appropriate for your domain
4. **Monitor Performance**: Track verification times and GPU utilization
5. **Store Provenance**: Maintain complete audit trails for compliance

## Performance Optimization

- **Batch Processing**: Verify multiple claims in parallel
- **Caching**: Enable embedding cache for repeated sources
- **GPU Memory**: Monitor and optimize GPU memory usage
- **Async Operations**: Use asynchronous verification for better throughput

## Next Steps

- [API Reference](api-reference.md)
- [Confidence Methods Guide](confidence-methods.md)
- [Provenance Tracking](provenance-tracking.md)
- [Examples](examples.md)
# Verification System API Reference

## Classes

### VerificationSystem

Main class for citation verification and fact-checking.

```python
class VerificationSystem:
    def __init__(self, config: Dict[str, Any])
    async def verify_claim(self, claim: str, sources: List[Dict[str, Any]]) -> VerificationResult
```

#### Parameters

- `config`: Configuration dictionary with the following options:
  - `enable_source_validation` (bool): Enable automatic source validation
  - `confidence_methods` (List[str]): List of confidence calculation methods
  - `device` (str): Device for computation ('cuda' or 'cpu')

#### Methods

##### verify_claim

Verify a claim against provided sources.

```python
async def verify_claim(
    self,
    claim: str,
    sources: List[Dict[str, Any]]
) -> VerificationResult
```

**Parameters:**
- `claim`: The claim to verify
- `sources`: List of source dictionaries containing:
  - `url`: Source URL
  - `type`: Source type (paper, book, website, etc.)
  - `title`: Optional source title
  - `author`: Optional author

**Returns:**
- `VerificationResult` object

### VerificationResult

Results from verification process.

```python
@dataclass
class VerificationResult:
    claim: str
    confidence: float
    sources_verified: List[Source]
    method_scores: Dict[ConfidenceMethod, float]
    provenance_chain: List[ProvenanceRecord]
    verification_time_ms: float
    gpu_utilization: float
    explanations: Dict[str, str]
```

#### Fields

- `claim`: Original claim text
- `confidence`: Overall confidence score (0-1)
- `sources_verified`: List of validated sources
- `method_scores`: Individual scores from each confidence method
- `provenance_chain`: W3C PROV-compliant provenance records
- `verification_time_ms`: Processing time in milliseconds
- `gpu_utilization`: GPU usage percentage
- `explanations`: Human-readable explanations

### Source

Represents a verification source.

```python
@dataclass
class Source:
    url: str
    type: SourceType
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[datetime] = None
    reliability_score: float = 0.5
```

### ProvenanceRecord

W3C PROV-compliant provenance record.

```python
@dataclass
class ProvenanceRecord:
    entity_id: str
    activity_id: str
    agent_id: str
    timestamp: datetime
    derivation: Optional[str] = None
    attribution: Optional[str] = None
```

## Enums

### ConfidenceMethod

Available confidence calculation methods.

```python
class ConfidenceMethod(Enum):
    BAYESIAN = "bayesian"
    FUZZY = "fuzzy"
    DEMPSTER_SHAFER = "dempster_shafer"
    ENSEMBLE = "ensemble"
```

### SourceType

Types of verification sources.

```python
class SourceType(Enum):
    PAPER = "paper"
    BOOK = "book"
    WEBSITE = "website"
    DATABASE = "database"
    API = "api"
    EXPERT = "expert"
```

## Confidence Scorers

### BayesianConfidenceScorer

Implements Bayesian inference for confidence scoring.

```python
class BayesianConfidenceScorer:
    def __init__(self)
    def score(self, evidence: List[float]) -> float
```

### FuzzyLogicScorer

Implements fuzzy logic for handling uncertainty.

```python
class FuzzyLogicScorer:
    def __init__(self)
    def score(self, evidence: List[float]) -> float
```

### DempsterShaferScorer

Implements Dempster-Shafer evidence theory.

```python
class DempsterShaferScorer:
    def __init__(self)
    def score(self, mass_functions: List[Dict[str, float]]) -> float
```

## Factory Functions

### create_verification_system

Factory function to create a verification system.

```python
def create_verification_system(config: Dict[str, Any]) -> VerificationSystem
```

## Error Handling

The verification system may raise the following exceptions:

- `ValueError`: Invalid configuration or parameters
- `RuntimeError`: GPU initialization or processing errors
- `TimeoutError`: Verification timeout exceeded

## Example Usage

### Complete Verification Example

```python
import asyncio
from aiq.verification import (
    create_verification_system,
    SourceType,
    ConfidenceMethod
)

async def verify_research_claim():
    # Configure verification system
    config = {
        'enable_source_validation': True,
        'confidence_methods': ['bayesian', 'fuzzy', 'dempster_shafer'],
        'device': 'cuda'
    }
    
    verifier = create_verification_system(config)
    
    # Define claim and sources
    claim = "Transformer models achieve state-of-the-art performance on NLP tasks"
    sources = [
        {
            'url': 'https://arxiv.org/abs/1706.03762',
            'type': 'paper',
            'title': 'Attention Is All You Need',
            'author': 'Vaswani et al.'
        },
        {
            'url': 'https://arxiv.org/abs/1810.04805',
            'type': 'paper',
            'title': 'BERT: Pre-training of Deep Bidirectional Transformers'
        }
    ]
    
    # Verify claim
    result = await verifier.verify_claim(claim, sources)
    
    # Process results
    print(f"Claim: {result.claim}")
    print(f"Overall confidence: {result.confidence:.2%}")
    print("\nConfidence by method:")
    for method, score in result.method_scores.items():
        print(f"  {method.value}: {score:.2%}")
    
    print(f"\nVerification time: {result.verification_time_ms:.1f}ms")
    print(f"GPU utilization: {result.gpu_utilization:.1%}")
    
    # Check provenance
    print("\nProvenance chain:")
    for record in result.provenance_chain:
        print(f"  {record.timestamp}: {record.activity_id}")
    
    return result

# Run verification
result = asyncio.run(verify_research_claim())
```

### Batch Verification

```python
async def batch_verify_claims(claims_and_sources):
    verifier = create_verification_system(config)
    
    # Verify multiple claims concurrently
    tasks = [
        verifier.verify_claim(claim, sources)
        for claim, sources in claims_and_sources
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

### Custom Source Validation

```python
class CustomSourceValidator(SourceValidator):
    async def validate_source(self, source: Source) -> float:
        # Custom validation logic
        if source.type == SourceType.PAPER:
            # Check citation count, impact factor, etc.
            return await self._validate_academic_paper(source)
        else:
            return await super().validate_source(source)

# Use custom validator
verifier = VerificationSystem(config)
verifier.source_validator = CustomSourceValidator()
```
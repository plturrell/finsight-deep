# Neural-Symbolic Retriever

## Overview

The Neural-Symbolic Retriever combines the semantic understanding capabilities of neural networks with the logical reasoning power of symbolic AI. This hybrid approach provides more accurate and explainable retrieval results by leveraging both deep learning and knowledge graph technologies.

## Key Features

- **Hybrid Architecture**: Combines neural embeddings with symbolic reasoning
- **Knowledge Graph Integration**: Direct connection to RDF/SPARQL endpoints
- **Multi-hop Reasoning**: Performs logical inference across knowledge graphs
- **Explainable Results**: Provides reasoning chains for transparency
- **GPU Acceleration**: Optimized neural components for fast retrieval
- **Context-Aware**: Understands relationships and context

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Neural-Symbolic Retriever                     │
├─────────────────────────────────────────────────────────────┤
│  ┌───────────────┐    ┌───────────────┐    ┌───────────────┐│
│  │ Neural Encoder│    │ Symbolic      │    │ Hybrid        ││
│  │ - Embeddings  │    │ Reasoner      │    │ Integrator    ││
│  │ - Similarity  │    │ - SPARQL      │    │ - Fusion      ││
│  │ - Context     │    │ - Inference   │    │ - Ranking     ││
│  └───────┬───────┘    └───────┬───────┘    └───────┬───────┘│
│          │                    │                    │         │
│  ┌───────┴────────────────────┴────────────────────┴────────┐│
│  │          Knowledge Graph + Vector Database               ││
│  └──────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────┘
```

## Basic Usage

```python
from aiq.retriever.neural_symbolic import (
    NeuralSymbolicRetriever,
    KnowledgeGraph,
    RetrievalConfig
)

# Initialize retriever
retriever = NeuralSymbolicRetriever(
    knowledge_graph_endpoint="http://localhost:3030/sparql",
    embedding_dim=768,
    device="cuda",
    enable_reasoning=True
)

# Simple retrieval
results = await retriever.retrieve(
    query="What are the applications of quantum computing?",
    top_k=10
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Content: {result.content}")
    print(f"Reasoning: {result.reasoning_chain}")
```

## Knowledge Graph Integration

### Configuration

```python
# Configure knowledge graph connection
kg_config = {
    "endpoint": "http://localhost:3030/sparql",
    "dataset": "research_knowledge",
    "auth": ("user", "password"),  # Optional
    "timeout": 30,
    "cache_queries": True
}

retriever = NeuralSymbolicRetriever(
    knowledge_graph_config=kg_config,
    use_graph_embeddings=True
)
```

### SPARQL Queries

```python
# Custom SPARQL template
sparql_template = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX dbo: <http://dbpedia.org/ontology/>

SELECT ?entity ?label ?description
WHERE {
    ?entity rdfs:label ?label .
    ?entity dbo:abstract ?description .
    FILTER(CONTAINS(LCASE(?description), LCASE("$QUERY")))
}
LIMIT $LIMIT
"""

retriever.set_sparql_template(sparql_template)
```

### Graph Construction

```python
# Build knowledge graph from documents
graph_builder = KnowledgeGraphBuilder()

# Add documents
documents = [
    {"id": "doc1", "content": "Quantum computing uses qubits..."},
    {"id": "doc2", "content": "Machine learning algorithms..."}
]

for doc in documents:
    entities = graph_builder.extract_entities(doc["content"])
    relations = graph_builder.extract_relations(doc["content"])
    graph_builder.add_to_graph(entities, relations)

# Export to RDF
graph_builder.export_rdf("knowledge_graph.ttl")

# Load into retriever
retriever.load_knowledge_graph("knowledge_graph.ttl")
```

## Neural Components

### Embedding Generation

```python
# Configure neural encoder
encoder_config = {
    "model_name": "sentence-transformers/all-mpnet-base-v2",
    "max_length": 512,
    "normalize_embeddings": True,
    "pooling_strategy": "mean"
}

retriever.configure_encoder(encoder_config)

# Generate embeddings
embeddings = await retriever.generate_embeddings([
    "Quantum computing applications",
    "Neural network architectures",
    "Symbolic AI reasoning"
])
```

### Similarity Search

```python
# Configure similarity metrics
retriever.set_similarity_metric("cosine")  # or "euclidean", "dot_product"

# Perform similarity search
query_embedding = await retriever.encode_query("quantum computing")
similar_docs = await retriever.similarity_search(
    query_embedding,
    top_k=20,
    threshold=0.7
)
```

## Symbolic Reasoning

### Multi-hop Reasoning

```python
# Enable multi-hop reasoning
reasoning_config = {
    "max_hops": 3,
    "inference_rules": ["transitive", "symmetric", "inverse"],
    "confidence_threshold": 0.6
}

retriever.configure_reasoning(reasoning_config)

# Query with reasoning
results = await retriever.retrieve_with_reasoning(
    query="How does quantum computing relate to cryptography?",
    reasoning_depth=3
)

# Examine reasoning chains
for result in results:
    print(f"Content: {result.content}")
    print("Reasoning chain:")
    for step in result.reasoning_chain:
        print(f"  {step.relation}: {step.source} -> {step.target}")
        print(f"  Confidence: {step.confidence:.2f}")
```

### Logical Inference

```python
# Define inference rules
inference_rules = [
    # Transitive property
    {
        "name": "transitive_subclass",
        "pattern": "(?x rdfs:subClassOf ?y) ∧ (?y rdfs:subClassOf ?z)",
        "inference": "?x rdfs:subClassOf ?z"
    },
    # Symmetric property
    {
        "name": "symmetric_related",
        "pattern": "(?x :relatedTo ?y)",
        "inference": "?y :relatedTo ?x"
    }
]

retriever.add_inference_rules(inference_rules)
```

## Hybrid Integration

### Fusion Strategies

```python
# Configure fusion of neural and symbolic results
fusion_config = {
    "strategy": "weighted_combination",  # or "reranking", "cascading"
    "neural_weight": 0.6,
    "symbolic_weight": 0.4,
    "normalize_scores": True
}

retriever.configure_fusion(fusion_config)

# Retrieve with hybrid approach
results = await retriever.hybrid_retrieve(
    query="Quantum computing security implications",
    use_neural=True,
    use_symbolic=True,
    use_reasoning=True
)
```

### Context Enhancement

```python
# Enable context-aware retrieval
context_config = {
    "use_entity_context": True,
    "expand_with_related": True,
    "context_window": 3,
    "weight_by_distance": True
}

retriever.configure_context(context_config)

# Query with context
results = await retriever.retrieve_with_context(
    query="transformer architecture",
    context={
        "domain": "natural language processing",
        "related_concepts": ["attention", "BERT", "GPT"]
    }
)
```

## Advanced Features

### Query Expansion

```python
# Enable query expansion
expansion_config = {
    "use_synonyms": True,
    "use_hypernyms": True,
    "use_related_concepts": True,
    "max_expansions": 5
}

retriever.configure_query_expansion(expansion_config)

# Retrieve with expanded query
expanded_results = await retriever.retrieve_with_expansion(
    query="AI safety",
    expansion_strategy="semantic"
)
```

### Explanation Generation

```python
# Enable detailed explanations
retriever.enable_explanations(
    include_reasoning_steps=True,
    include_confidence_scores=True,
    include_provenance=True
)

# Get explained results
results = await retriever.retrieve_with_explanations(query)

for result in results:
    print(f"Content: {result.content}")
    print(f"Explanation: {result.explanation}")
    print(f"Evidence:")
    for evidence in result.evidence:
        print(f"  - {evidence.source}: {evidence.text}")
        print(f"    Confidence: {evidence.confidence:.2f}")
```

### Graph Visualization

```python
# Visualize reasoning paths
visualization_config = {
    "layout": "force_directed",
    "show_confidence": True,
    "highlight_path": True,
    "interactive": True
}

retriever.visualize_reasoning(
    query="quantum computing applications",
    output_path="reasoning_graph.html",
    config=visualization_config
)
```

## Integration Examples

### With Research Task Executor

```python
from aiq.research import ResearchTaskExecutor, ResearchTask

class SymbolicResearchExecutor(ResearchTaskExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.retriever = NeuralSymbolicRetriever(
            knowledge_graph_endpoint="http://localhost:3030/sparql"
        )
    
    async def _execute_retrieval(self, task: ResearchTask):
        # Use neural-symbolic retrieval
        results = await self.retriever.hybrid_retrieve(
            query=task.query,
            top_k=20,
            use_reasoning=True
        )
        
        # Process results
        processed_results = []
        for result in results:
            processed_results.append({
                "content": result.content,
                "score": result.score,
                "reasoning": result.reasoning_chain,
                "explanation": result.explanation
            })
        
        return {"results": processed_results}
```

### With Verification System

```python
from aiq.verification import VerificationSystem

# Create verified retriever
class VerifiedNeuralSymbolicRetriever(NeuralSymbolicRetriever):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.verifier = VerificationSystem()
    
    async def retrieve_and_verify(self, query: str, top_k: int = 10):
        # Retrieve results
        results = await self.hybrid_retrieve(query, top_k)
        
        # Verify each result
        verified_results = []
        for result in results:
            # Extract claims
            claims = self.extract_claims(result.content)
            
            # Verify claims
            verifications = []
            for claim in claims:
                verification = await self.verifier.verify_claim(
                    claim=claim,
                    sources=result.sources
                )
                verifications.append(verification)
            
            # Add verification to result
            result.verifications = verifications
            result.avg_confidence = np.mean([v.confidence for v in verifications])
            verified_results.append(result)
        
        # Re-rank by verification confidence
        verified_results.sort(key=lambda r: r.avg_confidence, reverse=True)
        
        return verified_results
```

## Performance Optimization

### GPU Acceleration

```python
# Configure GPU usage
gpu_config = {
    "device": "cuda:0",
    "batch_size": 32,
    "mixed_precision": True,
    "pin_memory": True
}

retriever.configure_gpu(gpu_config)

# Batch processing
queries = ["query1", "query2", "query3", ...]
batch_results = await retriever.batch_retrieve(
    queries,
    batch_size=16,
    num_workers=4
)
```

### Caching

```python
# Enable caching
cache_config = {
    "cache_embeddings": True,
    "cache_sparql_results": True,
    "cache_reasoning_paths": True,
    "max_cache_size_mb": 1024,
    "ttl_seconds": 3600
}

retriever.configure_caching(cache_config)

# Use cached results
results = await retriever.retrieve(
    query="cached query",
    use_cache=True
)
```

### Index Optimization

```python
# Build optimized indices
index_config = {
    "vector_index_type": "hnsw",  # or "flat", "ivf", "lsh"
    "graph_index_type": "property_graph",
    "build_gpu_index": True
}

retriever.build_indices(index_config)

# Save indices
retriever.save_indices("indices/")

# Load pre-built indices
retriever.load_indices("indices/")
```

## Monitoring and Analytics

### Performance Metrics

```python
# Enable metrics collection
retriever.enable_metrics(
    track_latency=True,
    track_reasoning_depth=True,
    track_cache_hits=True
)

# Execute queries
results = await retriever.retrieve(query)

# Get metrics
metrics = retriever.get_metrics()
print(f"Average latency: {metrics['avg_latency_ms']:.1f}ms")
print(f"Cache hit rate: {metrics['cache_hit_rate']:.2%}")
print(f"Avg reasoning depth: {metrics['avg_reasoning_depth']:.1f}")
```

### Query Analytics

```python
# Analyze query patterns
analytics = retriever.analyze_queries(
    time_window="24h",
    group_by="domain"
)

print("Query distribution by domain:")
for domain, count in analytics['domain_distribution'].items():
    print(f"  {domain}: {count}")

print("\nMost common reasoning patterns:")
for pattern in analytics['reasoning_patterns'][:5]:
    print(f"  {pattern['pattern']}: {pattern['count']}")
```

## Best Practices

### Configuration Guidelines

1. **Balance Neural/Symbolic**: Adjust weights based on your domain
2. **Optimize Reasoning Depth**: Deeper isn't always better
3. **Cache Strategically**: Cache expensive SPARQL queries
4. **Monitor Performance**: Track metrics to identify bottlenecks
5. **Update Knowledge Graph**: Keep symbolic knowledge current

### Error Handling

```python
try:
    results = await retriever.retrieve(query)
except SPARQLEndpointError as e:
    logger.error(f"SPARQL endpoint error: {e}")
    # Fallback to neural-only retrieval
    results = await retriever.neural_retrieve(query)
except ReasoningTimeoutError as e:
    logger.error(f"Reasoning timeout: {e}")
    # Return partial results
    results = retriever.get_partial_results()
except Exception as e:
    logger.error(f"Retrieval error: {e}")
    # Return empty results
    results = []
```

### Production Deployment

```python
# Production configuration
production_config = {
    "neural_symbolic_config": {
        "knowledge_graph_endpoint": "http://kg.prod.company.com/sparql",
        "embedding_dim": 768,
        "device": "cuda",
        "enable_reasoning": True,
        "max_reasoning_depth": 3
    },
    "performance_config": {
        "batch_size": 64,
        "cache_size_mb": 2048,
        "timeout_seconds": 30
    },
    "monitoring_config": {
        "enable_metrics": True,
        "metrics_port": 8083,
        "alert_on_errors": True
    }
}

retriever = NeuralSymbolicRetriever(**production_config["neural_symbolic_config"])
retriever.configure_performance(**production_config["performance_config"])
retriever.configure_monitoring(**production_config["monitoring_config"])

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "sparql_connected": await retriever.check_sparql_connection(),
        "gpu_available": retriever.gpu_available,
        "cache_size": retriever.get_cache_size(),
        "avg_latency": retriever.get_average_latency()
    }
```
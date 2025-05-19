# Document Management with Apache Jena Integration

The AIQ Toolkit includes a comprehensive document management system that integrates with Apache Jena for triple store capabilities, providing both traditional document storage and semantic web features.

## Overview

The document management system combines:
- Traditional document storage and retrieval
- Apache Jena triple store for RDF/OWL representations
- Vector embeddings for similarity search
- SPARQL querying capabilities
- Dublin Core metadata support
- AI/ML research output storage

## Architecture

```
┌─────────────────────┐
│  Document Upload    │
│  (PDF, Web, Text)   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Document Manager   │
└──────────┬──────────┘
           │
     ┌─────┴─────┐
     │           │
     ▼           ▼
┌─────────┐ ┌────────────┐
│  Jena   │ │   Vector   │
│  Store  │ │   Store    │
└─────────┘ └────────────┘
     │           │
     └─────┬─────┘
           │
           ▼
┌─────────────────────┐
│  SPARQL & Vector    │
│     Search          │
└─────────────────────┘
```

## Key Components

### 1. Document Manager

The main orchestrator that handles:
- Document upload and processing
- Metadata generation
- Vector embedding creation
- Jena triple store integration

```python
from aiq.document_management.manager import DocumentManager

# Initialize
doc_manager = DocumentManager()

# Upload document
document = doc_manager.upload_document(
    file_path="research.pdf",
    metadata={
        'title': 'AI Research Paper',
        'creator': 'John Doe',
        'subject': ['AI', 'Machine Learning']
    }
)
```

### 2. Jena Triple Store

Provides semantic web capabilities:
- RDF/OWL storage
- SPARQL querying
- Knowledge graph representation

```python
# SPARQL query example
query = """
PREFIX aiq: <http://aiqtoolkit.org/ontology#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>

SELECT ?doc ?title ?creator
WHERE {
    ?doc a aiq:Document ;
         dc:title ?title ;
         dc:creator ?creator .
}
"""

results = doc_manager.jena_store.sparql_query(query)
```

### 3. Vector Search

Enables semantic similarity search:
- Document embeddings
- Query embeddings
- Similarity ranking

```python
# Vector search
search_results = doc_manager.search_documents(
    query="neural networks and deep learning",
    limit=10
)
```

## Features

### Dublin Core Metadata

All documents are catalogued with Dublin Core metadata:
- Title, Creator, Subject, Description
- Date, Type, Format, Language
- Rights, Coverage, Relation

### Research Output Storage

Store AI/ML analysis results:
```python
research_output = doc_manager.store_research_output(
    agent_name="DocumentAnalyzer",
    output_data={
        'type': 'analysis',
        'content': 'Document analysis results',
        'findings': ['key finding 1', 'key finding 2']
    },
    source_doc_id=document.id
)
```

### Document Types

Supported document types:
- PDF documents
- Excel spreadsheets
- Text files
- Web pages
- JSON data
- Research outputs

## Configuration

The system can be configured via environment variables:

```bash
# Jena Fuseki endpoints
JENA_SPARQL_ENDPOINT=http://localhost:3030/dataset/sparql
JENA_UPDATE_ENDPOINT=http://localhost:3030/dataset/update
JENA_DATA_ENDPOINT=http://localhost:3030/dataset/data

# Vector store endpoint
VECTOR_STORE_ENDPOINT=http://localhost:8001/vectors

# Document storage path
DOCUMENT_STORAGE_PATH=/var/aiq/documents
```

## API Endpoints

The document management system provides REST API endpoints:

- `POST /api/documents/upload` - Upload single document
- `POST /api/documents/upload-batch` - Upload multiple documents
- `GET /api/documents/{id}` - Get document details
- `POST /api/documents/search` - Search documents
- `POST /api/sparql/query` - Execute SPARQL query
- `POST /api/research/store` - Store research output
- `GET /api/metadata/dublin-core/{id}` - Get Dublin Core metadata

## Example Usage

Complete example of using the document management system:

```python
import asyncio
from aiq.document_management.manager import DocumentManager

async def main():
    # Initialize
    doc_manager = DocumentManager()
    
    # Upload document
    doc = doc_manager.upload_document(
        "research_paper.pdf",
        metadata={
            'title': 'Advanced AI Techniques',
            'creator': 'Research Team',
            'subject': ['AI', 'Neural Networks']
        }
    )
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # SPARQL query
    results = doc_manager.jena_store.sparql_query("""
        SELECT ?doc ?title
        WHERE {
            ?doc dc:title ?title .
            ?doc dc:subject "AI" .
        }
    """)
    
    # Vector search
    similar_docs = doc_manager.search_documents(
        "artificial intelligence applications"
    )
    
    # Store research output
    output = doc_manager.store_research_output(
        agent_name="Analyzer",
        output_data={'summary': 'Key findings...'},
        source_doc_id=doc.id
    )

asyncio.run(main())
```

## Best Practices

1. **Metadata Quality**: Always provide comprehensive Dublin Core metadata
2. **Document Organization**: Use consistent subject taxonomies
3. **SPARQL Optimization**: Use prefixes and limit results
4. **Vector Search**: Combine with SPARQL for hybrid search
5. **Research Outputs**: Link outputs to source documents

## Troubleshooting

Common issues and solutions:

1. **Jena Connection Failed**
   - Check Fuseki is running
   - Verify endpoints are correct
   - Check network connectivity

2. **Vector Search Not Working**
   - Ensure documents are fully processed
   - Check embedder configuration
   - Verify vector store is running

3. **SPARQL Queries Return Empty**
   - Check namespace prefixes
   - Verify data is loaded in Jena
   - Test with simpler queries first

## Next Steps

- Set up Apache Jena Fuseki server
- Configure vector store
- Design your document ontology
- Implement custom metadata schemas
- Create specialized search interfaces
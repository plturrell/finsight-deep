# AIQ Document Management System

A comprehensive document management system integrated with NVIDIA RAG, Apache Jena, and Dublin Core metadata standards.

## Features

- **Multi-format Document Support**: PDF, Excel, Web pages, JSON, and text files
- **NVIDIA RAG Integration**: Advanced vector search and retrieval with NVIDIA embeddings
- **Apache Jena Triple Store**: OWL/RDF storage with SPARQL query support
- **Dublin Core Metadata**: Automated metadata cataloging following DC standards
- **Web Crawler**: Deep research data collection from websites
- **Research Output Storage**: Store and catalog AI-generated research outputs
- **RESTful API**: Complete API for frontend integration

## Architecture

```
document_management/
├── __init__.py            # Module initialization
├── models.py              # Data models (Document, Metadata, etc.)
├── manager.py             # Core document manager
├── embedder.py            # NVIDIA embedding integration
├── nvidia_rag_integration.py  # Advanced RAG features
├── jena_integration.py    # Apache Jena triple store
├── crawler.py             # Web crawler for research
├── metadata_catalog.py    # Dublin Core metadata cataloger
├── api.py                 # FastAPI endpoints
└── register.py            # AIQ integration
```

## Quick Start

### 1. Configure the System

```yaml
# config.yml
name: document_management
components:
  functions:
    - name: doc_upload
      type: document_upload
      config:
        enable_rag: true
        auto_metadata: true
```

### 2. Upload Documents

```python
from aiq.document_management import DocumentManager

manager = DocumentManager()
document = manager.upload_document(
    file_path="research_paper.pdf",
    metadata={
        "creator": "John Doe",
        "subject": ["machine learning", "AI"]
    }
)
```

### 3. Search with NVIDIA RAG

```python
from aiq.document_management import NVIDIARAGDocumentManager

rag_manager = NVIDIARAGDocumentManager()
results = await rag_manager.advanced_search(
    query="neural network optimization",
    search_type="hybrid"
)
```

### 4. Query with SPARQL

```python
query = """
PREFIX dc: <http://purl.org/dc/elements/1.1/>
SELECT ?title ?creator ?date
WHERE {
    ?doc dc:title ?title ;
         dc:creator ?creator ;
         dc:date ?date .
    FILTER(CONTAINS(?title, "GPU"))
}
"""
results = manager.jena_store.sparql_query(query)
```

## API Endpoints

### Document Upload
```
POST /api/documents/upload
Content-Type: multipart/form-data

Parameters:
- file: File to upload
- title: Optional document title
- creator: Optional creator name
- description: Optional description
- subject: Optional subject keywords
```

### Document Search
```
POST /api/documents/search
Content-Type: application/json

{
    "query": "search query",
    "limit": 10,
    "search_type": "hybrid",
    "filters": {
        "type": ["pdf", "research_output"],
        "date_range": "2024-01-01/2024-12-31"
    }
}
```

### Web Crawling
```
POST /api/documents/crawl
Content-Type: application/json

{
    "url": "https://example.com",
    "max_depth": 3,
    "max_pages": 100,
    "keywords": ["AI", "machine learning"]
}
```

### Research Output Storage
```
POST /api/research/store
Content-Type: application/json

{
    "agent_name": "research_agent",
    "output_data": {
        "type": "analysis",
        "content": "Research findings..."
    },
    "source_document_id": "uuid-here"
}
```

## Dublin Core Metadata

The system automatically generates Dublin Core metadata for all documents:

- **title**: Document title
- **creator**: Author or creator
- **subject**: Keywords and topics
- **description**: Document description
- **publisher**: Publishing entity
- **date**: Creation or publication date
- **type**: Document type/genre
- **format**: MIME type or format
- **identifier**: Unique identifier
- **source**: Original source
- **language**: Language code
- **coverage**: Spatial/temporal coverage
- **rights**: Copyright information

## NVIDIA RAG Features

### Intelligent Chunking
- Semantic boundary detection
- Document-type specific chunking
- Context-aware chunk generation

### Hybrid Search
- Combines semantic and keyword search
- Cross-collection searching
- NVIDIA reranking for relevance

### Vector Storage
- NVIDIA embeddings (nvidia-embed-qa-4)
- Similarity search with GPU acceleration
- Multi-modal embedding support

## Apache Jena Integration

### Triple Store
- OWL/RDF representation
- SPARQL query interface
- Linked data support

### Vector-Triple Hybrid
- Store embeddings alongside triples
- Query by semantics and structure
- Knowledge graph integration

## Requirements

- Python 3.8+
- Apache Jena Fuseki (for triple store)
- NVIDIA NIM (for embeddings)
- NVIDIA RAG service
- PostgreSQL or MongoDB (optional, for persistence)

## Environment Variables

```bash
# NVIDIA Services
NVIDIA_RAG_URL=http://localhost:8080/v1/retrieval
NVIDIA_NIM_URL=http://localhost:8000

# Apache Jena
JENA_SPARQL_ENDPOINT=http://localhost:3030/dataset/sparql
JENA_UPDATE_ENDPOINT=http://localhost:3030/dataset/update

# Storage
DOCUMENT_STORAGE_PATH=/data/aiq_documents
```

## Docker Deployment

```yaml
# docker-compose.yml
version: '3.8'

services:
  document-management:
    image: aiqtoolkit/document-management:latest
    environment:
      - NVIDIA_RAG_URL=http://nvidia-rag:8080/v1/retrieval
      - JENA_SPARQL_ENDPOINT=http://jena:3030/aiq/sparql
    volumes:
      - ./documents:/data/aiq_documents
    ports:
      - "8000:8000"

  jena-fuseki:
    image: stain/jena-fuseki
    environment:
      - ADMIN_PASSWORD=admin
    ports:
      - "3030:3030"
    volumes:
      - ./jena-data:/fuseki

  nvidia-rag:
    image: nvcr.io/nvidia/rag-service:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    ports:
      - "8080:8080"
```

## Example Workflow

```python
import asyncio
from aiq.document_management import (
    DocumentManager,
    NVIDIARAGDocumentManager,
    ResearchWebCrawler
)

async def research_workflow():
    # Initialize components
    manager = DocumentManager()
    rag = NVIDIARAGDocumentManager()
    crawler = ResearchWebCrawler()
    
    # 1. Crawl research papers
    documents = await crawler.crawl_site(
        "https://arxiv.org",
        keywords=["transformer", "attention"]
    )
    
    # 2. Upload and process documents
    for doc in documents:
        uploaded = manager.upload_web_content(
            doc.url,
            doc.content,
            doc.metadata.to_dict()
        )
        
        # 3. Ingest into NVIDIA RAG
        await rag.ingest_document_with_rag(uploaded)
    
    # 4. Search for specific topics
    results = await rag.advanced_search(
        "self-attention mechanisms in transformers",
        search_type="hybrid"
    )
    
    # 5. Generate summaries
    for result in results[:5]:
        summary = await rag.generate_rag_based_summary(
            result["document_id"]
        )
        
        # 6. Store as research output
        manager.store_research_output(
            agent_name="summary_agent",
            output_data={"summary": summary},
            source_doc_id=result["document_id"]
        )
    
    return results

# Run the workflow
asyncio.run(research_workflow())
```

## Contributing

See the main AIQToolkit contributing guidelines. Key areas for contribution:

1. Additional file format support
2. Enhanced metadata extraction
3. Improved chunking strategies
4. Custom embedding models
5. Advanced search algorithms

## License

Apache 2.0 - See LICENSE file for details.